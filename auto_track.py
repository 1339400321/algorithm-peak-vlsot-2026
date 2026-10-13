"""Automatic text-conditioned target selection across detector frames."""
from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image

from auto_selector import DATA, ROOT, features, frame_name, overlap_fraction, records

FEATURES = ("detection", "color", "type", "distance", "region", "visible")
# Fixed generic rule weights; no fitting on validation identities.
WEIGHTS = np.asarray([.06, .22, .45, .12, .12, .03])


def score_candidates(image, proposals, record, frame):
    if not proposals:
        return np.asarray([], dtype=float)
    data = np.asarray([[features(image, proposal["box"], proposal["class"],
                                 proposal["confidence"], record, frame)[name] for name in FEATURES]
                       for proposal in proposals])
    score = data @ WEIGHTS
    return score - max(score)


def transition(previous, current, gap, state):
    a = np.asarray([proposal["box"] for proposal in previous], dtype=float)
    b = np.asarray([proposal["box"] for proposal in current], dtype=float)
    ac, bc = (a[:, :2] + a[:, 2:]) / 2, (b[:, :2] + b[:, 2:]) / 2
    aw, bw = np.maximum(a[:, 2:] - a[:, :2], 1), np.maximum(b[:, 2:] - b[:, :2], 1)
    scale = np.maximum(np.maximum(aw[:, None, :], bw[None, :, :]), 70)
    distance = np.linalg.norm((ac[:, None, :] - bc[None, :, :]) / scale, axis=2)
    size = np.abs(np.log(aw[:, None, :] / bw[None, :, :])).mean(axis=2)
    position_scale = math.sqrt(max(gap, 1)) * (1.5 if state.lower() == "moving" else 1.0)
    return np.minimum(distance / position_scale, 3) + .35 * np.minimum(size, 2)


def select(record, proposals_by_frame, split="test", weight=0.4, frames=None):
    frames = sorted(proposals_by_frame if frames is None else frames)
    image_dir = DATA / split / "img"
    valid = [frame for frame in frames if proposals_by_frame[frame]]
    if not valid:
        return {frame: None for frame in frames}
    local = {}
    for frame in valid:
        with Image.open(image_dir / frame_name(record, frame)) as handle:
            local[frame] = score_candidates(handle.convert("RGB"), proposals_by_frame[frame], record, frame)
    state = str(record.get("state", ""))
    dp = local[valid[0]].copy()
    pointers = {}
    for before, after in zip(valid, valid[1:]):
        penalty = transition(proposals_by_frame[before], proposals_by_frame[after], after - before, state)
        paths = dp[:, None] - weight * penalty
        pointers[after] = paths.argmax(axis=0)
        dp = local[after] + paths.max(axis=0)
    chosen = {valid[-1]: int(dp.argmax())}
    for before, after in zip(valid[-2::-1], valid[:0:-1]):
        chosen[before] = int(pointers[after][chosen[after]])
    return {frame: chosen.get(frame) for frame in frames}


def evaluate():
    cache = json.loads((ROOT / "auto_detector_annotated_cache.json").read_text(encoding="utf8"))
    recent = json.loads((ROOT / "user_identity_labels_corrected.json").read_text(encoding="utf8"))["boxes"]
    labels = dict(recent)
    old = json.loads((ROOT / "manual_boxes.json").read_text(encoding="utf8"))
    excluded = set(json.loads((ROOT / "manual_annotation_policy.json").read_text(encoding="utf8"))["exclude_from_training_until_review"])
    for video, boxes in old.items():
        if video not in excluded:
            labels.update({f"{video}_{frame}": box for frame, box in boxes.items()})
    training = records("train")
    results = []
    for strength in (0., .1, .2, .4, .8, 1.6):
        hits, recent_hits = [], []
        groups = {}
        for key, box in labels.items():
            video, frame = key.rsplit("_", 1)
            groups.setdefault(video, {})[int(frame)] = (key, box)
        for video, marked in groups.items():
            proposals = {int(frame): rows for frame, rows in cache[video].items()}
            selected = select(training[video], proposals, split="train", weight=strength)
            for frame, (key, box) in marked.items():
                index = selected[frame]
                hit = bool(index is not None and overlap_fraction(proposals[frame][index]["box"], box) >= .25)
                hits.append(hit)
                if key in recent:
                    recent_hits.append(hit)
        result = {"strength": strength, "top1": float(np.mean(hits)), "recent_top1": float(np.mean(recent_hits)),
                  "frames": len(hits)}
        results.append(result)
        print(result, flush=True)
    (ROOT / "auto_track_validation.json").write_text(json.dumps(results, indent=2), encoding="utf8")


if __name__ == "__main__":
    evaluate()
