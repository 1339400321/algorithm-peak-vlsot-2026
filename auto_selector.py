"""Prompt-to-vehicle ranking with no video-ID rules or manual inference anchors."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from auto_prompt import frame_segment, parse_prompt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "VLSOT-dataset" / "dataset"
WEIGHTS = {
    "balanced": {"detection": .10, "color": .25, "type": .16, "distance": .24, "region": .22, "visible": .03},
    "color": {"detection": .07, "color": .38, "type": .14, "distance": .18, "region": .20, "visible": .03},
    "type_aware": {"detection": .06, "color": .22, "type": .45, "distance": .12, "region": .12, "visible": .03},
    "space": {"detection": .08, "color": .19, "type": .14, "distance": .30, "region": .26, "visible": .03},
}


def records(split: str) -> dict[str, dict]:
    return {str(r["videoID"]): r for path in (DATA / split / "json").glob("*.json")
            for r in json.loads(path.read_text(encoding="utf8"))}


def frame_name(record: dict, frame: int) -> str:
    value = record[f"frame{frame}"]
    return value[0] if isinstance(value, list) else value


def overlap_fraction(box: list[float], label: list[float]) -> float:
    width = max(0., min(box[2], label[2]) - max(box[0], label[0]))
    height = max(0., min(box[3], label[3]) - max(box[1], label[1]))
    return width * height / max(1., (label[2] - label[0]) * (label[3] - label[1]))


def body_color(image: Image.Image, box: list[float]) -> dict[str, float]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    crop = image.crop((max(0, int(x1 + .15 * width)), max(0, int(y1 + .22 * height)),
                       min(image.width, int(x2 - .15 * width)), min(image.height, int(y2 - .12 * height))))
    if crop.width < 2 or crop.height < 2:
        return {name: 0. for name in ("white", "black", "silver grey", "red", "yellow orange", "blue")}
    hsv = np.asarray(crop.resize((32, 32)).convert("HSV"), dtype=float) / 255
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    neutral = saturation < .22
    scores = {
        "white": float(np.mean(neutral & (value > .68))),
        "black": float(np.mean(value < .38)),
        "silver grey": float(np.mean(neutral & (value >= .35) & (value <= .75))),
        "red": float(np.mean(((hue < .06) | (hue > .94)) & (saturation > .38) & (value > .20))),
        "yellow orange": float(np.mean((hue >= .05) & (hue < .19) & (saturation > .35) & (value > .30))),
        "blue": float(np.mean((hue >= .52) & (hue < .75) & (saturation > .25))),
    }
    return scores


def region_score(category: str, x: float, y: float) -> float:
    if category.startswith("upper"):
        vertical = 1. - y
    elif category.startswith("lower"):
        vertical = y
    else:
        vertical = max(0., 1. - abs(y - .5) / .5)
    if category.endswith("left"):
        horizontal = 1. - x
    elif category.endswith("right"):
        horizontal = x
    else:
        horizontal = max(0., 1. - abs(x - .5) / .5)
    return .55 * vertical + .45 * horizontal


def features(image: Image.Image, box: list[float], cls: str, confidence: float,
             record: dict, frame: int) -> dict[str, float]:
    info = parse_prompt(record["description"])
    category = frame_segment(info, frame)[0]
    distance = float(np.interp(frame, (0, 14, 29), info["distances"][:3])) if len(info["distances"]) >= 3 else float(info["distances"][0])
    want_color = str(record.get("color", info["attrs"][1])).lower().replace("yellow-orange", "yellow orange")
    want_type = str(record.get("type", info["attrs"][0])).lower()
    x = (box[0] + box[2]) / (2 * image.width)
    y = (box[1] + box[3]) / (2 * image.height)
    body = body_color(image, box)
    color = body.get(want_color, 0.)
    if want_color == "silver grey":
        color += .25 * body["white"]
    if want_color == "white":
        color += .15 * body["silver grey"]
    color = min(1., color * 3.5)
    aspect = (box[2] - box[0]) / max(1., box[3] - box[1])
    if want_type == "van":
        # COCO has no van class; either car or truck can represent one.
        type_match = {"car": .35, "truck": .85, "bus": .7}.get(cls, .1)
        type_match += .35 * max(0., 1. - abs(aspect - .95) / 1.0)
    elif want_type == "truck":
        type_match = 1. if cls == "truck" else (.25 if cls == "car" else .1)
    else:
        type_match = 1. if cls == want_type else .1
    expected_y = float(np.clip(.72 - (distance - 22.) * .0065, .24, .75))
    distance_match = max(0., 1. - abs(y - expected_y) / .35)
    visible = min(box[0], box[1], image.width - box[2], image.height - box[3]) >= 5
    first_clause = record["description"].lower().split("adjacent to")[0].split("closest to")[0]
    truncated = ("truncated" in first_clause and "not truncated" not in first_clause
                 and "without truncation" not in first_clause)
    visibility = float(visible != truncated)
    return {"detection": confidence, "color": color, "type": type_match,
            "distance": distance_match, "region": region_score(category, x, y), "visible": visibility}


def rank(image: Image.Image, proposals: list[dict], record: dict, frame: int,
         policy: str = "balanced") -> list[dict]:
    result = []
    for proposal in proposals:
        evidence = features(image, proposal["box"], proposal["class"], proposal["confidence"], record, frame)
        score = sum(WEIGHTS[policy][name] * evidence[name] for name in WEIGHTS[policy])
        result.append({**proposal, "score": score, "evidence": evidence})
    return sorted(result, key=lambda row: row["score"], reverse=True)


def evaluate() -> dict:
    cache = json.loads((ROOT / "auto_detector_annotated_cache.json").read_text(encoding="utf8"))
    training = records("train")
    recent = json.loads((ROOT / "user_identity_labels_corrected.json").read_text(encoding="utf8"))["boxes"]
    old = json.loads((ROOT / "manual_boxes.json").read_text(encoding="utf8"))
    excluded = set(json.loads((ROOT / "manual_annotation_policy.json").read_text(encoding="utf8"))["exclude_from_training_until_review"])
    labels = {**{f"{video}_{frame}": box for video, frames in old.items() if video not in excluded
                 for frame, box in frames.items()}, **recent}
    counters = {name: {"hits": 0, "frames": 0, "recent_hits": 0, "recent_frames": 0} for name in WEIGHTS}
    oracle = {"hits": 0, "frames": 0, "recent_hits": 0, "recent_frames": 0}
    diagnostics = []
    for key, target in labels.items():
        video, frame_string = key.rsplit("_", 1)
        frame = int(frame_string)
        record = training[video]
        image = Image.open(DATA / "train" / "img" / frame_name(record, frame)).convert("RGB")
        candidates = cache[video][frame_string]
        overlaps = [overlap_fraction(row["box"], target) for row in candidates]
        is_recent = key in recent
        oracle["frames"] += 1
        oracle["recent_frames"] += int(is_recent)
        found = bool(overlaps and max(overlaps) >= .25)
        oracle["hits"] += int(found)
        oracle["recent_hits"] += int(is_recent and found)
        row = {"video": video, "frame": frame, "oracle": found, "max_overlap": max(overlaps, default=0.)}
        for name in WEIGHTS:
            ranked = rank(image, candidates, record, frame, name)
            hit = bool(ranked and overlap_fraction(ranked[0]["box"], target) >= .25)
            counters[name]["frames"] += 1
            counters[name]["hits"] += int(hit)
            counters[name]["recent_frames"] += int(is_recent)
            counters[name]["recent_hits"] += int(is_recent and hit)
            row[name] = hit
        diagnostics.append(row)
    output = {"oracle": oracle, "policies": counters, "diagnostics": diagnostics,
              "note": "96 later user-labeled frames are independent of original 8-group calibration. Labels are rough identity marks, not accurate box edges. This is an identity diagnostic, not platform Recall@0.5."}
    (ROOT / "auto_selector_validation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        result = evaluate()
        print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}, ensure_ascii=False, indent=2))
