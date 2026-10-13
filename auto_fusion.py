"""Visual geometry conditioning, evaluated with whole scenes held out."""
from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image

import auto_geometry as geometry
import auto_track
from auto_selector import DATA, ROOT, frame_name, records

ANCHORS = (0, 14, 29)


def box_feature(box, width=1920., height=1080.):
    if box is None:
        return np.asarray([.5, .5, 0., 0., 0.])
    return np.asarray([(box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height),
                       (box[2] - box[0]) / width, (box[3] - box[1]) / height, 1.])


def build_features(split):
    cache = json.loads((ROOT / f"auto_detector_{split}_cache.json").read_text(encoding="utf8"))
    features = {}
    for video, record in records(split).items():
        proposals = {frame: cache[video][str(frame)] for frame in ANCHORS}
        selected = auto_track.select(record, proposals, split=split, weight=.1)
        features[video] = {str(frame): proposals[frame][selected[frame]]["box"]
                           if selected[frame] is not None else None for frame in ANCHORS}
        if len(features) % 25 == 0:
            print(split, "automatic tracks", len(features), flush=True)
    (ROOT / f"auto_fusion_{split}_features.json").write_text(json.dumps(features), encoding="utf8")
    return features


def train_rows(items, boxes):
    x, y, owners = [], [], []
    for record in items:
        video = str(record["videoID"])
        for frame in ANCHORS:
            query, distance, _ = geometry.query(record, frame)
            x.append(np.r_[query, box_feature(boxes[video][str(frame)])])
            y.append(np.asarray(record[f"frame{frame}"][4:7], dtype=float)[1:] / distance)
            owners.append((str(record["sequence_id"]), str(record["track_id"])))
    return np.asarray(x), np.asarray(y), owners


def predict(record, boxes, x, y, visual_weight=.5, k=20):
    text_size = x.shape[1] - 5
    scale = np.ones(x.shape[1])
    scale[:8] = [1., .2, .2, .2, .1, .1, .1, .15]
    scale[8:17] = 1.
    scale[17:text_size] = .15
    scale[text_size:] = [visual_weight * 3, visual_weight * 3, visual_weight, visual_weight, .05]
    ratios = []
    for frame in ANCHORS:
        feature, _, _ = geometry.query(record, frame)
        query = np.r_[feature, box_feature(boxes[str(frame)])]
        distances = np.sqrt(np.mean(((x - query) * scale) ** 2, axis=1))
        indices = np.argsort(distances)[:k]
        weights = 1 / np.maximum(distances[indices], .025) ** 2
        ratios.append(np.average(y[indices], axis=0, weights=weights))
    ratio = np.column_stack([np.interp(np.arange(30), ANCHORS, np.asarray(ratios)[:, axis]) for axis in range(2)])
    output = []
    for frame in range(30):
        _, distance, _ = geometry.query(record, frame)
        yz = ratio[frame] * distance
        norm = np.linalg.norm(yz)
        limit = np.sqrt(max(distance ** 2 - .25, 0.))
        if norm > limit:
            yz *= limit / norm
        output.append([np.sqrt(max(distance ** 2 - np.sum(yz ** 2), .25)), *yz])
    return np.asarray(output)


def evaluate(selected=False):
    items = geometry.records("train")
    path = ROOT / "auto_fusion_train_features.json"
    boxes = build_features("train") if selected else (json.loads(path.read_text()) if path.exists() else build_features("train"))
    x, y, owners = train_rows(items, boxes)
    scenes = sorted({str(record["sequence_id"]) for record in items})
    results = []
    for visual_weight in ((1.,) if selected else (0., .1, .25, .5, 1., 2.)):
        for k in ((5,) if selected else (5, 20)):
            errors = []
            for fold in range(5):
                held = set(scenes[fold::5])
                mask = np.asarray([owner[0] not in held for owner in owners])
                for record in items:
                    if str(record["sequence_id"]) not in held:
                        continue
                    prediction = predict(record, boxes[str(record["videoID"])], x[mask], y[mask], visual_weight, k)
                    truth = np.asarray([record[f"frame{i}"][4:7] for i in range(30)], dtype=float)
                    errors.extend(np.linalg.norm(prediction - truth, axis=1).tolist())
            errors = np.asarray(errors)
            result = {"visual_weight": visual_weight, "k": k, "APE_m": float(errors.mean()),
                      "position_under_0_5m_proxy": float(np.mean(errors < .5)), "frames": len(errors)}
            results.append(result)
            print(result, flush=True)
    report = {"protocol": "5-fold sequence_id holdout; automatic detector + fixed color rule + temporal selection; no manual boxes at inference",
              "results": results, "best_by_APE": min(results, key=lambda row: row["APE_m"])}
    (ROOT / ("auto_fusion_scene_cv_selected.json" if selected else "auto_fusion_scene_cv.json")).write_text(json.dumps(report, indent=2), encoding="utf8")
    np.savez(ROOT / "auto_geometry_model.npz", x=x, y=y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", action="store_true")
    evaluate(parser.parse_args().selected)
