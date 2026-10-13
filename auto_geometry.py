"""Scene-held-out, text-conditioned 3-D trajectory baseline for VLSOT.

The released test text includes the target's approximate metric distance and
dimensions. No test labels, hand-selected boxes, or video-specific policies
are loaded here. Image-conditioned target selection is handled separately.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from auto_prompt import distance_at, frame_segment, parse_prompt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "VLSOT-dataset" / "dataset"
ANCHORS = (0, 14, 29)
CATS = ("lower_center", "lower_left", "lower_right", "middle_center", "middle_left",
        "middle_right", "upper_center", "upper_left", "upper_right")
TYPES = ("car", "van", "truck", "bus")
COLORS = ("white", "black", "silver grey", "red", "blue", "yellow orange")


def records(split: str) -> list[dict]:
    return [record for path in sorted((DATA / split / "json").glob("*.json"))
            for record in json.loads(path.read_text(encoding="utf8"))]


def query(record: dict, frame: int) -> tuple[np.ndarray, float, list[float]]:
    info = parse_prompt(record["description"])
    cat, _, _, order, _ = frame_segment(info, frame)
    typ, color, state, az = info["attrs"]
    dist = distance_at(info["distances"], frame)
    dims = info["dims"]
    feature = [math.log(max(dist, 1.0)) / 2, float(order) / 4,
               math.sin(math.radians(az)), math.cos(math.radians(az)),
               *(d / 6 for d in dims), frame / 29]
    feature += [float(cat == value) for value in CATS]
    feature += [float(typ == value) for value in TYPES]
    feature += [float(color == value) for value in COLORS]
    feature += [float(state == "moving")]
    return np.asarray(feature, dtype=np.float64), dist, dims


def training_rows(train: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    for record in train:
        for frame in ANCHORS:
            feature, dist, _ = query(record, frame)
            label = np.asarray(record[f"frame{frame}"][4:7], dtype=np.float64)
            features.append(feature)
            targets.append(label[1:] / max(dist, 1.0))
    return np.stack(features), np.stack(targets)


def predict(record: dict, train_x: np.ndarray, train_y: np.ndarray,
            k: int, mode: str) -> np.ndarray:
    scale = np.ones(train_x.shape[1])
    # These are broad semantic cues; only the region and target distance are
    # trusted strongly. Vehicle identity and scene-specific trajectory IDs
    # never enter the feature vector.
    if mode == "region_distance":
        scale[:8] = [1.0, .08, .0, .0, .0, .0, .0, .0]
        scale[8:17] = 1.0
        scale[17:] = 0.0
    elif mode == "attributes":
        scale[:8] = [1.0, .2, .2, .2, .1, .1, .1, .15]
        scale[8:17] = 1.0
        scale[17:] = .15
    else:
        raise ValueError(mode)
    ratios = []
    for frame in ANCHORS:
        feature, _, _ = query(record, frame)
        distances = np.sqrt(np.mean(((train_x - feature) * scale) ** 2, axis=1))
        ids = np.argsort(distances)[:k]
        weights = 1.0 / np.maximum(distances[ids], .025) ** 2
        ratios.append(np.average(train_y[ids], axis=0, weights=weights))
    interpolated = np.column_stack([
        np.interp(np.arange(30), ANCHORS, np.asarray(ratios)[:, axis]) for axis in range(2)
    ])
    output = []
    for frame, ratio in enumerate(interpolated):
        _, distance, _ = query(record, frame)
        y, z = ratio * distance
        yz = y * y + z * z
        if yz > distance * distance - .25:
            factor = math.sqrt(max(distance * distance - .25, 0.0) / yz)
            y *= factor
            z *= factor
        x = math.sqrt(max(distance * distance - y * y - z * z, .25))
        output.append([x, y, z])
    return np.asarray(output)


def evaluate() -> dict:
    all_records = records("train")
    scenes = sorted(set(str(r["sequence_id"]) for r in all_records))
    folds = [{scene for i, scene in enumerate(scenes) if i % 5 == fold} for fold in range(5)]
    results = []
    for mode in ("region_distance", "attributes"):
        for k in (1, 3, 5, 10, 20):
            errors = []
            per_scene = {}
            for holdout in folds:
                train = [r for r in all_records if str(r["sequence_id"]) not in holdout]
                validation = [r for r in all_records if str(r["sequence_id"]) in holdout]
                x, y = training_rows(train)
                for record in validation:
                    pred = predict(record, x, y, k, mode)
                    truth = np.asarray([record[f"frame{i}"][4:7] for i in range(30)], dtype=float)
                    err = np.linalg.norm(pred - truth, axis=1)
                    errors.extend(err.tolist())
                    per_scene.setdefault(str(record["sequence_id"]), []).extend(err.tolist())
            e = np.asarray(errors)
            results.append({"mode": mode, "k": k, "groups": len(all_records), "frames": len(e),
                            "APE_m": float(e.mean()), "median_m": float(np.median(e)),
                            "Recall_position_lt_0.5m_proxy": float(np.mean(e < .5)),
                            "p90_m": float(np.percentile(e, 90)),
                            "scene_mean_m": {s: float(np.mean(v)) for s, v in per_scene.items()}})
            print(mode, k, f"APE={e.mean():.3f}", f"R0.5={np.mean(e < .5):.3f}", flush=True)
    best = min(results, key=lambda r: (r["APE_m"], -r["Recall_position_lt_0.5m_proxy"]))
    output = {"protocol": "Five-fold scene-held-out; no images or test labels used in this text-only 3-D baseline",
              "best_by_APE": best, "results": results}
    (ROOT / "auto_geometry_scene_cv.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        result = evaluate()
        print(json.dumps({k: v for k, v in result["best_by_APE"].items() if k != "scene_mean_m"}, indent=2))
