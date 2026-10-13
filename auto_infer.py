"""Reproducible, fully automatic VLSOT inference and platform ZIP export."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

import auto_geometry
import auto_fusion
import auto_selector
import auto_track
from auto_detector_cache import CLASSES, WEIGHTS, load_detector
from auto_prompt import parse_prompt


def candidates(image, model, device):
    with torch.inference_mode():
        prediction = model([to_tensor(image).to(device)])[0]
    output = []
    for box, label, score in zip(prediction["boxes"].cpu(), prediction["labels"].cpu(),
                                 prediction["scores"].cpu()):
        name = CLASSES.get(int(label))
        if name is not None and float(score) >= .18:
            output.append({"box": [round(float(value), 2) for value in box.tolist()],
                           "class": name, "confidence": round(float(score), 4)})
    return output


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Path to VLSOT-dataset/dataset")
    parser.add_argument("--detector-weights", type=Path, default=WEIGHTS)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--cache", type=Path, help="Optional automatic detector cache; missing frames are inferred")
    parser.add_argument("--train-cache", type=Path, help="Optional automatic training detector cache")
    parser.add_argument("--geometry-model", type=Path, help="Pretrained geometry prototypes (.npz); test inference needs no training split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=0, help="Smoke test only")
    args = parser.parse_args()
    data = args.dataset.resolve()
    if not (data / args.split / "json").is_dir():
        parser.error("Dataset split JSON directory missing")
    args.output.mkdir(parents=True, exist_ok=True)
    auto_selector.DATA = data
    auto_geometry.DATA = data
    auto_track.DATA = data
    training = [] if args.geometry_model and args.split == "test" else auto_geometry.records("train")
    records = auto_geometry.records(args.split)
    if args.max_groups:
        records = records[:args.max_groups]
    cache = json.loads(args.cache.read_text(encoding="utf8")) if args.cache and args.cache.exists() else {}
    train_cache = json.loads(args.train_cache.read_text(encoding="utf8")) if args.train_cache and args.train_cache.exists() else {}
    model = device = None
    if args.geometry_model and args.split == "test":
        with np.load(args.geometry_model) as saved:
            train_x, train_y = saved["x"], saved["y"]
        owners = []
    else:
        train_boxes = {}
        for record in training:
            video = str(record["videoID"])
            anchors = {}
            for frame in (0, 14, 29):
                cached = train_cache.get(video, {}).get(str(frame))
                if cached is not None:
                    anchors[frame] = cached
                else:
                    if model is None:
                        model, device = load_detector(args.detector_weights)
                    with Image.open(data / "train" / "img" / auto_selector.frame_name(record, frame)) as handle:
                        anchors[frame] = candidates(handle.convert("RGB"), model, device)
            indices = auto_track.select(record, anchors, split="train", weight=.1)
            train_boxes[video] = {str(frame): anchors[frame][indices[frame]]["box"]
                                  if indices[frame] is not None else None for frame in (0, 14, 29)}
        train_x, train_y, owners = auto_fusion.train_rows(training, train_boxes)
    visual_dir = args.output / "preview"
    visual_dir.mkdir(exist_ok=True)
    groups = []
    json_groups = []
    result_lines = {}
    for record in records:
        video = str(record["videoID"])
        proposals = {}
        for frame in range(30):
            cached = cache.get(video, {}).get(str(frame))
            if cached is not None:
                proposals[frame] = cached
            else:
                if model is None:
                    model, device = load_detector(args.detector_weights)
                with Image.open(data / args.split / "img" / auto_selector.frame_name(record, frame)) as handle:
                    proposals[frame] = candidates(handle.convert("RGB"), model, device)
        indices = auto_track.select(record, proposals, split=args.split, weight=.1)
        # Match the three-anchor selection used during geometry validation.
        anchor_indices = auto_track.select(record, {frame: proposals[frame] for frame in (0, 14, 29)},
                                           split=args.split, weight=.1)
        anchor_boxes = {str(frame): proposals[frame][anchor_indices[frame]]["box"]
                        if anchor_indices[frame] is not None else None for frame in (0, 14, 29)}
        prompt = parse_prompt(record["description"])
        height, length, width = prompt["dims"]
        if args.split == "train":
            # Training split diagnostics must exclude the held-out target identity.
            identity = (record["sequence_id"], record["track_id"])
            mask = np.asarray([owner != identity for owner in owners])
            position = auto_fusion.predict(record, anchor_boxes, train_x[mask], train_y[mask],
                                           visual_weight=1.0, k=3)
        else:
            position = auto_fusion.predict(record, anchor_boxes, train_x, train_y,
                                           visual_weight=1.0, k=3)
        lines = []
        boxes = []
        json_frames = []
        for frame in range(30):
            selected = proposals[frame][indices[frame]] if indices[frame] is not None else None
            boxes.append(selected["box"] if selected else None)
            xyz = position[frame]
            values = [length, width, height, *xyz]
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"Non-finite output for {video} frame {frame}")
            lines.append(f"{frame + 1} " + " ".join(f"{float(value):.6f}" for value in values))
            json_frames.append({"frame_id": frame + 1, "target_id": str(record["track_id"]),
                                "length": float(length), "width": float(width), "height": float(height),
                                "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]),
                                "confidence": float(selected["confidence"]) if selected else 0.0})
            if frame in (0, 14, 29):
                with Image.open(data / args.split / "img" / auto_selector.frame_name(record, frame)) as handle:
                    image = handle.convert("RGB")
                if selected:
                    ImageDraw.Draw(image).rectangle(selected["box"], outline="#16c97d", width=9)
                image.thumbnail((800, 450))
                image.save(visual_dir / f"{video}_{frame:02d}.jpg", quality=88)
        result_lines[f"result/{video}_pred_box0_.txt"] = "\n".join(lines) + "\n"
        groups.append({"video_id": video, "track_id_from_input": record["track_id"],
                       "frames": 30, "detected_frames": sum(box is not None for box in boxes),
                       "selected_boxes": boxes})
        json_groups.append({"video_id": video, "target_id": str(record["track_id"]),
                            "predictions": json_frames})
        print(video, "selected", groups[-1]["detected_frames"], "/ 30", flush=True)
    archive = args.output / "automatic_result.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for name, content in sorted(result_lines.items()):
            handle.writestr(name, content)
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise ValueError("ZIP CRC failed")
    (args.output / "prediction.json").write_text(json.dumps({"groups": json_groups}, ensure_ascii=False, indent=2),
                                                   encoding="utf8")
    manifest = {"pipeline": "automatic_text_detector_temporal_geometry_v1",
                "manual_test_boxes_used": False, "test_id_special_cases": False,
                "groups": groups, "zip_sha256": sha256(archive), "zip_files": sorted(result_lines)}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps({"zip": str(archive), "sha256": manifest["zip_sha256"],
                      "groups": len(groups), "frames": sum(group["frames"] for group in groups)}, indent=2))


if __name__ == "__main__":
    main()
