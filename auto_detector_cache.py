"""Cache COCO vehicle proposals without using target annotations at inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "VLSOT-dataset" / "dataset"
WEIGHTS = ROOT / "torch_cache" / "hub" / "checkpoints" / "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
CLASSES = {3: "car", 6: "bus", 8: "truck"}


def load_detector(weight_path=WEIGHTS):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    model.transform.min_size = (500,)
    model.transform.max_size = 1000
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--annotated-only", action="store_true")
    parser.add_argument("--anchors-only", action="store_true")
    args = parser.parse_args()
    if args.annotated_only and args.split != "train":
        parser.error("--annotated-only only applies to training frames")
    labels = json.loads((ROOT / "user_identity_labels_corrected.json").read_text(encoding="utf8"))["boxes"] if args.annotated_only else {}
    needed = {}
    for key in labels:
        video, frame = key.rsplit("_", 1)
        needed.setdefault(video, set()).add(int(frame))
    if args.annotated_only:
        manual = json.loads((ROOT / "manual_boxes.json").read_text(encoding="utf8"))
        policy = json.loads((ROOT / "manual_annotation_policy.json").read_text(encoding="utf8"))
        excluded = set(policy["exclude_from_training_until_review"])
        for video, frame_boxes in manual.items():
            if video not in excluded:
                needed.setdefault(video, set()).update(map(int, frame_boxes))
    out = ROOT / ("auto_detector_annotated_cache.json" if args.annotated_only else f"auto_detector_{args.split}_cache.json")
    cache = json.loads(out.read_text(encoding="utf8")) if out.exists() else {}
    model, device = load_detector()
    completed = 0
    for path in sorted((DATA / args.split / "json").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf8"))[0]
        video = str(record["videoID"])
        if args.annotated_only and video not in needed:
            continue
        frames = sorted(needed[video]) if args.annotated_only else ((0, 14, 29) if args.anchors_only else range(30))
        cache.setdefault(video, {})
        for frame in frames:
            if str(frame) in cache[video]:
                continue
            value = record[f"frame{frame}"]
            image_name = value[0] if isinstance(value, list) else value
            with Image.open(DATA / args.split / "img" / image_name) as handle:
                image = handle.convert("RGB")
            with torch.inference_mode():
                prediction = model([to_tensor(image).to(device)])[0]
            proposals = []
            for box, label, score in zip(prediction["boxes"].cpu(), prediction["labels"].cpu(), prediction["scores"].cpu()):
                cls = CLASSES.get(int(label))
                if cls is None or float(score) < .18:
                    continue
                proposals.append({"box": [round(float(x), 2) for x in box.tolist()],
                                  "class": cls, "confidence": round(float(score), 4)})
            cache[video][str(frame)] = proposals
        out.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf8")
        completed += 1
        print(f"{args.split} {video}: {len(cache[video])} frames, {completed} videos", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
