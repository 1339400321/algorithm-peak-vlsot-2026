"""Export the torchvision Faster R-CNN proposal detector to ONNX."""
import argparse
from pathlib import Path

import onnx
import torch

from auto_detector_cache import load_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))
    parser.add_argument("--output", type=Path, default=Path("fasterrcnn_detector.onnx"))
    args = parser.parse_args()
    model, device = load_detector(args.weights)
    model = model.cpu().eval()
    sample = torch.zeros(3, 450, 800)
    with torch.inference_mode():
        torch.onnx.export(model, ([sample],), str(args.output), opset_version=17,
                          input_names=["image"], output_names=["boxes", "labels", "scores"],
                          do_constant_folding=True)
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    print(args.output, args.output.stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    main()
