import argparse
import shutil
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

from utils import build_model, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained Faster R-CNN model.")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--images-dir", help="Directory of images to run inference on")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", help="Optional directory to save annotated images")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--arch", choices=["fasterrcnn", "ssdlite"], default=None, help="Model architecture (default: from checkpoint)")
    parser.add_argument("--max-dim", type=int, default=None, help="Downscale images so the longest side is <= this value (0 disables)")
    parser.add_argument("--min-size", type=int, default=None, help="Model transform min size (default varies by arch/device)")
    parser.add_argument("--max-size", type=int, default=None, help="Model transform max size (default varies by arch/device)")
    parser.add_argument("--copy-recognized", action="store_true", help="Copy recognized images into recognized/{high,medium,low}")
    return parser.parse_args()


def infer_ssdlite_backbone_flag(state_dict) -> Optional[bool]:
    weight = state_dict.get("backbone.features.1.0.3.0.weight")
    if weight is None:
        return None
    if hasattr(weight, "shape") and len(weight.shape) >= 1:
        if weight.shape[0] == 160:
            return True
        if weight.shape[0] == 80:
            return False
    return None


def load_checkpoint(
    path: Path,
    device: torch.device,
    min_size: int,
    max_size: int,
    arch_override: Optional[str],
    ssdlite_backbone_weights: Optional[bool],
):
    checkpoint = torch.load(path, map_location="cpu")
    num_classes = checkpoint.get("num_classes")
    class_names = checkpoint.get("class_names", [])
    if num_classes is None:
        num_classes = len(class_names) + 1
    architecture = arch_override or checkpoint.get("architecture") or "fasterrcnn"

    model = build_model(
        num_classes=num_classes,
        pretrained=False,
        min_size=min_size,
        max_size=max_size,
        arch=architecture,
        ssdlite_backbone_weights=ssdlite_backbone_weights,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names, architecture, checkpoint


def list_images(images_dir: Path):
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield path


def draw_detections(image: Image.Image, detections, class_names, score_threshold: float):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(detections["boxes"], detections["labels"], detections["scores"]):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        name = class_names[label - 1] if 0 < label <= len(class_names) else str(int(label))
        text = f"{name} {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), text, fill="red")
    return image


def resize_for_inference(image: Image.Image, max_dim: int):
    if max_dim <= 0:
        return image, 1.0
    width, height = image.size
    max_side = max(width, height)
    if max_side <= max_dim:
        return image, 1.0
    scale = max_dim / float(max_side)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = F.resize(image, [new_height, new_width])
    return resized, scale


def run_inference(model, class_names, image_path: Path, device: torch.device, score_threshold: float, max_dim: int):
    original = Image.open(image_path).convert("RGB")
    resized, scale = resize_for_inference(original, max_dim)
    tensor = F.to_tensor(resized).to(device)
    with torch.no_grad():
        output = model([tensor])[0]

    detections = {
        "boxes": output["boxes"].cpu(),
        "labels": output["labels"].cpu(),
        "scores": output["scores"].cpu(),
    }

    if scale != 1.0:
        detections["boxes"] = detections["boxes"] / scale

    results = []
    for box, label, score in zip(detections["boxes"], detections["labels"], detections["scores"]):
        if score < score_threshold:
            continue
        name = class_names[label - 1] if 0 < label <= len(class_names) else str(int(label))
        results.append((name, float(score), [float(x) for x in box.tolist()]))

    return original, detections, results


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model_path = Path(args.model)
    checkpoint = torch.load(model_path, map_location="cpu")
    arch = args.arch or checkpoint.get("architecture") or "fasterrcnn"
    ssdlite_backbone_weights = checkpoint.get("ssdlite_backbone_weights")
    if arch == "ssdlite" and ssdlite_backbone_weights is None:
        ssdlite_backbone_weights = infer_ssdlite_backbone_flag(checkpoint.get("model_state", {}))

    max_dim = args.max_dim
    if max_dim is None:
        if arch == "ssdlite":
            max_dim = 320
        else:
            max_dim = 768 if device.type == "mps" else 1024

    min_size = args.min_size
    max_size = args.max_size
    if min_size is None:
        if arch == "ssdlite":
            min_size = 320
        else:
            min_size = 512 if device.type == "mps" else 800
    if max_size is None:
        if arch == "ssdlite":
            max_size = 320
        else:
            max_size = 768 if device.type == "mps" else 1333

    model, class_names, resolved_arch, _ = load_checkpoint(
        model_path,
        device,
        min_size,
        max_size,
        arch,
        ssdlite_backbone_weights,
    )
    print(
        f"Architecture: {resolved_arch} | max-dim: {max_dim} | min-size: {min_size} | max-size: {max_size} | "
        f"ssdlite_backbone_weights: {ssdlite_backbone_weights}"
    )

    if args.image is None and args.images_dir is None:
        raise SystemExit("Provide --image or --images-dir")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.images_dir:
        image_paths.extend(list_images(Path(args.images_dir)))

    for image_path in image_paths:
        print(f"Processing {image_path}")
        try:
            image, detections, results = run_inference(
                model,
                class_names,
                image_path,
                device,
                args.score_threshold,
                max_dim,
            )
        except RuntimeError as exc:
            if device.type != "mps" or max_dim <= 512:
                raise
            reduced = max(512, int(max_dim * 0.75))
            print(f"MPS error detected, retrying with --max-dim {reduced}")
            image, detections, results = run_inference(
                model,
                class_names,
                image_path,
                device,
                args.score_threshold,
                reduced,
            )
        if results:
            top_results = results[:5]
            summary = ", ".join([f"{name} ({score:.2f})" for name, score, _ in top_results])
            print(f"Top 5: {summary}")
        else:
            print("Top 5: no detections")

        if args.copy_recognized:
            recognized_root = image_path.parent / "recognized"
            dest_dir = None
            dest_name = image_path.name
            if results:
                top_label, top_score, _ = results[0]
                safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in top_label).strip("_")
                if safe_label:
                    dest_name = f"{safe_label}_{image_path.name}"
                if top_score >= 0.9:
                    dest_dir = recognized_root / "high"
                elif top_score >= 0.6:
                    dest_dir = recognized_root / "medium"
                elif top_score >= 0.3:
                    dest_dir = recognized_root / "low"
            else:
                dest_dir = recognized_root / "none"

            if dest_dir is not None:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / dest_name
                if not dest_path.exists():
                    shutil.copy2(image_path, dest_path)

        if output_dir:
            annotated = draw_detections(image, detections, class_names, args.score_threshold)
            out_path = output_dir / image_path.name
            annotated.save(out_path)


if __name__ == "__main__":
    main()
