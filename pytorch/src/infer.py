import argparse
import time
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

from utils import build_model, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained detection model.")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pth/.pt)")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--images-dir", help="Directory of images to run inference on")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", help="Optional directory to save annotated images")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument(
        "--arch",
        choices=["fasterrcnn", "ssdlite", "yolo"],
        default=None,
        help="Model architecture (default: from checkpoint)",
    )
    parser.add_argument("--max-dim", type=int, default=None, help="Downscale images so the longest side is <= this value (0 disables)")
    parser.add_argument("--min-size", type=int, default=None, help="Model transform min size (default varies by arch/device)")
    parser.add_argument("--max-size", type=int, default=None, help="Model transform max size (default varies by arch/device)")
    parser.add_argument(
        "--box",
        action="store_true",
        help="Copy images into detected/empty subfolders and draw class-colored boxes on detections",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Copy images into subfolders named by the count of unique labels detected (0, 1, 2, etc.)",
    )
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


def yolo_device_arg(device: torch.device) -> str:
    if device.type == "cuda":
        return "0"
    if device.type == "mps":
        return "mps"
    return "cpu"


def load_yolo_model(path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required for YOLO inference. Install with: pip install ultralytics") from exc
    return YOLO(str(path))


def run_yolo_inference(model, image_path: Path, score_threshold: float, imgsz: int, device_arg: str):
    original = Image.open(image_path).convert("RGB")
    yolo_results = model.predict(
        source=str(image_path),
        conf=score_threshold,
        imgsz=imgsz,
        device=device_arg,
        verbose=False,
    )
    results = []
    if yolo_results:
        result = yolo_results[0]
        boxes = result.boxes
        names = result.names or getattr(model, "names", {})
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
            for idx in range(len(xyxy)):
                cls_idx = int(clss[idx]) if clss is not None else idx
                if isinstance(names, dict):
                    name = names.get(cls_idx, str(cls_idx))
                else:
                    name = names[cls_idx] if cls_idx < len(names) else str(cls_idx)
                score = float(confs[idx]) if confs is not None else 0.0
                results.append((name, score, [float(x) for x in xyxy[idx].tolist()]))
    results.sort(key=lambda item: item[1], reverse=True)
    return original, {}, results


def list_images(images_dir: Path):
    for path in sorted(images_dir.iterdir()):
        # Skip hidden files (starting with '.')
        if path.name.startswith('.'):
            continue
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".mov", ".mp4", ".avi"}:
            yield path


COLOR_BY_LABEL = {
    "badger": (255, 165, 0),
    "fox": (255, 0, 0),
    "mouse": (255, 255, 0),
    "bird": (0, 200, 0),
    "squirrel": (0, 0, 255),
}


def color_for_label(label: str):
    return COLOR_BY_LABEL.get(label.strip().lower(), (255, 255, 255))


def draw_detections(image: Image.Image, results):
    draw = ImageDraw.Draw(image)
    for name, score, box in results:
        x1, y1, x2, y2 = box
        color = color_for_label(name)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return image


def draw_summary_lines(image: Image.Image, lines: List[Tuple[str, float]]):
    if not lines:
        return image
    draw = ImageDraw.Draw(image)
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except Exception:
        try:
            font = ImageFont.load_default(size=36)
        except TypeError:
            font = ImageFont.load_default()

    def measure(text: str):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            return draw.textsize(text, font=font)

    text_w_sample, text_h_sample = measure("Ag")
    circle_d = max(1, int(text_h_sample))
    padding = 4
    gap = 6
    line_spacing = 4

    line_metrics = []
    max_line_w = 0
    for name, score in lines:
        text = f"{name} [{score:.2f}]"
        text_w, text_h = measure(text)
        line_h = max(text_h, circle_d)
        line_w = circle_d + gap + text_w
        max_line_w = max(max_line_w, line_w)
        line_metrics.append((text, text_w, text_h, line_h))

    total_h = sum(m[3] for m in line_metrics)
    if len(line_metrics) > 1:
        total_h += line_spacing * (len(line_metrics) - 1)
    box_w = max_line_w + padding * 2
    box_h = total_h + padding * 2
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0))
    y = padding
    for (name, _), (text, text_w, text_h, line_h) in zip(lines, line_metrics):
        color = color_for_label(name)
        circle_x = padding
        circle_y = y + (line_h - circle_d) / 2
        draw.ellipse(
            [circle_x, circle_y, circle_x + circle_d, circle_y + circle_d],
            fill=color,
            outline=color,
        )
        text_x = circle_x + circle_d + gap
        text_y = y + (line_h - text_h) / 2
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        y += line_h + line_spacing
    return image


def draw_summary(image: Image.Image, results):
    lines = [(name, score) for name, score, _ in results]
    return draw_summary_lines(image, lines)


def annotate_image(image: Image.Image, results):
    annotated = image.copy()
    draw_detections(annotated, results)
    draw_summary(annotated, results)
    return annotated


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


def run_inference_on_image(model, class_names, image: Image.Image, device: torch.device, score_threshold: float, max_dim: int):
    resized, scale = resize_for_inference(image, max_dim)
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

    return detections, results


def run_inference(model, class_names, image_path: Path, device: torch.device, score_threshold: float, max_dim: int):
    original = Image.open(image_path).convert("RGB")
    detections, results = run_inference_on_image(model, class_names, original, device, score_threshold, max_dim)
    return original, detections, results


def reduce_results_by_label(results: List[Tuple[str, float, List[float]]]):
    label_boxes: Dict[str, List[float]] = {}
    label_scores: Dict[str, float] = {}
    for name, score, box in results:
        if score > label_scores.get(name, -1.0):
            label_scores[name] = score
            label_boxes[name] = box
    return label_boxes, label_scores


def interpolate_boxes(
    prev_boxes: Dict[str, List[float]],
    next_boxes: Dict[str, List[float]],
    t: float,
) -> Dict[str, List[float]]:
    labels = set(prev_boxes.keys()) | set(next_boxes.keys())
    blended: Dict[str, List[float]] = {}
    for label in labels:
        box_a = prev_boxes.get(label)
        box_b = next_boxes.get(label)
        if box_a is not None and box_b is not None:
            blended[label] = [a + (b - a) * t for a, b in zip(box_a, box_b)]
        elif box_a is not None:
            blended[label] = box_a
        elif box_b is not None:
            blended[label] = box_b
    return blended


def process_video(
    video_path: Path,
    model,
    class_names,
    device: torch.device,
    score_threshold: float,
    max_dim: int,
    dest_root: Path,
):
    try:
        import cv2
    except Exception as exc:
        raise SystemExit(f"opencv-python is required for video processing: {exc}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    max_scores: Dict[str, float] = {}
    has_any = False
    last_detected_frame = -1
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    step = max(1, int(round(fps)))
    keyframes: Dict[int, Dict[str, List[float]]] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            _, results = run_inference_on_image(model, class_names, pil, device, score_threshold, max_dim)
            label_boxes, label_scores = reduce_results_by_label(results)
            keyframes[frame_idx] = label_boxes
            if label_boxes:
                has_any = True
                last_detected_frame = frame_idx
                for name, score in label_scores.items():
                    if score > max_scores.get(name, 0.0):
                        max_scores[name] = score
        frame_idx += 1
    cap.release()

    if total_frames <= 0:
        total_frames = frame_idx

    summary_lines = sorted(max_scores.items(), key=lambda item: item[1], reverse=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dest_dir = dest_root / ("detected" if has_any else "empty")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{video_path.stem}.mp4"
    writer = cv2.VideoWriter(str(dest_path), fourcc, fps, (width, height))

    try:
        if has_any:
            trim_frames = int(round(2 * fps))
            last_frame_to_write = min(last_detected_frame + trim_frames, total_frames - 1)
        else:
            last_frame_to_write = total_frames - 1
        total_seconds = (last_frame_to_write + 1) / fps if last_frame_to_write >= 0 else 0.0
        last_printed_second = -1
        frame_idx = 0
        key_indices = sorted(keyframes.keys())
        next_pos = 0
        prev_idx = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx > last_frame_to_write:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            annotated = pil.copy()
            while next_pos < len(key_indices) and key_indices[next_pos] < frame_idx:
                prev_idx = key_indices[next_pos]
                next_pos += 1
            next_idx = key_indices[next_pos] if next_pos < len(key_indices) else None
            if next_idx is not None and frame_idx == next_idx:
                interpolated = keyframes.get(next_idx, {})
            elif prev_idx is None and next_idx is None:
                interpolated = {}
            elif prev_idx is None:
                interpolated = {}
            elif next_idx is None:
                interpolated = keyframes.get(prev_idx, {})
            elif prev_idx == next_idx:
                interpolated = keyframes.get(prev_idx, {})
            else:
                prev_boxes = keyframes.get(prev_idx, {})
                next_boxes = keyframes.get(next_idx, {})
                if not prev_boxes and not next_boxes:
                    interpolated = {}
                elif not prev_boxes and next_boxes:
                    interpolated = {}
                elif prev_boxes and not next_boxes:
                    interpolated = prev_boxes
                else:
                    t = (frame_idx - prev_idx) / float(next_idx - prev_idx)
                    interpolated = interpolate_boxes(prev_boxes, next_boxes, t)

            if interpolated:
                results = [(label, 1.0, box) for label, box in interpolated.items()]
                draw_detections(annotated, results)
            if summary_lines:
                draw_summary_lines(annotated, summary_lines)
            bgr = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            frame_idx += 1

            if fps > 0:
                current_second = int(frame_idx / fps)
                if current_second != last_printed_second:
                    remaining = max(0.0, total_seconds - (frame_idx / fps))
                    sys.stdout.write(
                        f"\r{video_path.name}: {frame_idx / fps:.1f}s processed, {remaining:.1f}s remaining"
                    )
                    sys.stdout.flush()
                    last_printed_second = current_second
    finally:
        cap.release()
        writer.release()
        if fps > 0:
            sys.stdout.write("\n")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model_path = Path(args.model)
    if args.arch == "yolo":
        arch = "yolo"
        model = load_yolo_model(model_path)
        device_arg = yolo_device_arg(device)
        max_dim = args.max_dim or 640
        print(f"Architecture: yolo | imgsz: {max_dim}")
    else:
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
        suffix = image_path.suffix.lower()
        if suffix in {".mov", ".mp4", ".avi"}:
            if not args.box:
                print("Skipping video (use --box to process videos).")
                continue
            if arch == "yolo":
                print("Skipping video (YOLO video inference not implemented).")
                continue
            dest_root = Path(args.images_dir) if args.images_dir else image_path.parent
            process_video(
                image_path,
                model,
                class_names,
                device,
                args.score_threshold,
                max_dim,
                dest_root,
            )
            continue
        start_time = time.perf_counter()
        try:
            if arch == "yolo":
                image, detections, results = run_yolo_inference(
                    model,
                    image_path,
                    args.score_threshold,
                    max_dim,
                    device_arg,
                )
            else:
                image, detections, results = run_inference(
                    model,
                    class_names,
                    image_path,
                    device,
                    args.score_threshold,
                    max_dim,
                )
        except RuntimeError as exc:
            if arch == "yolo" or device.type != "mps" or max_dim <= 512:
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
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        timing_note = f" ({elapsed_ms:.1f} ms)"
        if results:
            top_results = results[:5]
            summary = ", ".join([f"{name} ({score:.2f})" for name, score, _ in top_results])
            print(f"Top 5: {summary}{timing_note}")
        else:
            print(f"Top 5: no detections{timing_note}")

        annotated = None
        if args.box:
            dest_root = Path(args.images_dir) if args.images_dir else image_path.parent
            dest_dir = dest_root / ("detected" if results else "empty")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / image_path.name
            if not dest_path.exists():
                if results:
                    annotated = annotate_image(image, results)
                    annotated.save(dest_path)
                else:
                    shutil.copy2(image_path, dest_path)

        if args.classify:
            dest_root = Path(args.images_dir) if args.images_dir else image_path.parent
            # Count unique labels detected
            unique_labels = set(name for name, _, _ in results)
            label_count = len(unique_labels)
            dest_dir = dest_root / str(label_count)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / image_path.name
            if not dest_path.exists():
                shutil.copy2(image_path, dest_path)

        if output_dir:
            if results:
                if annotated is None:
                    annotated = annotate_image(image, results)
            else:
                annotated = image
            out_path = output_dir / image_path.name
            annotated.save(out_path)


if __name__ == "__main__":
    main()
