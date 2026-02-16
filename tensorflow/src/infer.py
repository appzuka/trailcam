import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import tensorflow as tf
from PIL import Image, ImageDraw

from utils import build_model, load_image, preprocess_image, resolve_device, safe_label


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a TensorFlow RetinaNet model.")
    parser.add_argument("--model", required=True, help="Path to model weights (.weights.h5)")
    parser.add_argument("--metadata", help="Path to metadata JSON (defaults to model path with .json suffix)")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--images-dir", help="Directory of images to run inference on")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", help="Optional directory to save annotated images")
    parser.add_argument("--copy-recognized", action="store_true", help="Copy recognized images into recognized/{high,medium,low,none}")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    return parser.parse_args()


def load_metadata(model_path: Path, metadata_path: Path | None) -> Dict:
    if metadata_path is None:
        metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")
    return json.loads(metadata_path.read_text())


def list_images(images_dir: Path):
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield path


def decode_predictions(predictions, class_names: List[str], score_threshold: float):
    if isinstance(predictions, dict):
        boxes = predictions.get("boxes")
        classes = predictions.get("classes")
        scores = predictions.get("confidence") or predictions.get("scores")
    else:
        raise ValueError("Unexpected prediction format")

    boxes = boxes.numpy()
    classes = classes.numpy()
    scores = scores.numpy()

    results = []
    for box, cls, score in zip(boxes, classes, scores):
        if score < score_threshold:
            continue
        label_index = int(cls)
        if label_index <= 0 or label_index > len(class_names):
            label = str(label_index)
        else:
            label = class_names[label_index - 1]
        results.append((label, float(score), [float(x) for x in box]))
    results.sort(key=lambda item: item[1], reverse=True)
    return results


def draw_detections(image: Image.Image, results, score_threshold: float):
    draw = ImageDraw.Draw(image)
    for label, score, box in results:
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{label} {score:.2f}", fill="red")
    return image


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model_path = Path(args.model)
    metadata = load_metadata(model_path, Path(args.metadata) if args.metadata else None)
    class_names = metadata.get("class_names", [])
    num_classes = metadata.get("num_classes", len(class_names) + 1)

    model = build_model(num_classes=num_classes)
    model.load_weights(model_path)

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
        image_tensor = preprocess_image(load_image(image_path))
        batch = tf.expand_dims(image_tensor, axis=0)
        predictions = model.predict(batch, verbose=0)[0]

        results = decode_predictions(predictions, class_names, args.score_threshold)
        if results:
            top_results = results[:5]
            summary = ", ".join([f"{label} ({score:.2f})" for label, score, _ in top_results])
            print(f"Top 5: {summary}")
        else:
            print("Top 5: no detections")

        if args.copy_recognized:
            recognized_root = image_path.parent / "recognized"
            dest_dir = None
            dest_name = image_path.name
            if results:
                top_label, top_score, _ = results[0]
                cleaned = safe_label(top_label)
                if cleaned:
                    dest_name = f"{cleaned}_{image_path.name}"
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
            image_pil = Image.open(image_path).convert("RGB")
            annotated = draw_detections(image_pil, results, args.score_threshold)
            annotated.save(output_dir / image_path.name)


if __name__ == "__main__":
    main()
