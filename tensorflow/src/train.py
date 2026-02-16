import argparse
import json
from pathlib import Path

import keras_cv
import tensorflow as tf

from utils import build_dataset, build_model, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TensorFlow RetinaNet model on COCO-format data.")
    parser.add_argument("--coco-json", required=True, help="Path to COCO result.json / annotations.json")
    parser.add_argument("--images-dir", required=True, help="Path to images directory")
    parser.add_argument("--output", required=True, help="Output weights path (.weights.h5)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    coco_json = Path(args.coco_json)
    images_dir = Path(args.images_dir)
    dataset, class_names, cat_id_to_contig, record_count = build_dataset(
        coco_json,
        images_dir,
        batch_size=args.batch_size,
        shuffle=True,
    )
    num_classes = len(class_names) + 1
    print(f"Dataset images: {record_count} | classes: {len(class_names)}")

    model = build_model(num_classes=num_classes)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    model.compile(
        classification_loss=keras_cv.losses.FocalLoss(from_logits=True),
        box_loss=keras_cv.losses.SmoothL1Loss(),
        optimizer=optimizer,
    )

    model.fit(dataset, epochs=args.epochs)

    output_path = Path(args.output)
    if output_path.suffix != ".h5" and not output_path.name.endswith(".weights.h5"):
        output_path = output_path.with_suffix(".weights.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(output_path)

    metadata = {
        "class_names": class_names,
        "cat_id_to_contig": cat_id_to_contig,
        "num_classes": num_classes,
        "architecture": "retinanet",
        "bounding_box_format": "xyxy",
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved weights to {output_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
