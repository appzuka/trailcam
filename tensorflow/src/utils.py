import json
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import keras_cv
import numpy as np
import tensorflow as tf


COCO_BOX_FORMAT = "xywh"
TARGET_BOX_FORMAT = "xyxy"


def resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "cpu":
        tf.config.set_visible_devices([], "GPU")
        return "cpu"

    gpus = tf.config.list_physical_devices("GPU")
    if requested in ("gpu", "auto") and gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        return "gpu"

    tf.config.set_visible_devices([], "GPU")
    return "cpu"


def load_coco(coco_json: Path) -> Tuple[Dict[int, dict], Dict[int, List[dict]], Dict[int, int], List[str]]:
    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    categories = coco.get("categories", [])
    categories_sorted = sorted(categories, key=lambda c: c["id"])
    cat_id_to_contig = {cat["id"]: i + 1 for i, cat in enumerate(categories_sorted)}
    class_names = [cat["name"] for cat in categories_sorted]

    return images, anns_by_image, cat_id_to_contig, class_names


def normalize_file_name(file_name: str) -> str:
    decoded = urllib.parse.unquote(file_name).replace("\\", "/")

    if decoded.startswith("file://"):
        try:
            return Path(urllib.parse.urlparse(decoded).path).name
        except Exception:
            return Path(decoded).name

    if decoded.startswith("../../label-studio/data/media/"):
        decoded = decoded[len("../../label-studio/data/media/") :]
    elif "label-studio/data/media/" in decoded:
        decoded = decoded.split("label-studio/data/media/", 1)[1]

    if "data/local-files" in decoded:
        if "?d=" in decoded:
            parsed = urllib.parse.urlparse(decoded)
            query = urllib.parse.parse_qs(parsed.query)
            d_value = query.get("d", [""])[0]
            if d_value:
                decoded = urllib.parse.unquote(d_value)
        if "data/local-files/" in decoded:
            decoded = decoded.split("data/local-files/", 1)[1]

    decoded = decoded.lstrip("/")
    if decoded.startswith("local-files/"):
        decoded = decoded[len("local-files/") :]

    return decoded


def build_records(coco_json: Path, images_dir: Path) -> Tuple[List[dict], List[str], Dict[int, int]]:
    images, anns_by_image, cat_id_to_contig, class_names = load_coco(coco_json)
    records = []
    for image_id, info in images.items():
        file_name = normalize_file_name(info.get("file_name", ""))
        image_path = images_dir / file_name
        if not image_path.exists():
            image_path = images_dir / Path(file_name).name
        boxes = []
        classes = []
        for ann in anns_by_image.get(image_id, []):
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            label = cat_id_to_contig.get(ann.get("category_id"))
            if label is None:
                continue
            boxes.append([x, y, w, h])
            classes.append(label)
        records.append(
            {
                "image_path": str(image_path),
                "boxes": boxes,
                "classes": classes,
            }
        )
    return records, class_names, cat_id_to_contig


def _generator(records: List[dict]):
    for record in records:
        yield record


def build_dataset(
    coco_json: Path,
    images_dir: Path,
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[tf.data.Dataset, List[str], Dict[int, int], int]:
    records, class_names, cat_id_to_contig = build_records(coco_json, images_dir)

    output_signature = {
        "image_path": tf.TensorSpec(shape=(), dtype=tf.string),
        "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        "classes": tf.TensorSpec(shape=(None,), dtype=tf.int32),
    }

    def gen():
        for record in records:
            boxes = np.array(record["boxes"], dtype=np.float32)
            classes = np.array(record["classes"], dtype=np.int32)
            yield {
                "image_path": record["image_path"],
                "boxes": boxes,
                "classes": classes,
            }

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(records)))

    def load_example(example):
        image_path = example["image_path"]
        image_data = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        boxes = example["boxes"]
        classes = example["classes"]
        bounding_boxes = {
            "boxes": boxes,
            "classes": classes,
        }
        bounding_boxes = keras_cv.bounding_box.convert_format(
            bounding_boxes,
            source=COCO_BOX_FORMAT,
            target=TARGET_BOX_FORMAT,
            images=image,
        )
        return {
            "images": image,
            "bounding_boxes": bounding_boxes,
        }

    ds = ds.map(load_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(
        batch_size,
        padded_shapes={
            "images": [None, None, 3],
            "bounding_boxes": {
                "boxes": [None, 4],
                "classes": [None],
            },
        },
        padding_values={
            "images": tf.constant(0, dtype=tf.float32),
            "bounding_boxes": {
                "boxes": tf.constant(0, dtype=tf.float32),
                "classes": tf.constant(-1, dtype=tf.int32),
            },
        },
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_names, cat_id_to_contig, len(records)


def build_model(num_classes: int):
    backbone = keras_cv.models.ResNet50Backbone.from_preset("resnet50_imagenet")
    model = keras_cv.models.RetinaNet(
        num_classes=num_classes,
        bounding_box_format=TARGET_BOX_FORMAT,
        backbone=backbone,
    )
    return model


def load_image(path: Path) -> tf.Tensor:
    data = tf.io.read_file(str(path))
    image = tf.image.decode_image(data, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_label(label: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label).strip("_")
    return cleaned
