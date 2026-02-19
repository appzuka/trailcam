import json
import os
import random
import re
import shutil
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "mps":
        return torch.device("mps")
    return torch.device("cpu")


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


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, image, target):
        if random.random() < self.probability:
            image = F.hflip(image)
            if "boxes" in target:
                width = image.shape[-1]
                boxes = target["boxes"]
                boxes = boxes.clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class ResizeMaxDim:
    def __init__(self, max_dim: int):
        self.max_dim = max_dim

    def __call__(self, image, target):
        if self.max_dim <= 0:
            return image, target
        width, height = image.size
        max_side = max(width, height)
        if max_side <= self.max_dim:
            return image, target
        scale = self.max_dim / float(max_side)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        image = F.resize(image, [new_height, new_width])
        if "boxes" in target:
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = boxes * torch.tensor([scale, scale, scale, scale])
                target["boxes"] = boxes
        return image, target


def get_transform(train: bool, max_dim: int) -> Compose:
    transforms = [ResizeMaxDim(max_dim), ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


class COCODataset(Dataset):
    def __init__(self, coco_json: Path, images_dir: Path, train: bool = True, max_dim: int = 0):
        self.images, self.anns_by_image, self.cat_id_to_contig, self.class_names = load_coco(coco_json)
        self.images_dir = images_dir
        self.image_ids = sorted(self.images.keys())
        self.transforms = get_transform(train, max_dim)
        self._basename_index: Optional[Dict[str, Path]] = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def _resolve_image_path(self, file_name: str) -> Path:
        normalized = self._normalize_file_name(file_name)
        candidate = self.images_dir / normalized
        if candidate.exists():
            return candidate
        fallback = self.images_dir / Path(normalized).name
        if fallback.exists():
            return fallback
        if self._basename_index is None:
            self._basename_index = self._build_basename_index()
        basename = Path(normalized).name
        mapped = self._basename_index.get(basename)
        if mapped is not None:
            return mapped
        return candidate

    def _build_basename_index(self) -> Dict[str, Path]:
        index: Dict[str, Path] = {}
        if not self.images_dir.exists():
            return index
        for path in self.images_dir.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            if name not in index:
                index[name] = path
        return index

    def _normalize_file_name(self, file_name: str) -> str:
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
            idx = decoded.find("data/local-files/")
            if idx != -1:
                decoded = decoded[idx + len("data/local-files/") :]

        decoded = decoded.lstrip("/")
        if decoded.startswith("local-files/"):
            decoded = decoded[len("local-files/") :]
        while decoded.startswith("images/"):
            decoded = decoded[len("images/") :]

        parts = decoded.split("/")
        if parts:
            parts[-1] = re.sub(r"^[a-f0-9]{8,64}__", "", parts[-1], flags=re.IGNORECASE)
            decoded = "/".join(parts)

        return decoded

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        info = self.images[image_id]
        image_path = self._resolve_image_path(info["file_name"])
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in self.anns_by_image.get(image_id, []):
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            label = self.cat_id_to_contig.get(ann.get("category_id"))
            if label is None:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(label)
            areas.append(float(w * h))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id]),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        image, target = self.transforms(image, target)
        return image, target


def _link_or_copy(src: Path, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def export_yolo_dataset(
    coco_json: Path,
    images_dir: Path,
    output_dir: Path,
    val_split: float,
    seed: int,
) -> Tuple[Path, List[str]]:
    output_dir = output_dir.resolve()
    images, anns_by_image, cat_id_to_contig, class_names = load_coco(coco_json)
    dataset = COCODataset(coco_json, images_dir, train=True, max_dim=0)
    image_ids = sorted(images.keys())

    if val_split > 0:
        rng = random.Random(seed)
        rng.shuffle(image_ids)
        split_idx = int(len(image_ids) * (1 - val_split))
        train_ids = image_ids[:split_idx]
        val_ids = image_ids[split_idx:]
    else:
        train_ids = image_ids
        val_ids = []

    output_dir.mkdir(parents=True, exist_ok=True)
    train_images_dir = output_dir / "images" / "train"
    val_images_dir = output_dir / "images" / "val"
    train_labels_dir = output_dir / "labels" / "train"
    val_labels_dir = output_dir / "labels" / "val"

    def resolve_size(info: dict, image_path: Path) -> Tuple[int, int]:
        width = info.get("width")
        height = info.get("height")
        if width and height:
            return int(width), int(height)
        with Image.open(image_path) as img:
            return img.size

    def write_split(ids: List[int], split_images: Path, split_labels: Path):
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            info = images[image_id]
            image_path = dataset._resolve_image_path(info["file_name"])
            suffix = image_path.suffix if image_path.suffix else ".jpg"
            dest_image = split_images / f"{image_id}{suffix}"
            _link_or_copy(image_path, dest_image)

            width, height = resolve_size(info, image_path)
            anns = anns_by_image.get(image_id, [])
            lines = []
            for ann in anns:
                if ann.get("iscrowd"):
                    continue
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue
                class_idx = cat_id_to_contig.get(ann["category_id"])
                if class_idx is None:
                    continue
                class_idx -= 1
                x_center = (x + w / 2.0) / width
                y_center = (y + h / 2.0) / height
                w_norm = w / width
                h_norm = h / height
                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                w_norm = min(max(w_norm, 0.0), 1.0)
                h_norm = min(max(h_norm, 0.0), 1.0)
                lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            label_path = split_labels / f"{image_id}.txt"
            label_path.write_text("\n".join(lines), encoding="utf-8")

    write_split(train_ids, train_images_dir, train_labels_dir)
    if val_ids:
        write_split(val_ids, val_images_dir, val_labels_dir)

    data_yaml = output_dir / "data.yaml"
    val_path = "images/val" if val_ids else "images/train"
    lines = [
        f"path: {output_dir}",
        "train: images/train",
        f"val: {val_path}",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml, class_names


def build_model(
    num_classes: int,
    pretrained: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
    arch: str = "fasterrcnn",
    ssdlite_backbone_weights: bool | None = None,
):
    arch = arch.lower()
    if arch in ("ssdlite", "ssdlite320", "ssdlite320_mobilenet_v3_large"):
        if pretrained:
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            expected_classes = len(weights.meta.get("categories", []))
            if expected_classes and num_classes != expected_classes:
                backbone_weights = MobileNet_V3_Large_Weights.DEFAULT
                model = ssdlite320_mobilenet_v3_large(
                    weights=None,
                    weights_backbone=backbone_weights,
                    num_classes=num_classes,
                )
            else:
                model = ssdlite320_mobilenet_v3_large(weights=weights, num_classes=num_classes)
        else:
            backbone_weights = None
            if ssdlite_backbone_weights:
                backbone_weights = MobileNet_V3_Large_Weights.DEFAULT
            model = ssdlite320_mobilenet_v3_large(
                weights=None,
                weights_backbone=backbone_weights,
                num_classes=num_classes,
            )
        if hasattr(model, "transform"):
            model.transform.min_size = (int(min_size),)
            model.transform.max_size = int(max_size)
        return model

    if arch in ("fasterrcnn", "fasterrcnn_resnet50_fpn"):
        if pretrained:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(weights=weights)
        else:
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.transform.min_size = (int(min_size),)
        model.transform.max_size = int(max_size)
        return model

    raise ValueError(f"Unsupported architecture: {arch}")
