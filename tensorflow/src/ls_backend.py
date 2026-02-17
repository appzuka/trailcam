import argparse
import base64
import json
import os
import threading
import urllib.parse
import urllib.request
import uuid
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import keras_cv
import numpy as np
import tensorflow as tf

from utils import build_dataset, build_model, load_image, preprocess_image, resolve_device, safe_label


def parse_args():
    parser = argparse.ArgumentParser(description="Label Studio backend for TensorFlow RetinaNet.")
    parser.add_argument("--model", required=True, help="Path to model weights (.weights.h5)")
    parser.add_argument("--metadata", help="Path to metadata JSON (defaults to model path with .json suffix)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--from-name", required=True, help="Label Studio from_name")
    parser.add_argument("--to-name", required=True, help="Label Studio to_name")
    parser.add_argument("--data-key", default="image", help="Task data key for image URL")
    parser.add_argument("--data-root", help="Root directory for local-files")
    parser.add_argument("--label-studio-url", help="Base Label Studio URL")
    parser.add_argument("--label-studio-token", help="Label Studio token (defaults to LS_TOKEN)")
    parser.add_argument("--project-id", type=int, help="Default project ID for /train")
    parser.add_argument("--train-output", help="Output weights path for training")
    parser.add_argument("--model-version", default="recognize-tensorflow")
    parser.add_argument("--allow-absolute", action="store_true", help="Allow absolute paths in tasks")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--images-root", help="Override images root when training")
    parser.add_argument("--image-size", type=int, default=640, help="Resize images to a fixed square size for training/inference")
    return parser.parse_args()


def resolve_token(token: Optional[str]) -> Optional[str]:
    if token:
        return token
    env = os.environ.get("LS_TOKEN")
    if env:
        return env
    return None


def looks_like_jwt(token: str) -> bool:
    return token.count(".") == 2


def decode_jwt_payload(token: str) -> Optional[Dict[str, Any]]:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        data = base64.urlsafe_b64decode(payload + padding)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def auth_header_value(token: str, label_studio_url: Optional[str]) -> Optional[str]:
    if " " in token:
        return token
    if looks_like_jwt(token):
        payload = decode_jwt_payload(token)
        if payload and payload.get("token_type") == "refresh" and label_studio_url:
            access = refresh_access_token(token, label_studio_url)
            if access:
                return f"Bearer {access}"
        return f"Bearer {token}"
    return f"Token {token}"


def refresh_access_token(refresh_token: str, base_url: str) -> Optional[str]:
    url = base_url.rstrip("/") + "/api/token/refresh/"
    payload = json.dumps({"refresh": refresh_token}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
            return data.get("access")
    except Exception:
        return None


def http_request(url: str, token: Optional[str], label_studio_url: Optional[str]) -> tuple[bytes, int, Dict[str, str]]:
    headers = {}
    if token:
        auth_value = auth_header_value(token, label_studio_url)
        if auth_value:
            headers["Authorization"] = auth_value
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
        return data, resp.status, dict(resp.headers)


def export_coco(project_id: int, label_studio_url: str, token: Optional[str]) -> Path:
    base_url = label_studio_url.rstrip("/")
    export_url = f"{base_url}/api/projects/{project_id}/export?exportType=COCO"
    data, status, _ = http_request(export_url, token, label_studio_url)
    if status < 200 or status >= 300:
        raise RuntimeError(f"Label Studio export failed: HTTP {status}")

    temp_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"ls_export_{uuid.uuid4()}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if data[:2] == b"PK":
        zip_path = temp_dir / "export.zip"
        zip_path.write_bytes(data)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)
        return temp_dir

    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception:
        payload = None

    if isinstance(payload, dict):
        if payload.get("annotations") and payload.get("images"):
            (temp_dir / "result.json").write_bytes(data)
            return temp_dir
        download_url = payload.get("download_url") or payload.get("url")
        if download_url:
            if not download_url.startswith("http"):
                download_url = f"{base_url}/{download_url.lstrip('/')}"
            data, status, _ = http_request(download_url, token, label_studio_url)
            if status < 200 or status >= 300:
                raise RuntimeError(f"Export download failed: HTTP {status}")
            if data[:2] == b"PK":
                zip_path = temp_dir / "export.zip"
                zip_path.write_bytes(data)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(temp_dir)
                return temp_dir
            (temp_dir / "result.json").write_bytes(data)
            return temp_dir

    (temp_dir / "result.json").write_bytes(data)
    return temp_dir


def locate_result_json(export_dir: Path) -> Path:
    candidates = list(export_dir.rglob("*.json"))
    for name in ("result.json", "annotations.json"):
        for candidate in candidates:
            if candidate.name == name:
                return candidate
    if candidates:
        return candidates[0]
    raise RuntimeError("No JSON found in export")


def resolve_image_value(value: Any, data_root: Optional[Path], allow_absolute: bool, token: Optional[str], base_url: Optional[str]) -> Path:
    if not isinstance(value, str):
        raise RuntimeError("Image value is not a string")

    decoded = urllib.parse.unquote(value)

    if decoded.startswith("http://") or decoded.startswith("https://"):
        data, status, _ = http_request(decoded, token, base_url)
        if status < 200 or status >= 300:
            raise RuntimeError(f"Download failed: HTTP {status}")
        temp_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "ls_backend_images"
        temp_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(urllib.parse.urlparse(decoded).path).name or f"image_{uuid.uuid4()}.jpg"
        dest = temp_dir / filename
        dest.write_bytes(data)
        return dest

    if decoded.startswith("file://"):
        return Path(urllib.parse.urlparse(decoded).path)

    if "/data/local-files/" in decoded:
        parsed = urllib.parse.urlparse(decoded)
        if parsed.query:
            query = urllib.parse.parse_qs(parsed.query)
            d_value = query.get("d", [""])[0]
            if d_value:
                decoded = urllib.parse.unquote(d_value)
        if "data/local-files/" in decoded:
            decoded = decoded.split("data/local-files/", 1)[1]

    decoded = decoded.lstrip("/")
    if decoded.startswith("local-files/"):
        decoded = decoded[len("local-files/") :]

    if decoded.startswith("/"):
        if allow_absolute:
            return Path(decoded)
        if data_root:
            return data_root / decoded.lstrip("/")

    if data_root:
        return data_root / decoded
    return Path(decoded)


def load_metadata(model_path: Path, metadata_path: Path | None) -> Dict:
    if metadata_path is None:
        metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")
    return json.loads(metadata_path.read_text())


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "numpy"):
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.array(value)


def predict(model, class_names, image_path: Path, threshold: float, image_size: int | None):
    image_tensor = preprocess_image(load_image(image_path))
    original_shape = tf.shape(image_tensor)
    original_height = tf.cast(original_shape[0], tf.float32)
    original_width = tf.cast(original_shape[1], tf.float32)
    scale_x = 1.0
    scale_y = 1.0
    if image_size and image_size > 0:
        resized = tf.image.resize(image_tensor, (image_size, image_size))
        scale_x = original_width / float(image_size)
        scale_y = original_height / float(image_size)
        batch = tf.expand_dims(resized, axis=0)
    else:
        batch = tf.expand_dims(image_tensor, axis=0)
    predictions = model.predict(batch, verbose=0)
    boxes = predictions.get("boxes")
    classes = predictions.get("classes")
    scores = predictions.get("confidence")
    if scores is None:
        scores = predictions.get("scores")
    if boxes is None or classes is None or scores is None:
        return []

    boxes = _to_numpy(boxes)
    classes = _to_numpy(classes)
    scores = _to_numpy(scores)

    if len(boxes.shape) == 3:
        boxes = boxes[0]
    if len(classes.shape) == 2:
        classes = classes[0]
    if len(scores.shape) == 2:
        scores = scores[0]

    width = float(original_width.numpy())
    height = float(original_height.numpy())

    results = []
    for box, cls, score in zip(boxes, classes, scores):
        if score < threshold:
            continue
        label_index = int(cls)
        if label_index <= 0 or label_index > len(class_names):
            label = str(label_index)
        else:
            label = class_names[label_index - 1]
        x1, y1, x2, y2 = box.tolist()
        x1 *= float(scale_x)
        x2 *= float(scale_x)
        y1 *= float(scale_y)
        y2 *= float(scale_y)
        results.append(
            {
                "label": label,
                "score": float(score),
                "x": max(0.0, min(100.0, x1 / width * 100.0)),
                "y": max(0.0, min(100.0, y1 / height * 100.0)),
                "width": max(0.0, min(100.0, (x2 - x1) / width * 100.0)),
                "height": max(0.0, min(100.0, (y2 - y1) / height * 100.0)),
                "orig_width": int(width),
                "orig_height": int(height),
            }
        )
    return results


def train_model(
    coco_json: Path,
    images_dir: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    image_size: int,
):
    dataset, class_names, cat_id_to_contig, record_count = build_dataset(
        coco_json,
        images_dir,
        batch_size=batch_size,
        shuffle=True,
        image_size=image_size,
    )
    num_classes = len(class_names) + 1
    steps_per_epoch = max(1, int((record_count + batch_size - 1) / batch_size))
    dataset = dataset.repeat()

    model = build_model(num_classes=num_classes)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, weight_decay=weight_decay)
    model.compile(
        classification_loss=keras_cv.losses.FocalLoss(from_logits=True),
        box_loss=keras_cv.losses.SmoothL1Loss(),
        optimizer=optimizer,
    )

    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(output_path)

    metadata = {
        "class_names": class_names,
        "cat_id_to_contig": cat_id_to_contig,
        "num_classes": num_classes,
        "architecture": "retinanet",
        "bounding_box_format": "xyxy",
        "image_size": self._args.image_size,
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))


def parse_project_id(payload: Dict[str, Any], fallback: Optional[int]) -> Optional[int]:
    if "project_id" in payload:
        value = payload.get("project_id")
        return int(value)
    if "project" in payload:
        project = payload.get("project")
        if isinstance(project, dict) and "id" in project:
            return int(project["id"])
        try:
            return int(project)
        except Exception:
            return fallback
    return fallback


class ModelStore:
    def __init__(self, model, class_names):
        self._model = model
        self._class_names = class_names
        self._lock = threading.Lock()

    def with_model(self):
        with self._lock:
            return self._model, self._class_names

    def update(self, model, class_names):
        with self._lock:
            self._model = model
            self._class_names = class_names


class TrainingController:
    def __init__(self, args, model_store):
        self._args = args
        self._store = model_store
        self._lock = threading.Lock()
        self._in_progress = False

    def start(self, payload: Dict[str, Any]) -> str:
        with self._lock:
            if self._in_progress:
                return "training already in progress"
            self._in_progress = True

        thread = threading.Thread(target=self._run_training, args=(payload,), daemon=True)
        thread.start()
        return "training started"

    def _run_training(self, payload: Dict[str, Any]):
        try:
            if self._args.image_size <= 0:
                raise RuntimeError("--image-size must be > 0 for training")
            project_id = parse_project_id(payload, self._args.project_id)
            if project_id is None:
                raise RuntimeError("Missing project_id for training")
            if not self._args.label_studio_url:
                raise RuntimeError("Missing --label-studio-url for training export")

            export_dir = export_coco(project_id, self._args.label_studio_url, self._args.label_studio_token)
            coco_json = locate_result_json(export_dir)
            images_dir = None
            if self._args.images_root:
                images_dir = Path(self._args.images_root)
            else:
                candidate = export_dir / "images"
                if candidate.exists():
                    images_dir = candidate
            if images_dir is None:
                raise RuntimeError("Images directory not found; set --images-root")

            output_path = Path(self._args.train_output or self._args.model)
            train_model(
                coco_json,
                images_dir,
                output_path,
                epochs=self._args.epochs,
                batch_size=self._args.batch_size,
                lr=self._args.lr,
                weight_decay=self._args.weight_decay,
                image_size=self._args.image_size,
            )

            metadata = load_metadata(output_path, None)
            num_classes = metadata.get("num_classes", len(metadata.get("class_names", [])) + 1)
            model = build_model(num_classes=num_classes)
            model.load_weights(output_path)
            self._store.update(model, metadata.get("class_names", []))
            print("Training complete. Model updated.")
        except Exception as exc:
            print(f"Training failed: {exc}")
        finally:
            with self._lock:
                self._in_progress = False


class BackendHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b"{}"

        if self.path.rstrip("/") == "/setup":
            self._send_json({"model_version": self.server.model_version})
            return

        if self.path.rstrip("/") in ("/train", "/webhook"):
            payload = self._parse_json(body)
            message = self.server.training.start(payload)
            self._send_json({"status": message})
            return

        if self.path.rstrip("/") == "/predict":
            payload = self._parse_json(body)
            results = self._handle_predict(payload)
            self._send_json({"results": results})
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _handle_predict(self, payload: Dict[str, Any]):
        tasks = payload.get("tasks") or []
        responses = []
        for task in tasks:
            data = task.get("data", {}) if isinstance(task, dict) else {}
            image_value = data.get(self.server.data_key)
            detections = []
            top_score = 0.0

            if image_value is not None:
                try:
                    image_path = resolve_image_value(
                        image_value,
                        self.server.data_root,
                        self.server.allow_absolute,
                        self.server.label_studio_token,
                        self.server.label_studio_url,
                    )
                    model, class_names = self.server.model_store.with_model()
                    results = predict(
                        model,
                        class_names,
                        image_path,
                        self.server.threshold,
                        self.server.image_size,
                    )
                    for det in results:
                        detections.append(
                            {
                                "from_name": self.server.from_name,
                                "to_name": self.server.to_name,
                                "type": "rectanglelabels",
                                "value": {
                                    "x": det["x"],
                                    "y": det["y"],
                                    "width": det["width"],
                                    "height": det["height"],
                                    "rectanglelabels": [det["label"]],
                                },
                                "score": det["score"],
                                "original_width": det["orig_width"],
                                "original_height": det["orig_height"],
                                "image_rotation": 0,
                            }
                        )
                        top_score = max(top_score, det["score"])
                except Exception as exc:
                    print(f"Prediction error: {exc}")

            responses.append(
                {
                    "result": detections,
                    "score": top_score,
                    "model_version": self.server.model_version,
                }
            )
        return responses

    def _parse_json(self, data: bytes) -> Dict[str, Any]:
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return {}

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class BackendServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, *, model_store, training, args):
        super().__init__(server_address, handler_class)
        self.model_store = model_store
        self.training = training
        self.threshold = args.threshold
        self.from_name = args.from_name
        self.to_name = args.to_name
        self.data_key = args.data_key
        self.data_root = Path(args.data_root).expanduser() if args.data_root else None
        self.allow_absolute = args.allow_absolute
        self.model_version = args.model_version
        self.label_studio_url = args.label_studio_url
        self.label_studio_token = args.label_studio_token
        self.image_size = args.image_size


def main():
    args = parse_args()
    resolve_device(args.device)
    args.label_studio_token = resolve_token(args.label_studio_token)

    model_path = Path(args.model)
    metadata = load_metadata(model_path, Path(args.metadata) if args.metadata else None)
    num_classes = metadata.get("num_classes", len(metadata.get("class_names", [])) + 1)
    if args.image_size <= 0 and metadata.get("image_size"):
        args.image_size = int(metadata["image_size"])

    model = build_model(num_classes=num_classes)
    model.load_weights(model_path)

    model_store = ModelStore(model, metadata.get("class_names", []))
    training = TrainingController(args, model_store)

    server = BackendServer((args.host, args.port), BackendHandler, model_store=model_store, training=training, args=args)
    print(f"Label Studio backend listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
