---
apply: always
---

# PyTorch implementation

This directory contains a PyTorch-based training and inference pipeline that mirrors the Swift toolchain, using Faster R-CNN or YOLO detectors trained on COCO-format data (such as Label Studio COCO exports).

## Environment setup

### macOS (Apple Silicon / MPS)
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r pytorch/requirements.txt
```
PyTorch will use the Apple GPU via MPS when available. The scripts default to `--device auto` and will select MPS automatically on Apple Silicon.

### Linux (CUDA / RTX 2070)
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r pytorch/requirements.txt --index-url https://download.pytorch.org/whl/cu118
```
Use the CUDA wheel index that matches your system. The scripts default to `--device auto` and will select CUDA automatically when available.

## Data expectations
- Training expects a COCO JSON file (`result.json` or `annotations.json`) and a directory of images.
- For Label Studio exports, this typically looks like:
  - `result.json`
  - `images/` (all referenced images)

## Training
Script: `pytorch/src/train.py`

Example (Label Studio export):

Assume running in the pytorch folder

```
python src/train.py \
  --coco-json ../training/trainingset1/result.json \
  --images-dir /Volumes/2TB/Trailcam/labelling/images \
  --output models/animals_frcnn.pth \
  --epochs 10 \
  --batch-size 2 \
  --max-dim 1024
```

Notes:
- `--device auto` is the default and will pick CUDA on Linux or MPS on macOS if available.
- Use `--no-pretrained` to start from random weights (Faster R-CNN / SSDLite).
- On MPS, the scripts default to smaller image sizes; adjust with `--max-dim`, `--min-size`, and `--max-size` if needed.
- On MPS, the default model architecture is `ssdlite` for stability. Override with `--arch fasterrcnn` if you need it.
- On Jetson Nano, use the NVIDIA-provided PyTorch/torchvision builds for JetPack and expect to run smaller models (e.g. `--arch ssdlite` or YOLO nano) with reduced image sizes.

YOLO training example:
```
python src/train.py \
  --arch yolo \
  --yolo-model yolov8n.pt \
  --coco-json ../training/trainingset1/result.json \
  --images-dir /Volumes/2TB/Trailcam/labelling/images \
  --output models/animals_yolo.pt \
  --epochs 50 \
  --batch-size 8 \
  --imgsz 640
```

## Inference
Script: `pytorch/src/infer.py`

Single image example:
```
python src/infer.py \
  --model models/animals_frcnn.pth \
  --image ./images/example.jpg \
  --score-threshold 0.4 \
  --output-dir outputs \
  --max-dim 1024
```

Batch directory example:
```
python src/infer.py \
  --model models/animals_frcnn.pth \
  --images-dir ./images \
  --score-threshold 0.4 \
  --max-dim 1024
```
YOLO example:
```
python src/infer.py \
  --arch yolo \
  --model models/animals_yolo.pt \
  --images-dir ./images \
  --score-threshold 0.4 \
  --max-dim 640
```
To copy images into `detected/` or `empty/` and draw class-colored boxes:
```
python src/infer.py \
  --model models/animals_frcnn.pth \
  --images-dir ./images \
  --box
```

## Code overview
- `pytorch/src/utils.py`: COCO dataset loader, data transforms, device selection, and model creation.
- `pytorch/src/train.py`: training loop (Faster R-CNN / SSDLite / YOLO), checkpoint saving, optional validation split.
- `pytorch/src/infer.py`: loads a checkpoint, runs detection on images, and optionally writes annotated outputs.
- `pytorch/src/ls_backend.py`: Label Studio ML backend using Faster R-CNN (full resolution) with `/predict`, `/train`, and `/webhook`. Supports device selection for inference and training.
Note: `--max-dim` reduces the longest image side to lower memory use, which can improve stability on MPS.

## Checkpoints
Training outputs a `.pth` checkpoint containing:
- model weights
- class names from the COCO categories
- a contiguous class-id mapping

Inference uses the checkpoint metadata to map labels back to class names.

## Label Studio backend (Faster R-CNN / full resolution)
This backend runs inference and training using **Faster R-CNN** without resizing inputs. It listens on `http://localhost:9091` by default.
Note: the Label Studio backend currently supports Faster R-CNN only (not YOLO).

Run the backend (from the `pytorch` folder):
```
python src/ls_backend.py \
  --model models/animals_frcnn.pth \
  --from-name label \
  --to-name image \
  --label-studio-url http://localhost:8080 \
  --label-studio-token "$LS_TOKEN" \
  --data-root ~/2TB \
  --project-id 2 \
  --device cuda \
  --train-device cuda
```

Label Studio connection:
- In Label Studio, add an ML backend with URL `http://localhost:9091`.
- Use the same `from_name`/`to_name` as in your labeling config (they must match the `--from-name` and `--to-name` flags).

Training:
- Hitting `/train` or `/webhook` on the backend triggers a fresh COCO export from Label Studio and trains a new Faster R-CNN checkpoint at full resolution.
- The backend updates the in-memory model after training completes.

Notes:
- `--data-root` should point at the host path that contains `images/...` (for example `~/2TB`).
- You can also set `LS_DATA_ROOT=~/2TB` or `LS_DATA_ROOTS=/path/one:/path/two` to avoid repeating flags.
- Use `--device` to select the inference device (`auto`, `cuda`, `mps`, `cpu`).
- Use `--train-device` to select the training device (`auto`, `cuda`, `mps`, `cpu`).
