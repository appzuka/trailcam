# TensorFlow implementation

This directory contains a TensorFlow/KerasCV object detection pipeline (RetinaNet) that mirrors the PyTorch functionality: training on COCO exports, inference on images, and a Label Studio backend with `/predict`, `/train`, and `/webhook` routes.

## Environment setup

### macOS (Apple Silicon / M1)
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow-macos tensorflow-metal
pip install -r tensorflow/requirements.txt
```
Notes:
- `tensorflow-metal` enables GPU (MPS) support on Apple Silicon.

### Linux (CUDA / RTX 2070)
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r tensorflow/requirements.txt
```
Use the official TensorFlow GPU build for your CUDA version (see TensorFlow install docs if you need a specific CUDA wheel).

## Data expectations
- Training expects a COCO JSON file (`result.json` or `annotations.json`) and a directory of images.
- For Label Studio exports, this typically looks like:
  - `result.json`
  - `images/` (all referenced images)

## Training
Script: `tensorflow/src/train.py`

Assume running from the repo root:
```
python tensorflow/src/train.py \
  --coco-json training/trainingset1/result.json \
  --images-dir /Volumes/2TB/Trailcam/labelling/images \
  --output tensorflow/models/animals_retinanet.weights.h5 \
  --epochs 10 \
  --batch-size 2
```
Notes:
- `--device auto` is the default and will pick GPU if available.
- Training saves weights to the path you pass and writes metadata to a sibling `.json` file.

## Inference
Script: `tensorflow/src/infer.py`

Single image example:
```
python tensorflow/src/infer.py \
  --model tensorflow/models/animals_retinanet.weights.h5 \
  --image ./images/example.jpg \
  --score-threshold 0.4 \
  --output-dir tensorflow/outputs
```

Batch directory example:
```
python tensorflow/src/infer.py \
  --model tensorflow/models/animals_retinanet.weights.h5 \
  --images-dir ./images \
  --score-threshold 0.4
```

Copy recognized images into `recognized/high`, `recognized/medium`, `recognized/low`, or `recognized/none`:
```
python tensorflow/src/infer.py \
  --model tensorflow/models/animals_retinanet.weights.h5 \
  --images-dir ./images \
  --copy-recognized
```

## Label Studio backend
Script: `tensorflow/src/ls_backend.py`

Run the backend (defaults to `http://localhost:9091`):
```
python tensorflow/src/ls_backend.py \
  --model tensorflow/models/animals_retinanet.weights.h5 \
  --from-name label \
  --to-name image \
  --label-studio-url http://localhost:8080 \
  --label-studio-token "$LS_TOKEN" \
  --data-root /Volumes/2TB/Trailcam/labelling/images \
  --project-id 2
```

Label Studio connection:
- Add an ML backend with URL `http://localhost:9091`.
- Ensure `from_name`/`to_name` match your labeling config and backend flags.

Training:
- Hitting `/train` or `/webhook` triggers a fresh COCO export from Label Studio and retrains the model.
- The backend hot-swaps the in-memory model after training completes.

## Code overview
- `tensorflow/src/utils.py`: COCO loader, path normalization, device setup, and model creation.
- `tensorflow/src/train.py`: training loop and checkpoint + metadata save.
- `tensorflow/src/infer.py`: inference and optional copying to recognized folders.
- `tensorflow/src/ls_backend.py`: Label Studio backend with `/predict`, `/train`, `/webhook`.
