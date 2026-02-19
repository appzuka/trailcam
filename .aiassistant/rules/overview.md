---
apply: always
---

# Recognize

Recognize is a macOS 14+ Swift command-line tool that trains and runs Core ML / Vision models to detect animals in images and video. It can:
- Train an MLObjectDetector from a Label Studio COCO export.
- Run object detection on image folders or sampled video frames.
- Run a Label Studio ML backend server for live predictions and on-demand training.
- Provide a legacy video classifier mode that moves videos into per-animal folders using Vision image classification.

The project is a single Swift Package executable with its main entry point in `Sources/Recognize/Recognize.swift`.

## Build and run
- Build: `swift build`
- Run (local build): `swift run recognize --help`
- Output binary (release): `swift build -c release` then `./.build/release/recognize ...`

## Label Studio script setup
Several helper scripts in `scripts/` expect the Label Studio token in `LS_TOKEN`.

Add it to your zsh profile (example uses `~/.zshrc`):
```
export LS_TOKEN="YOUR_LABEL_STUDIO_TOKEN"
```
Reload your shell or run `source ~/.zshrc`.

## Label Studio (Docker)
Label Studio runs in a Docker container for local labeling.

Start the container (from `scripts/docker.sh`):
```
docker run -it -e LOCAL_FILES_SERVING_ENABLED=true -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/data -p 8080:8080 -v /Volumes/Samsung\ 2TB/Trailcam/labelling/images:/data/local-files/images -v /Volumes/Samsung\ 2TB/Trailcam/mydata:/label-studio/data heartexlabs/label-studio:latest
```
Access it at `http://localhost:8080`.

## Commands and examples

### Train from Label Studio COCO export
Command:
```
recognize train <coco_dir> <output_model> [--images-root <path>]
```
Examples:
```
recognize train training/trainingset1 Models/Animals.mlmodel
recognize train training/trainingset1 Models/Animals.mlmodel --images-root "/Volumes/Samsung 2TB/Trailcam/labelling/images"
```
Notes:
- `coco_dir` must include `result.json` and `images/` (from Label Studio COCO export).
- `--images-root` overrides the image lookup if the JSON paths are incorrect.

### Detect objects in images
Command:
```
recognize detect <model_path> <images_dir> [--threshold <0-1>]
```
Example:
```
recognize detect Models/Animals.mlmodel ./images --threshold 0.3
```

### Detect objects in video (sampled frames)
Command:
```
recognize detect-video <model_path> <video_dir> [options]
```
Example:
```
recognize detect-video Models/Animals.mlmodel ./videos --interval 1.0 --max-samples 60 --threshold 0.3
```
Common options:
- `--interval <seconds>`: seconds between sampled frames.
- `--max-samples <count>`: max frames to sample per file.
- `--threshold <0-1>`: confidence threshold.

### Label Studio backend server
Command:
```
recognize ls-backend --model <model_path> --from-name <name> --to-name <name> [options]
```
Example:
```
recognize ls-backend --model Models/Animals.mlmodel --from-name label --to-name image --host 0.0.0.0 --port 9090
```
Examples using the Label Studio token:
```
export LS_TOKEN="YOUR_LABEL_STUDIO_TOKEN"
recognize ls-backend --model Models/Animals.mlmodel --from-name label --to-name image --label-studio-url http://localhost:8080
```
```
recognize ls-backend --model Models/Animals.mlmodel --from-name label --to-name image --label-studio-url http://localhost:8080 --label-studio-token "$LS_TOKEN"
```
Common options:
- `--threshold <0-1>`: confidence threshold for predictions.
- `--data-key <key>`: task data key for image URL (default: `image`).
- `--data-root <path>`: root directory for resolving relative paths.
- `--label-studio-url <url>` and `--label-studio-token <token>`: enable image downloads and training export (token defaults to `LS_TOKEN` if not set on the command line).
- `--project-id <id>` and `--train-output <path>`: defaults for `/train` endpoint.
- `--model-version <text>`: version string in predictions (default: `recognize`).
Note: hitting `/train` triggers a fresh COCO export from Label Studio via the API, so it uses the latest project data. The standalone `recognize train` command still expects a local COCO export directory.

### Legacy video classification (move into folders)
Command:
```
recognize <video_dir> [legacy options]
```
Example:
```
recognize ./videos --interval 2 --max-samples 30 --threshold 0.2 --log-labels
```
This mode uses Vision image classification to detect a small set of animals and moves matching videos into subfolders like `bird/`, `badger/`, `fox/`, etc. It supports additional heuristics like center crop, grid sampling, and motion-based ROI.

## Features
- **Create ML training pipeline**: converts COCO exports into a Create ML dataset and trains `MLObjectDetector` models.
- **Model loading**: supports `.mlmodel` and `.mlmodelc`, compiling when needed.
- **Image detection**: runs Vision Core ML on still images and reports top detections.
- **Video detection**: samples frames and tracks best detection per label.
- **Legacy classifier**: classification-based animal detection with per-class thresholds and multiple passes (full frame, motion ROI, center crop, tiled grid).
- **Label Studio backend**: lightweight HTTP server with `/health`, `/setup`, `/predict`, `/train`, and `/webhook` endpoints.
- **Label Studio integration**: downloads tasks, resolves `data/local-files` paths, and refreshes JWT tokens automatically when required.

## Main functions and flow (high level)
- `parseCommand` and `run`: parse CLI arguments and dispatch to the selected subcommand.
- `prepareCreateMLDataset` and `trainModel`: create the dataset from COCO export and train the `MLObjectDetector`.
- `loadCoreMLModel` and `detectObjects`: load a Core ML model and run Vision inference on images.
- `runDetect` and `runDetectVideo`: iterate files, run detection, and print summaries.
- `detectAnimal` and helpers: legacy classifier path that samples frames, applies ROI strategies, and chooses the best animal match.
- `runLabelStudioBackend`, `HTTPConnectionHandler`, and `handlePredictRequest`: host the Label Studio backend and format predictions.
- `exportCocoFromLabelStudio` and training helpers: fetch COCO exports and kick off training via `/train`.

## Notes
- Requires macOS 14+ due to Vision image classification usage.
- The canonical CLI usage text is embedded in `Sources/Recognize/Recognize.swift` as `usageText`.

## ToDo
- Add a train/validation split (and basic metrics reporting) to the PyTorch training pipeline.
- Create a test dataset including tricky examples that is never used for training
- Plan test for trained models to compare models, parameters, resolution for accuracy and speed
- Make a utility to scan raw files and convert their name to the timestamp on the first frame
