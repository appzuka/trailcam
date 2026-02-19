#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  extract_frames.sh [base_dir] [options]

Behavior:
  - Looks for .mp4/.mov files in <base_dir>/ingest
  - Extracts frames into <base_dir>/images as <video>-<frame>.jpg
  - Moves processed videos into <base_dir>/done

Options:
  --interval <seconds>   Extract one frame every N seconds (default: 2.0)
  --fps <fps>            Extract at N frames per second (overrides --interval)
  --max-frames <count>   Cap frames per video (default: 0 = no cap)
  --ext <ext>            Output image extension: jpg or png (default: jpg)
  --crop                 Crop lower 85 pixels (1920x1080 -> 1920x995)
  -h, --help             Show this help

Examples:
  ./scripts/extract_frames.sh /path/to/project --interval 1.0
  ./scripts/extract_frames.sh /path/to/project --fps 0.5
USAGE
}

base_dir="."
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  base_dir="$1"
  shift
fi
base_dir="$(cd "$base_dir" && pwd)"

interval="2.0"
fps=""
max_frames="0"
ext="jpg"
crop="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      shift
      interval="${1:-}"
      ;;
    --fps)
      shift
      fps="${1:-}"
      ;;
    --max-frames)
      shift
      max_frames="${1:-}"
      ;;
    --ext)
      shift
      ext="${1:-}"
      ;;
    --crop)
      crop="true"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift || true
done

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Install via Homebrew: brew install ffmpeg" >&2
  exit 1
fi

if [[ "$ext" != "jpg" && "$ext" != "png" ]]; then
  echo "--ext must be 'jpg' or 'png'" >&2
  exit 1
fi

ingest_dir="$base_dir"
images_dir="$base_dir/images"
done_dir="$base_dir/done"

if [[ ! -d "$ingest_dir" ]]; then
  echo "Ingest directory not found: $ingest_dir" >&2
  exit 1
fi

ingest_dir="$(cd "$ingest_dir" && pwd)"
mkdir -p "$images_dir" "$done_dir"
images_dir="$(cd "$images_dir" && pwd)"
done_dir="$(cd "$done_dir" && pwd)"

collect_files() {
  find "$ingest_dir" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mov" \) -print0
}

# Collect all files into an array first to avoid stdin corruption
mapfile -t -d '' files < <(collect_files)

found_any="false"
for file in "${files[@]}"; do
  [[ -z "$file" ]] && continue
  if [[ ! -f "$file" ]]; then
    echo "Skipping missing file: $file" >&2
    continue
  fi
  found_any="true"
  base=$(basename "$file")
  name="${base%.*}"

  filter=""
  if [[ -n "$fps" ]]; then
    filter="fps=${fps}"
  else
    filter="fps=1/${interval}"
  fi

  # Add crop filter if requested (crop bottom 85 pixels: 1920x1080 -> 1920x995)
  if [[ "$crop" == "true" ]]; then
    filter="${filter},crop=1920:995:0:0"
  fi

  output_pattern="$images_dir/${name}-%06d.${ext}"

  if [[ "$max_frames" != "0" ]]; then
    ffmpeg -hide_banner -loglevel error -i "$file" -vf "$filter" -frames:v "$max_frames" "$output_pattern" 2>&1
  else
    ffmpeg -hide_banner -loglevel error -i "$file" -vf "$filter" "$output_pattern" 2>&1
  fi

  mv "$file" "$done_dir/"
done

if [[ "$found_any" != "true" ]]; then
  echo "No video files found in $ingest_dir" >&2
  exit 1
fi
