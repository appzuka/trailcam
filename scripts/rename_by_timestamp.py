#!/usr/bin/env python3
"""
Utility to rename trail camera videos based on their embedded timestamp.
Extracts the timestamp from the first frame and renames files to trailcam_YY_MM_DD_HH_MM_SS.ext
"""

import argparse
import cv2
import os
import re
import shutil
from pathlib import Path
try:
    import pytesseract
except ImportError:
    print("Error: pytesseract is required. Install with: pip install pytesseract")
    print("Also ensure tesseract-ocr is installed on your system:")
    print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
    print("  macOS: brew install tesseract")
    exit(1)


def extract_timestamp_from_frame(frame, debug_path=None):
    """
    Extract timestamp from the bottom portion of a frame.
    Expected format: DD/MM/YYYY HH:MM:SS

    Args:
        frame: The video frame to process
        debug_path: Optional path to save the processed frame if timestamp extraction fails
    """
    # Get the bottom 15% of the frame where timestamp typically appears
    height = frame.shape[0]
    bottom_section = frame[int(height * 0.85):, :]

    # Convert to grayscale
    gray = cv2.cvtColor(bottom_section, cv2.COLOR_BGR2GRAY)

    # Increase contrast to make white text more distinct from background
    # Apply adaptive thresholding to handle semi-transparent overlays
    # This converts the image to pure black and white, making OCR more reliable
    _, high_contrast = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use pytesseract to extract text from high-contrast image
    text = pytesseract.image_to_string(high_contrast, config='--psm 6')

    # Look for timestamp pattern: DD/MM/YYYY HH:MM:SS
    # Match formats like: 02/01/2026 23:55:29
    pattern = r'(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2}):(\d{2})'
    match = re.search(pattern, text)

    if match:
        day, month, year, hour, minute, second = match.groups()
        # Convert to YY_MM_DD_HH_MM_SS format
        year_short = year[-2:]
        return f"{year_short}_{month}_{day}_{hour}_{minute}_{second}"

    # If timestamp not found and debug path provided, save the processed frame
    if debug_path:
        cv2.imwrite(debug_path, high_contrast)

    return None


def process_video(video_path, output_dir, save_debug_frames=False):
    """
    Process a single video file: extract timestamp and copy to renamed file.
    Tries successive frames if timestamp extraction fails.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save renamed files
        save_debug_frames: If True, save processed frames when OCR fails
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ❌ Could not open video: {video_path.name}")
            return False

        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Try extracting timestamp from frames until successful or exhausted
        timestamp = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Create debug path for this frame if debug mode is enabled
            debug_path = None
            if save_debug_frames:
                debug_filename = f"{video_path.stem}_frame_{frame_count:04d}.jpg"
                debug_path = output_dir / debug_filename

            timestamp = extract_timestamp_from_frame(frame, debug_path=debug_path)

            if timestamp:
                # Remove the debug file if timestamp was found
                if save_debug_frames and debug_path and debug_path.exists():
                    debug_path.unlink()
                break

        cap.release()

        if not timestamp:
            print(f"  ❌ Could not extract timestamp from any of {frame_count} frames: {video_path.name}")
            if save_debug_frames:
                print(f"     Debug frames saved to {output_dir}/{video_path.stem}_frame_*.jpg")
            return False

        if frame_count > 1:
            print(f"  ℹ️  Found timestamp in frame {frame_count}: {video_path.name}")

        # Create new filename
        extension = video_path.suffix
        new_filename = f"trailcam_{timestamp}{extension}"
        output_path = output_dir / new_filename

        # Copy file
        shutil.copy2(video_path, output_path)
        print(f"  ✓ {video_path.name} -> {new_filename}")
        return True

    except Exception as e:
        print(f"  ❌ Error processing {video_path.name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Rename trail camera videos based on embedded timestamps'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing video files to process'
    )
    parser.add_argument(
        '--debug-frames',
        action='store_true',
        help='Save processed frames when OCR fails (for debugging)'
    )

    args = parser.parse_args()

    # Validate directory
    input_dir = Path(args.directory)
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        exit(1)

    if not input_dir.is_dir():
        print(f"Error: Not a directory: {input_dir}")
        exit(1)

    # Create output directory
    output_dir = input_dir / "renamed"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f'*{ext}'))
        video_files.extend(input_dir.glob(f'*{ext.upper()}'))

    # Filter out hidden files (starting with '.')
    video_files = [f for f in video_files if not f.name.startswith('.')]

    if not video_files:
        print(f"No video files found in {input_dir}")
        exit(0)

    print(f"Found {len(video_files)} video file(s)")
    print()

    # Process each video
    success_count = 0
    for video_path in sorted(video_files):
        success = process_video(video_path, output_dir, save_debug_frames=args.debug_frames)
        if success:
            success_count += 1

    print()
    print(f"Successfully processed {success_count}/{len(video_files)} files")


if __name__ == "__main__":
    main()
