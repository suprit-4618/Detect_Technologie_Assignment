"""
Problem 2 — Medium: Face Blur on Video Chunks / Images
=======================================================
Detects and blurs all visible human faces in short video clips
or static images.

Handles:
    - Partial or side-profile faces   (MediaPipe full-range model)
    - Noisy / low-resolution input    (low detection threshold + preprocessing)
    - Multiple faces in the same frame

Stack:
    - mediapipe  : Face detection (model_selection=1 handles side/partial faces)
    - opencv     : Video I/O, Gaussian blur, frame writing

Usage:
    # Blur faces in a video
    python face_blur.py input_video.mp4

    # Blur faces in an image
    python face_blur.py photo.jpg

    # Custom output path + show bounding boxes
    python face_blur.py input_video.mp4 -o output.mp4 --show-boxes

    # Stronger blur
    python face_blur.py input_video.mp4 -b 99

Time Complexity  (per frame): O(H × W) for detection + blur
Space Complexity (per frame): O(H × W) for frame buffer
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def get_face_detector(min_confidence: float = 0.3):
    """
    Create a MediaPipe FaceDetection instance.

    model_selection=1  → full-range model (up to 5 m), better for
                          partial/side-profile faces.
    min_detection_confidence → lowered to 0.3 to catch partially
                               occluded or low-res faces.
    """
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_confidence,
    )


def detect_faces(frame: np.ndarray, detector) -> list[tuple]:
    """
    Run MediaPipe face detection on a BGR frame.

    Returns:
        List of (x, y, w, h, confidence) in pixel coordinates.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    faces = []
    if results.detections:
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            conf = float(det.score[0])
            x = int(bb.xmin * w)
            y = int(bb.ymin * h)
            fw = int(bb.width * w)
            fh = int(bb.height * h)
            faces.append((x, y, fw, fh, conf))
    return faces


def blur_face(frame: np.ndarray, bbox: tuple, kernel: int = 51) -> np.ndarray:
    """
    Apply Gaussian blur to the face region (with 20 % padding).

    Args:
        frame  : BGR image/frame.
        bbox   : (x, y, w, h) in pixels.
        kernel : Gaussian kernel size (must be odd).

    Returns:
        Frame with the face region blurred in-place.
    """
    h_f, w_f = frame.shape[:2]
    x, y, w, h = bbox

    # Add 20 % padding to cover hair / chin
    px, py = int(w * 0.20), int(h * 0.20)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(w_f, x + w + px), min(h_f, y + h + py)

    if x2 <= x1 or y2 <= y1:
        return frame

    k = kernel | 1  # ensure odd
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame


# ---------------------------------------------------------------------------
# Processing pipelines
# ---------------------------------------------------------------------------

def process_image(input_path: str, output_path: str,
                  blur_strength: int, show_boxes: bool) -> bool:
    """Detect and blur faces in a single image."""
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {input_path}")
        return False

    detector = get_face_detector()
    faces = detect_faces(frame, detector)
    print(f"[INFO] Detected {len(faces)} face(s).")

    for (x, y, w, h, conf) in faces:
        frame = blur_face(frame, (x, y, w, h), blur_strength)
        if show_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(output_path, frame)
    print(f"[INFO] Saved → {output_path}")
    return True


def process_video(input_path: str, output_path: str,
                  blur_strength: int, show_boxes: bool) -> bool:
    """Detect and blur faces in every frame of a video."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = get_face_detector()
    frame_num = 0

    print(f"[INFO] Video: {width}×{height} @ {fps:.1f} fps — {total} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        faces = detect_faces(frame, detector)
        for (x, y, w, h, conf) in faces:
            frame = blur_face(frame, (x, y, w, h), blur_strength)
            if show_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame)

        if frame_num % 30 == 0 or frame_num == total:
            pct = (frame_num / total * 100) if total else 0
            print(f"[INFO] {frame_num}/{total} frames ({pct:.0f}%) — "
                  f"{len(faces)} face(s) this frame")

    cap.release()
    out.release()
    print(f"[INFO] Saved → {output_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_output_path(input_path: str, user_output: str | None) -> str:
    if user_output:
        return user_output
    p = Path(input_path)
    return str(p.parent / f"{p.stem}_blurred{p.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Face Blur on Video / Images using MediaPipe + OpenCV"
    )
    parser.add_argument("input", help="Path to input video or image file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path (auto-named if omitted)")
    parser.add_argument("-b", "--blur-strength", type=int, default=51,
                        help="Gaussian kernel size — must be odd (default: 51)")
    parser.add_argument("--show-boxes", action="store_true",
                        help="Draw bounding boxes around detected faces")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    output_path = build_output_path(args.input, args.output)
    ext = Path(args.input).suffix.lower()

    if ext in IMAGE_EXTS:
        success = process_image(args.input, output_path,
                                args.blur_strength, args.show_boxes)
    elif ext in VIDEO_EXTS:
        success = process_video(args.input, output_path,
                                args.blur_strength, args.show_boxes)
    else:
        print(f"[ERROR] Unsupported file type: {ext}")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
