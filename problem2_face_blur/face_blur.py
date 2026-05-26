"""
Problem 2 — Medium: Face Blur on Video Chunks / Images
=======================================================
Detects and blurs all visible human faces in short video clips
or static images.

Handles:
    - Partial or side-profile faces   (frontal cascade + profile cascade)
    - Noisy / low-resolution input    (bilateral filter preprocessing + multi-scale)
    - Multiple faces in the same frame

Stack:
    - opencv-python : Haar cascades (frontalface + profileface), video I/O,
                      Gaussian blur, frame writing. Zero external model downloads.

Why OpenCV Haar Cascades?
    - haarcascade_frontalface_default.xml  — catches full-frontal and near-frontal
    - haarcascade_profileface.xml          — catches left/right side-profile faces
    - Running both + merging with IoU-dedup gives broad coverage
    - Works offline, no model download, fast

Usage:
    python face_blur.py input_video.mp4
    python face_blur.py photo.jpg
    python face_blur.py input_video.mp4 -o blurred.mp4 --show-boxes
    python face_blur.py input_video.mp4 -b 99

Time Complexity  (per frame): O(H x W) preprocessing + O(H x W) cascade scan
Space Complexity (per frame): O(H x W) frame buffer
"""

import cv2
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

IOU_MERGE_THRESHOLD = 0.3   # Merge overlapping detections above this IoU


# ---------------------------------------------------------------------------
# Detector setup
# ---------------------------------------------------------------------------

def load_detectors():
    """
    Load OpenCV Haar cascade detectors.
    Returns (frontal_detector, profile_detector).
    """
    hc = cv2.data.haarcascades
    frontal = cv2.CascadeClassifier(
        os.path.join(hc, "haarcascade_frontalface_default.xml")
    )
    profile = cv2.CascadeClassifier(
        os.path.join(hc, "haarcascade_profileface.xml")
    )
    return frontal, profile


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def preprocess_for_detection(frame: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale and apply equalisation + bilateral filter
    to handle low-resolution / noisy / poorly-lit input.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.equalizeHist(gray)
    return gray


def detect_faces_cascade(gray: np.ndarray,
                         frontal_det, profile_det) -> list:
    """
    Run both frontal and profile cascades.
    Also mirrors the frame and runs profile again to catch right-profile.

    Returns a list of (x, y, w, h) in pixel coords.
    """
    params = dict(
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30, 30),
    )

    # Frontal
    frontal_faces = frontal_det.detectMultiScale(gray, **params)

    # Left profile
    profile_left = profile_det.detectMultiScale(gray, **params)

    # Right profile (mirror image)
    mirrored = cv2.flip(gray, 1)
    profile_right_raw = profile_det.detectMultiScale(mirrored, **params)
    w_frame = gray.shape[1]
    profile_right = []
    for (x, y, w, h) in profile_right_raw:
        profile_right.append((w_frame - x - w, y, w, h))

    faces = []
    for det in [frontal_faces, profile_left, profile_right]:
        if len(det):
            for face in det:
                faces.append(tuple(face))

    return _merge_overlapping(faces)


def _iou(b1, b2) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union else 0.0


def _merge_overlapping(faces: list) -> list:
    """Merge/deduplicate overlapping bounding boxes."""
    merged = []
    used = [False] * len(faces)
    for i, f in enumerate(faces):
        if used[i]:
            continue
        used[i] = True
        group = [f]
        for j, g in enumerate(faces):
            if not used[j] and _iou(f, g) > IOU_MERGE_THRESHOLD:
                group.append(g)
                used[j] = True
        # Average the group
        xs = [b[0] for b in group]
        ys = [b[1] for b in group]
        ws = [b[2] for b in group]
        hs = [b[3] for b in group]
        merged.append((
            int(np.mean(xs)), int(np.mean(ys)),
            int(np.mean(ws)), int(np.mean(hs))
        ))
    return merged


# ---------------------------------------------------------------------------
# Blurring
# ---------------------------------------------------------------------------

def blur_face(frame: np.ndarray, bbox: tuple, kernel: int = 51) -> np.ndarray:
    """
    Apply Gaussian blur to the face region with 20 % padding.

    Args:
        frame  : BGR image
        bbox   : (x, y, w, h) pixels
        kernel : Gaussian kernel size (odd number)

    Returns:
        Frame with blurred face region.
    """
    h_f, w_f = frame.shape[:2]
    x, y, w, h = bbox
    px, py = int(w * 0.20), int(h * 0.20)
    x1 = max(0, x - px);  y1 = max(0, y - py)
    x2 = min(w_f, x + w + px);  y2 = min(h_f, y + h + py)

    if x2 <= x1 or y2 <= y1:
        return frame

    k = max(3, kernel) | 1   # ensure odd
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame


# ---------------------------------------------------------------------------
# Processing pipelines
# ---------------------------------------------------------------------------

def process_image(frontal_det, profile_det,
                  input_path: str, output_path: str,
                  blur_strength: int, show_boxes: bool) -> bool:
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {input_path}")
        return False

    gray  = preprocess_for_detection(frame)
    faces = detect_faces_cascade(gray, frontal_det, profile_det)
    print(f"[INFO] Detected {len(faces)} face(s).")

    for (x, y, w, h) in faces:
        frame = blur_face(frame, (x, y, w, h), blur_strength)
        if show_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_path, frame)
    print(f"[INFO] Saved -> {output_path}")
    return True


def process_video(frontal_det, profile_det,
                  input_path: str, output_path: str,
                  blur_strength: int, show_boxes: bool) -> bool:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps | {total} frames")
    frame_num  = 0
    total_hits = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        gray  = preprocess_for_detection(frame)
        faces = detect_faces_cascade(gray, frontal_det, profile_det)
        total_hits += len(faces)

        for (x, y, w, h) in faces:
            frame = blur_face(frame, (x, y, w, h), blur_strength)
            if show_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "face", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame)

        if frame_num % 30 == 0 or frame_num == total:
            pct = (frame_num / total * 100) if total else 0
            print(f"[INFO] {frame_num}/{total} frames ({pct:.0f}%) | "
                  f"{len(faces)} face(s) this frame")

    cap.release()
    out.release()
    print(f"[INFO] Total detections across all frames: {total_hits}")
    print(f"[INFO] Saved -> {output_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def auto_output(input_path: str, user_output) -> str:
    if user_output:
        return user_output
    p = Path(input_path)
    return str(p.parent / f"{p.stem}_blurred{p.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Face Blur on Video / Images — OpenCV Haar Cascades"
    )
    parser.add_argument("input",   help="Input video or image file")
    parser.add_argument("-o", "--output",       default=None)
    parser.add_argument("-b", "--blur-strength", type=int, default=51,
                        help="Gaussian kernel size (odd, default 51)")
    parser.add_argument("--show-boxes", action="store_true",
                        help="Draw bounding boxes (debugging)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Not found: {args.input}")
        sys.exit(1)

    frontal_det, profile_det = load_detectors()
    output_path = auto_output(args.input, args.output)
    ext = Path(args.input).suffix.lower()

    if ext in IMAGE_EXTS:
        ok = process_image(frontal_det, profile_det, args.input, output_path,
                           args.blur_strength, args.show_boxes)
    elif ext in VIDEO_EXTS:
        ok = process_video(frontal_det, profile_det, args.input, output_path,
                           args.blur_strength, args.show_boxes)
    else:
        print(f"[ERROR] Unsupported type: {ext}")
        sys.exit(1)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
