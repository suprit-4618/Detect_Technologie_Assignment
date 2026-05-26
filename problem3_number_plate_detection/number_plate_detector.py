"""
Problem 3 - Hard: Number Plate Detection
=========================================
Detects visible number plates in video/images and reads the plate text.

Expected Output (per detection):
    - Bounding box  : [x, y, width, height] in pixels
    - Plate text    : alphanumeric string read via OCR
    - Confidence    : float 0.0 - 1.0 from EasyOCR

Handles:
    - Angled plates          -> contour approxPolyDP + perspective correction
    - Partial occlusion      -> two detection methods combined (cascade + contour)
    - Varying lighting       -> bilateral filter + CLAHE preprocessing
    - Multiple plates/frame  -> all candidates evaluated; IoU-based deduplication

Stack:
    - opencv-python   : preprocessing, contour detection, drawing
    - easyocr         : OCR with per-text confidence scores
    - Haar cascade    : haarcascade_russian_plate_number.xml (OpenCV built-in)
                        as primary fast detector
    - Contour method  : aspect-ratio + area filtering as robust fallback

Usage:
    # Process a video
    python number_plate_detector.py input_video.mp4

    # Process an image
    python number_plate_detector.py car.jpg

    # Save JSON results alongside output
    python number_plate_detector.py input_video.mp4 --json

    # Use GPU for EasyOCR (if CUDA available)
    python number_plate_detector.py input_video.mp4 --gpu

    # Custom output
    python number_plate_detector.py input_video.mp4 -o detected.mp4

Time Complexity  (per frame): O(H x W) preprocessing + O(k) OCR candidates
Space Complexity (per frame): O(H x W) frame buffer
"""

import cv2
import numpy as np
import easyocr
import argparse
import os
import sys
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# Aspect ratio range for license plates  (width / height)
PLATE_ASPECT_MIN = 1.5
PLATE_ASPECT_MAX = 6.5

# Plate area relative to full frame
PLATE_AREA_MIN_RATIO = 0.002
PLATE_AREA_MAX_RATIO = 0.35

# IoU threshold for deduplication
IOU_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# NumberPlateDetector
# ---------------------------------------------------------------------------

class NumberPlateDetector:
    """
    Two-stage license plate detection and OCR pipeline.

    Stage 1 - Detection:
        • Haar cascade  (haarcascade_russian_plate_number.xml)
        • Contour-based aspect-ratio filter  (fallback / complement)

    Stage 2 - OCR:
        • EasyOCR on the cropped plate ROI (after perspective correction)
    """

    def __init__(self, languages: list = None, gpu: bool = False):
        if languages is None:
            languages = ["en"]

        print("[INFO] Loading EasyOCR reader …")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("[INFO] EasyOCR ready.")

        # Haar cascade - bundled with OpenCV
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_russian_plate_number.xml"
        )
        if os.path.exists(cascade_path):
            self.cascade = cv2.CascadeClassifier(cascade_path)
            print(f"[INFO] Haar cascade loaded: {cascade_path}")
        else:
            self.cascade = None
            print("[WARN] Haar cascade not found - using contour-only detection.")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale, denoise with bilateral filter, and
        enhance contrast with CLAHE.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    # ------------------------------------------------------------------
    # Detection - Method 1: Haar cascade
    # ------------------------------------------------------------------

    def _detect_cascade(self, gray: np.ndarray) -> list:
        if self.cascade is None:
            return []
        plates = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(50, 15),
            maxSize=(600, 250),
        )
        return [tuple(p) for p in plates] if len(plates) else []

    # ------------------------------------------------------------------
    # Detection - Method 2: Contour + aspect-ratio filter
    # ------------------------------------------------------------------

    def _detect_contour(self, frame: np.ndarray, gray: np.ndarray) -> list:
        h_f, w_f = frame.shape[:2]
        edges = cv2.Canny(gray, 30, 200)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:40]

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)

            # Accept 4-corner (rectangular) shapes
            if len(approx) < 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                continue

            aspect = w / h
            area_ratio = (w * h) / (w_f * h_f)

            if (PLATE_ASPECT_MIN <= aspect <= PLATE_ASPECT_MAX and
                    PLATE_AREA_MIN_RATIO <= area_ratio <= PLATE_AREA_MAX_RATIO):
                candidates.append((x, y, w, h))

        return candidates

    # ------------------------------------------------------------------
    # Perspective correction (deskew angled plates)
    # ------------------------------------------------------------------

    def _deskew_plate(self, roi: np.ndarray) -> np.ndarray:
        """
        Attempt to deskew a plate ROI using a four-point perspective
        transform.  Falls back to the original ROI if contours are unclear.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return roi

        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) != 4:
            return roi  # Cannot deskew; return original

        pts = approx.reshape(4, 2).astype(np.float32)
        # Order: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered = np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ], dtype=np.float32)

        (tl, tr, br, bl) = ordered
        w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        if w < 10 or h < 10:
            return roi

        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(roi, M, (w, h))

    # ------------------------------------------------------------------
    # OCR on a plate ROI
    # ------------------------------------------------------------------

    def _read_plate(self, plate_img: np.ndarray) -> tuple[str, float]:
        """
        Run EasyOCR on a plate image.

        Returns:
            (text, confidence)  where text is "" if nothing found.
        """
        # Upscale small plates for better OCR accuracy
        h, w = plate_img.shape[:2]
        scale = max(1, 200 // max(w, 1))
        if scale > 1:
            plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        results = self.reader.readtext(thresh, detail=1,
                                       allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                 "0123456789- ")
        if not results:
            return "", 0.0

        texts, confs = [], []
        for (_, text, conf) in results:
            cleaned = "".join(c for c in text.upper()
                              if c.isalnum() or c in "- ")
            if cleaned and conf > 0.1:
                texts.append(cleaned)
                confs.append(conf)

        if not texts:
            return "", 0.0

        return " ".join(texts), float(np.mean(confs))

    # ------------------------------------------------------------------
    # IoU deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(b1: list, b2: list) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ix1, iy1 = max(x1, x2), max(y1, y2)
        ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union else 0.0

    def _deduplicate(self, detections: list) -> list:
        detections = sorted(detections, key=lambda d: d["confidence"],
                            reverse=True)
        kept = []
        for det in detections:
            if not any(self._iou(det["bbox"], k["bbox"]) > IOU_THRESHOLD
                       for k in kept):
                kept.append(det)
        return kept

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect number plates and read text in a single frame.

        Returns:
            List of dicts: {"bbox": [x,y,w,h], "text": str, "confidence": float}
        """
        gray = self._preprocess(frame)
        h_f, w_f = frame.shape[:2]

        # Merge candidates from both detectors
        candidates = []
        seen_keys: set = set()

        for (x, y, w, h) in self._detect_cascade(gray):
            key = (x // 15, y // 15)
            if key not in seen_keys:
                candidates.append((x, y, w, h))
                seen_keys.add(key)

        for (x, y, w, h) in self._detect_contour(frame, gray):
            key = (x // 15, y // 15)
            if key not in seen_keys:
                candidates.append((x, y, w, h))
                seen_keys.add(key)

        detections = []
        for (x, y, w, h) in candidates:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_f, x + w)
            y2 = min(h_f, y + h)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_corrected = self._deskew_plate(roi)
            text, conf = self._read_plate(roi_corrected)

            if text:
                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "text": text,
                    "confidence": round(conf, 4),
                })

        return self._deduplicate(detections)

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Annotate frame with bounding boxes, text, and confidence scores."""
        for det in detections:
            x, y, w, h = det["bbox"]
            text = det["text"]
            conf = det["confidence"]
            label = f"{text}  ({conf:.2f})"

            # Green box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           0.65, 2)
            cv2.rectangle(frame, (x, y - lh - 12), (x + lw + 4, y),
                          (0, 255, 0), cv2.FILLED)

            # Label text (black on green)
            cv2.putText(frame, label, (x + 2, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        return frame


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------

def process_image(detector: NumberPlateDetector,
                  input_path: str, output_path: str,
                  save_json: bool) -> None:
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {input_path}")
        return

    print(f"[INFO] Processing image: {input_path}")
    detections = detector.detect(frame)
    _print_detections(detections)

    annotated = detector.draw(frame.copy(), detections)
    cv2.imwrite(output_path, annotated)
    print(f"[INFO] Saved -> {output_path}")

    if save_json:
        _save_json({"file": input_path, "detections": detections}, output_path)


def process_video(detector: NumberPlateDetector,
                  input_path: str, output_path: str,
                  save_json: bool, skip_frames: int = 5) -> None:
    """
    Process a video file for number plate detection.

    skip_frames: Run OCR every N frames; re-use last result for in-between
    frames. This is standard practice for real-time ANPR on CPU.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps - {total} frames")
    print(f"[INFO] OCR running every {skip_frames} frames (CPU optimisation)")

    all_detections = {}
    frame_num = 0
    last_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Run full detection every skip_frames; reuse result in between
        if frame_num % skip_frames == 1:
            last_detections = detector.detect(frame)
        detections = last_detections

        if detections:
            all_detections[frame_num] = detections

        annotated = detector.draw(frame.copy(), detections)
        out.write(annotated)

        if frame_num % 30 == 0 or frame_num == total:
            pct = (frame_num / total * 100) if total else 0
            plates_str = ", ".join(
                f"'{d['text']}' ({d['confidence']:.2f})" for d in detections
            ) or "-"
            print(f"[INFO] Frame {frame_num}/{total} ({pct:.0f}%) | "
                  f"Plates: {plates_str}")

    cap.release()
    out.release()
    print(f"[INFO] Saved -> {output_path}")

    if save_json:
        _save_json({"file": input_path, "frames": all_detections}, output_path)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _print_detections(detections: list) -> None:
    print(f"[INFO] Found {len(detections)} plate(s).")
    for i, d in enumerate(detections, 1):
        print(f"  [{i}] Text: '{d['text']}'  |  "
              f"Confidence: {d['confidence']:.4f}  |  "
              f"BBox: {d['bbox']}")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def _save_json(data: dict, output_path: str) -> None:
    json_path = Path(output_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    print(f"[INFO] JSON results -> {json_path}")


def _auto_output(input_path: str, user_output: str | None) -> str:
    if user_output:
        return user_output
    p = Path(input_path)
    return str(p.parent / f"{p.stem}_detected{p.suffix}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Number Plate Detection - OpenCV + EasyOCR"
    )
    parser.add_argument("input", help="Input video or image path")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path (auto-named if omitted)")
    parser.add_argument("--json", action="store_true",
                        help="Save detection results as JSON")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for EasyOCR (requires CUDA)")
    parser.add_argument("--lang", nargs="+", default=["en"],
                        help="EasyOCR language codes (default: en)")
    parser.add_argument("--skip-frames", type=int, default=5,
                        help="Run OCR every N frames (default: 5, use 1 for every frame)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    output_path = _auto_output(args.input, args.output)
    detector = NumberPlateDetector(languages=args.lang, gpu=args.gpu)
    ext = Path(args.input).suffix.lower()

    if ext in IMAGE_EXTS:
        process_image(detector, args.input, output_path, args.json)
    elif ext in VIDEO_EXTS:
        process_video(detector, args.input, output_path, args.json, skip_frames=args.skip_frames)
    else:
        print(f"[ERROR] Unsupported file type: {ext}")
        sys.exit(1)


if __name__ == "__main__":
    main()
