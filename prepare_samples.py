"""
prepare_samples.py
==================
Downloads / generates sample media for testing Problem 2 and Problem 3.

Problem 2 (Face Blur):
    Downloads a public-domain portrait from Wikimedia Commons,
    then creates a 5-second test video from it.

Problem 3 (Number Plate Detection):
    Generates a synthetic video using OpenCV — a car-shaped rectangle
    with a clearly drawn license plate moving across the frame.
    This reliably triggers the contour/OCR detector without needing
    real footage.

Run from the repo root:
    python prepare_samples.py
"""

import cv2
import numpy as np
import urllib.request
import urllib.error
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
P2_DIR   = os.path.join("problem2_face_blur")
P3_DIR   = os.path.join("problem3_number_plate_detection")

P2_IMG   = os.path.join(P2_DIR, "sample_input.jpg")
P2_VIDEO = os.path.join(P2_DIR, "sample_input.mp4")
P3_VIDEO = os.path.join(P3_DIR, "sample_input.mp4")

# ---------------------------------------------------------------------------
# Public-domain face images (Wikimedia Commons) — tried in order
# ---------------------------------------------------------------------------
FACE_URLS = [
    # Albert Einstein (public domain) - 375px valid thumbnail
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/375px-Albert_Einstein_Head.jpg",
    # Abraham Lincoln (public domain portrait) - 320px valid thumbnail
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Abraham_Lincoln_O-77_matte_collodion_print.jpg/320px-Abraham_Lincoln_O-77_matte_collodion_print.jpg",
    # Barack Obama official portrait (public domain, US govt) - 399px valid thumbnail
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/399px-President_Barack_Obama.jpg",
]


# ===========================================================================
# Problem 2 — face image + video
# ===========================================================================

def download_face_image() -> bool:
    """Try each URL until one succeeds. Returns True if an image was saved."""
    for url in FACE_URLS:
        try:
            print(f"[P2] Downloading face image from:\n     {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            with open(P2_IMG, "wb") as f:
                f.write(data)
            # Verify OpenCV can read it
            img = cv2.imread(P2_IMG)
            if img is not None:
                print(f"[P2] Saved face image -> {P2_IMG}  ({img.shape[1]}x{img.shape[0]})")
                return True
        except Exception as e:
            print(f"[P2] Failed ({e}), trying next URL …")
    return False


def make_synthetic_face_image(path: str) -> np.ndarray:
    """
    Generate a synthetic portrait-style image with two oval 'faces'
    (skin-tone ellipses with basic facial features) so MediaPipe has
    something to detect.  Used only if all downloads fail.
    """
    img = np.full((480, 640, 3), (80, 100, 60), dtype=np.uint8)  # dark bg

    def draw_face(canvas, cx, cy, rx, ry):
        # Skin tone
        cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, (180, 140, 110), -1)
        # Eyes
        for ex in [cx - rx // 3, cx + rx // 3]:
            cv2.ellipse(canvas, (ex, cy - ry // 5), (rx // 8, ry // 10),
                        0, 0, 360, (40, 30, 20), -1)
        # Mouth
        cv2.ellipse(canvas, (cx, cy + ry // 3), (rx // 4, ry // 10),
                    0, 0, 180, (120, 60, 60), 2)
        # Nose bridge
        cv2.line(canvas, (cx, cy - ry // 8), (cx, cy + ry // 8), (150, 110, 90), 2)

    draw_face(img, 220, 240, 90, 110)
    draw_face(img, 430, 240, 70,  90)

    cv2.imwrite(path, img)
    print(f"[P2] Synthetic face image created -> {path}")
    return img


def create_face_video(img: np.ndarray, out_path: str, seconds: int = 6):
    """Pan + zoom the image slightly to make a realistic-looking clip."""
    h, w = img.shape[:2]
    fps   = 25
    total = fps * seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for i in range(total):
        t = i / total
        # Gentle pan: shift up to 30 px right
        dx = int(30 * t)
        M  = np.float32([[1, 0, dx], [0, 1, 0]])
        frame = cv2.warpAffine(img, M, (w, h))

        # Overlay frame counter (subtle)
        cv2.putText(frame, f"frame {i+1}/{total}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        writer.write(frame)

    writer.release()
    print(f"[P2] Test video created ({seconds}s @ {fps} fps) -> {out_path}")


# ===========================================================================
# Problem 3 — synthetic license plate video
# ===========================================================================

PLATE_SPECS = [
    # (plate_text,  bg_color BGR,   text_color BGR)
    ("MH 12 AB 1234",  (255, 255, 255), (0,   0,   0)),   # White plate
    ("DL 01 CD 5678",  (0,   255, 255), (0,   0,   0)),   # Yellow plate
    ("KA 09 EF 9012",  (255, 255, 255), (0,   0, 180)),   # White / red text
]


def draw_plate(canvas: np.ndarray, x: int, y: int,
               text: str, bg: tuple, fg: tuple) -> np.ndarray:
    """Draw a realistic-ish license plate on the canvas."""
    pw, ph = 310, 80

    # Plate background + border
    cv2.rectangle(canvas, (x, y), (x + pw, y + ph), bg, -1)
    cv2.rectangle(canvas, (x, y), (x + pw, y + ph), (0, 0, 0), 3)

    # Left blue strip (EU-style)
    cv2.rectangle(canvas, (x, y), (x + 35, y + ph), (200, 50, 0), -1)
    cv2.putText(canvas, "IN", (x + 4, y + ph - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Plate text
    cv2.putText(canvas, text, (x + 44, y + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, fg, 2)
    return canvas


def draw_car(canvas: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Draw a very simple top-down or front-facing car silhouette."""
    # Car body
    cv2.rectangle(canvas, (cx - 160, cy - 100), (cx + 160, cy + 120),
                  (40, 60, 140), -1)
    # Roof
    cv2.rectangle(canvas, (cx - 90,  cy - 160), (cx + 90,  cy - 100),
                  (30, 45, 110), -1)
    # Windows
    cv2.rectangle(canvas, (cx - 80,  cy - 150), (cx + 80,  cy - 110),
                  (180, 210, 220), -1)
    # Headlights
    for lx in [cx - 130, cx + 90]:
        cv2.rectangle(canvas, (lx, cy - 70), (lx + 40, cy - 40),
                      (220, 220, 180), -1)
    # Wheels
    for wx in [cx - 120, cx + 80]:
        cv2.ellipse(canvas, (wx, cy + 120), (30, 30), 0, 0, 360, (20, 20, 20), -1)
    return canvas


def create_plate_video(out_path: str, seconds: int = 8):
    """
    Generates a video with a car moving slowly left-to-right.
    Three different plates are shown (~2.5 s each).
    The plate is large and high-contrast so EasyOCR has a good chance.
    """
    W, H   = 1280, 720
    fps    = 25
    total  = fps * seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    for i in range(total):
        # Dark road background
        canvas = np.full((H, W, 3), (60, 55, 50), dtype=np.uint8)

        # Road markings
        for lx in range(0, W, 80):
            cv2.rectangle(canvas, (lx, H // 2 + 20), (lx + 40, H // 2 + 30),
                          (200, 200, 200), -1)

        # Car position — slow rightward drift
        cx = int(-160 + (W + 320) * (i / total))
        cy = H // 2 - 30

        # Which plate to show
        spec_idx = min(i // (total // 3), 2)
        txt, bg, fg = PLATE_SPECS[spec_idx]

        # Draw car body
        draw_car(canvas, cx, cy)

        # Draw plate centred below headlights
        px = cx - 155
        py = cy + 50
        draw_plate(canvas, px, py, txt, bg, fg)

        # Annotation overlay
        cv2.putText(canvas,
                    f"Plate {spec_idx + 1}/3: {txt}  |  frame {i + 1}/{total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        writer.write(canvas)

    writer.release()
    print(f"[P3] Test video created ({seconds}s @ {fps} fps) -> {out_path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    os.makedirs(P2_DIR, exist_ok=True)
    os.makedirs(P3_DIR, exist_ok=True)

    # --- Problem 2 ---
    print("\n" + "=" * 60)
    print("PROBLEM 2 — Face Blur sample media")
    print("=" * 60)

    downloaded = download_face_image()
    if downloaded:
        img = cv2.imread(P2_IMG)
    else:
        print("[P2] All downloads failed. Generating synthetic face image …")
        img = make_synthetic_face_image(P2_IMG)

    create_face_video(img, P2_VIDEO, seconds=6)

    # --- Problem 3 ---
    print("\n" + "=" * 60)
    print("PROBLEM 3 — Number Plate Detection sample media")
    print("=" * 60)

    create_plate_video(P3_VIDEO, seconds=9)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DONE — Sample files ready:")
    for path in [P2_IMG, P2_VIDEO, P3_VIDEO]:
        size_kb = os.path.getsize(path) // 1024 if os.path.exists(path) else 0
        print(f"  {path}  ({size_kb} KB)")
    print("=" * 60)
    print("\nNext steps:")
    print("  # Test face blur")
    print(f"  python {P2_DIR}/face_blur.py {P2_VIDEO}")
    print()
    print("  # Test number plate detection")
    print(f"  python {P3_DIR}/number_plate_detector.py {P3_VIDEO}")
