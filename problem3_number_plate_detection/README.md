# Problem 3 — Number Plate Detection (Hard)

Detects visible number plates in videos or images, reads the plate text, and outputs bounding boxes with confidence scores.

## Stack

| Library | Role |
|---|---|
| **OpenCV** | Preprocessing, contour detection, Haar cascade, drawing |
| **EasyOCR** | OCR with per-segment confidence scores |
| **NumPy** | Array operations |

### Why this stack?
- **Haar cascade** (`haarcascade_russian_plate_number.xml`) bundled inside OpenCV — zero extra download, fast first-pass detection.
- **Contour + aspect-ratio filter** complements the cascade: catches plates the cascade misses (unusual angles, sizes).
- **EasyOCR** outperforms raw Tesseract on real-world plates; returns per-text confidence natively.
- **Perspective correction** (four-point transform) straightens angled plates before OCR.

---

## Setup

```bash
pip install -r requirements.txt
```

> First run downloads EasyOCR model weights (~100 MB) automatically.

---

## Usage

```bash
# Detect plates in a video
python number_plate_detector.py input_video.mp4

# Detect plates in an image
python number_plate_detector.py car.jpg

# Save JSON results (bbox + text + confidence per frame)
python number_plate_detector.py input_video.mp4 --json

# Specify output file
python number_plate_detector.py input_video.mp4 -o result.mp4

# GPU acceleration for EasyOCR (requires CUDA)
python number_plate_detector.py input_video.mp4 --gpu

# Multi-language plates (e.g., English + Hindi)
python number_plate_detector.py input_video.mp4 --lang en hi
```

---

## Output

- **Annotated video/image** with green bounding boxes, plate text, and confidence.
- **JSON** (optional, `--json`): per-frame list of `{bbox, text, confidence}`.

### Sample JSON output
```json
[
  {
    "bbox": [312, 540, 180, 55],
    "text": "MH 12 AB 1234",
    "confidence": 0.8731
  }
]
```

---

## Edge Cases Handled

| Scenario | How |
|---|---|
| Angled / tilted plates | Perspective correction via `getPerspectiveTransform` |
| Partial occlusion | Two detection methods (cascade + contour) combined |
| Varying lighting | Bilateral filter + CLAHE contrast enhancement |
| Multiple plates/frame | All candidates processed; IoU deduplication removes overlaps |
| Small/distant plates | ROI upscaled before OCR for better accuracy |

---

## Complexity

- **Time:** O(H × W) per frame for preprocessing + O(k) OCR per k plate candidates
- **Space:** O(H × W) frame buffer; EasyOCR model weights in memory (~100 MB)
