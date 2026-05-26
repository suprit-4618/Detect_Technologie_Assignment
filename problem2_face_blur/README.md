# Problem 2 — Face Blur on Video Chunks / Images

Detects and blurs **all visible human faces** in short video clips or static images.

## Stack
| Library | Role |
|---|---|
| **MediaPipe** | Face detection — handles partial/side-profile faces |
| **OpenCV** | Video I/O, Gaussian blur, frame writing |

### Why MediaPipe?
MediaPipe's `FaceDetection` (model_selection=1) is a lightweight, real-time model trained to handle:
- Full-frontal faces
- Side-profile / partial faces
- Multiple faces in the same frame
- Varying lighting and low-resolution input

A lower detection threshold (`min_detection_confidence=0.3`) is used to catch partially occluded faces.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Blur faces in a video
python face_blur.py input_video.mp4

# Blur faces in an image
python face_blur.py photo.jpg

# Specify output path
python face_blur.py input_video.mp4 -o output_blurred.mp4

# Show bounding boxes (for debugging)
python face_blur.py input_video.mp4 --show-boxes

# Stronger blur (kernel must be odd)
python face_blur.py input_video.mp4 -b 99
```

---

## Output
- Output file saved to the same folder as input, suffixed with `_blurred`
- e.g. `input_video.mp4` → `input_video_blurred.mp4`

---

## Edge Cases Handled
| Scenario | How |
|---|---|
| Partial / side-profile faces | MediaPipe full-range model (model_selection=1) |
| Multiple faces in frame | All detections iterated and blurred |
| Noisy / low-res input | Detection threshold lowered to 0.3; CLAHE not needed at detection stage |
| Face near frame edges | Bbox clamped to frame dimensions before blurring |

---

## Complexity
- **Time:** O(H × W) per frame for detection and Gaussian blur
- **Space:** O(H × W) for the frame buffer; no extra memory scales with frame count
