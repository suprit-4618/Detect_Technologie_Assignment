with open('problem3_number_plate_detection/number_plate_detector.py', 'r', encoding='utf-8') as f:
    src = f.read()

old = (
    'def process_video(detector: NumberPlateDetector,\n'
    '                  input_path: str, output_path: str,\n'
    '                  save_json: bool) -> None:\n'
    '    cap = cv2.VideoCapture(input_path)\n'
    '    if not cap.isOpened():\n'
    '        print(f"[ERROR] Cannot open video: {input_path}")\n'
    '        return\n'
    '\n'
    '    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0\n'
    '    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n'
    '    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n'
    '    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n'
    '\n'
    '    fourcc = cv2.VideoWriter_fourcc(*"mp4v")\n'
    '    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))\n'
    '\n'
    '    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps - {total} frames")\n'
    '\n'
    '    all_detections = {}\n'
    '    frame_num = 0\n'
    '\n'
    '    while True:\n'
    '        ret, frame = cap.read()\n'
    '        if not ret:\n'
    '            break\n'
    '        frame_num += 1\n'
    '\n'
    '        detections = detector.detect(frame)\n'
    '        if detections:\n'
    '            all_detections[frame_num] = detections\n'
    '\n'
    '        annotated = detector.draw(frame.copy(), detections)\n'
    '        out.write(annotated)\n'
    '\n'
    '        if frame_num % 30 == 0 or frame_num == total:\n'
    '            pct = (frame_num / total * 100) if total else 0\n'
    '            plates_str = ", ".join(\n'
    "                f\"'{d['text']}' ({d['confidence']:.2f})\" for d in detections\n"
    '            ) or "-"\n'
    '            print(f"[INFO] Frame {frame_num}/{total} ({pct:.0f}%) | "\n'
    '                  f"Plates: {plates_str}")\n'
    '\n'
    '    cap.release()\n'
    '    out.release()\n'
    '    print(f"[INFO] Saved -> {output_path}")\n'
    '\n'
    '    if save_json:\n'
    '        _save_json({"file": input_path, "frames": all_detections}, output_path)'
)

new = (
    'def process_video(detector: NumberPlateDetector,\n'
    '                  input_path: str, output_path: str,\n'
    '                  save_json: bool, skip_frames: int = 5) -> None:\n'
    '    """\n'
    '    Process a video file for number plate detection.\n'
    '\n'
    '    skip_frames: Run OCR every N frames; re-use last result for in-between\n'
    '    frames. This is standard practice for real-time ANPR on CPU.\n'
    '    """\n'
    '    cap = cv2.VideoCapture(input_path)\n'
    '    if not cap.isOpened():\n'
    '        print(f"[ERROR] Cannot open video: {input_path}")\n'
    '        return\n'
    '\n'
    '    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0\n'
    '    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n'
    '    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n'
    '    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n'
    '\n'
    '    fourcc = cv2.VideoWriter_fourcc(*"mp4v")\n'
    '    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))\n'
    '\n'
    '    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps - {total} frames")\n'
    '    print(f"[INFO] OCR running every {skip_frames} frames (CPU optimisation)")\n'
    '\n'
    '    all_detections = {}\n'
    '    frame_num = 0\n'
    '    last_detections = []\n'
    '\n'
    '    while True:\n'
    '        ret, frame = cap.read()\n'
    '        if not ret:\n'
    '            break\n'
    '        frame_num += 1\n'
    '\n'
    '        # Run full detection every skip_frames; reuse result in between\n'
    '        if frame_num % skip_frames == 1:\n'
    '            last_detections = detector.detect(frame)\n'
    '        detections = last_detections\n'
    '\n'
    '        if detections:\n'
    '            all_detections[frame_num] = detections\n'
    '\n'
    '        annotated = detector.draw(frame.copy(), detections)\n'
    '        out.write(annotated)\n'
    '\n'
    '        if frame_num % 30 == 0 or frame_num == total:\n'
    '            pct = (frame_num / total * 100) if total else 0\n'
    '            plates_str = ", ".join(\n'
    "                f\"'{d['text']}' ({d['confidence']:.2f})\" for d in detections\n"
    '            ) or "-"\n'
    '            print(f"[INFO] Frame {frame_num}/{total} ({pct:.0f}%) | "\n'
    '                  f"Plates: {plates_str}")\n'
    '\n'
    '    cap.release()\n'
    '    out.release()\n'
    '    print(f"[INFO] Saved -> {output_path}")\n'
    '\n'
    '    if save_json:\n'
    '        _save_json({"file": input_path, "frames": all_detections}, output_path)'
)

if old in src:
    src = src.replace(old, new)
    print("process_video updated with skip_frames")
else:
    print("Pattern not found!")

# Also add --skip-frames CLI arg
old_cli = (
    '    parser.add_argument("--lang", nargs="+", default=["en"],\n'
    '                        help="EasyOCR language codes (default: en)")\n'
    '    args = parser.parse_args()'
)
new_cli = (
    '    parser.add_argument("--lang", nargs="+", default=["en"],\n'
    '                        help="EasyOCR language codes (default: en)")\n'
    '    parser.add_argument("--skip-frames", type=int, default=5,\n'
    '                        help="Run OCR every N frames (default: 5, use 1 for every frame)")\n'
    '    args = parser.parse_args()'
)
if old_cli in src:
    src = src.replace(old_cli, new_cli)
    print("CLI --skip-frames added")
else:
    print("CLI pattern not found!")

# And pass skip_frames to process_video call
old_call = 'process_video(detector, args.input, output_path, args.json)'
new_call = 'process_video(detector, args.input, output_path, args.json, skip_frames=args.skip_frames)'
if old_call in src:
    src = src.replace(old_call, new_call)
    print("process_video call updated")

with open('problem3_number_plate_detection/number_plate_detector.py', 'w', encoding='utf-8') as f:
    f.write(src)
print("Done")
