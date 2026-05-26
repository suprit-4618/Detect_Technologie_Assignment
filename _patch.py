with open('problem3_number_plate_detection/number_plate_detector.py', 'r', encoding='utf-8') as f:
    src = f.read()

old_save = (
    'def _save_json(data: dict, output_path: str) -> None:\n'
    '    json_path = Path(output_path).with_suffix(".json")\n'
    '    with open(json_path, "w") as f:\n'
    '        json.dump(data, f, indent=2)\n'
    '    print(f"[INFO] JSON results \u2192 {json_path}")'
)

new_save = (
    'class _NumpyEncoder(json.JSONEncoder):\n'
    '    def default(self, obj):\n'
    '        if isinstance(obj, np.integer): return int(obj)\n'
    '        if isinstance(obj, np.floating): return float(obj)\n'
    '        if isinstance(obj, np.ndarray): return obj.tolist()\n'
    '        return super().default(obj)\n'
    '\n'
    '\n'
    'def _save_json(data: dict, output_path: str) -> None:\n'
    '    json_path = Path(output_path).with_suffix(".json")\n'
    '    with open(json_path, "w") as f:\n'
    '        json.dump(data, f, indent=2, cls=_NumpyEncoder)\n'
    '    print(f"[INFO] JSON results -> {json_path}")'
)

if old_save in src:
    src = src.replace(old_save, new_save)
    print("Found and replaced _save_json")
else:
    print("Pattern not found, replacing unicode chars only")

# Replace all Unicode special chars with ASCII
src = src.replace('\u2192', '->').replace('\u2014', '-').replace('\u00d7', 'x').replace('\u2013', '-')

with open('problem3_number_plate_detection/number_plate_detector.py', 'w', encoding='utf-8') as f:
    f.write(src)
print('Done')
