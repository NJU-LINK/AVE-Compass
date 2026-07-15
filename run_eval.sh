#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_ROOT="${1:-${ROOT_DIR}/input}"
MODEL_NAME="${2:-model}"
OUTPUT_PATH="${3:-${ROOT_DIR}/output}"
CONFIG_PATH="${4:-${ROOT_DIR}/configs/path.yml}"
DEVICE="${5:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - "${INPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = ("videos", "edit_instructions", "checklists", "edited_videos")
missing = [name for name in required if not (root / name).is_dir()]
if missing:
    raise SystemExit(f"Missing input directories: {', '.join(missing)}")

instructions = sorted((root / "edit_instructions").glob("*.json"))
if not instructions:
    raise SystemExit("No instruction JSON files found in edit_instructions/")

errors = []
for instruction_path in instructions:
    try:
        instruction = json.loads(instruction_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        errors.append(f"{instruction_path.name}: invalid JSON ({exc})")
        continue

    source_name = str(instruction.get("video", "")).strip()
    if not source_name or not (root / "videos" / source_name).is_file():
        errors.append(f"{instruction_path.name}: missing source video {source_name or '<video field>'}")

    stem = instruction_path.stem
    if not (root / "checklists" / f"{stem}_checklist.json").is_file():
        errors.append(f"{instruction_path.name}: missing checklists/{stem}_checklist.json")
    if not (root / "edited_videos" / f"{stem}.mp4").is_file():
        errors.append(f"{instruction_path.name}: missing edited_videos/{stem}.mp4")

if errors:
    raise SystemExit("Input validation failed:\n- " + "\n- ".join(errors))

print(f"Validated {len(instructions)} samples in {root}")
PY

exec "${PYTHON_BIN}" "${ROOT_DIR}/metrics/evaluate.py" \
  --input_root "${INPUT_ROOT}" \
  --model_name "${MODEL_NAME}" \
  --output_path "${OUTPUT_PATH}" \
  --name "${MODEL_NAME}_eval" \
  --device "${DEVICE}" \
  --config "${CONFIG_PATH}"
