import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from modules.objective.utils import extract_audio, is_silent_audio, metric_result, resolve_path, run_json_command


def compute_audio_aesthetic(sample: Dict[str, Any], config: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    metric = "audio_aesthetic"
    project_dir = Path(config["project_dir"])
    models = config.get("models", {})
    scripts = config.get("scripts", {})
    ckpt = resolve_path(project_dir, models.get("audiobox_checkpoint"))
    script = resolve_path(project_dir, scripts.get("audiobox_batch"))
    if not ckpt or not ckpt.exists():
        return metric_result(metric, None, f"missing AudioBox checkpoint: {ckpt}")
    if not script or not script.exists():
        return metric_result(metric, None, f"missing AudioBox batch script: {script}")

    # Skip if edited video has silent audio (model gives meaningless default scores)
    if is_silent_audio(sample["edited_video_path"]):
        return metric_result(metric, 0.0, "edited video audio is silent")

    try:
        audio_dir = work_dir / "audio_aesthetic_wav"
        audio_path = extract_audio(sample["edited_video_path"], str(audio_dir / f"{Path(sample['edited_video_path']).stem}.wav"))
        temp_audio_dir = work_dir / "audio_aesthetic_input"
        temp_audio_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(audio_path, temp_audio_dir / Path(audio_path).name)
        output_json = work_dir / "audio_aesthetic_result.json"
        cmd = [sys.executable, str(script), "--audio_dir", str(temp_audio_dir), "--output", str(output_json), "--ckpt", str(ckpt), "--batch_size", "1"]
        run_json_command(cmd)
        data = json.load(open(output_json, "r", encoding="utf-8"))
        first = (data.get("results") or [{}])[0]
        scores = first.get("scores", first)
        pq = float(scores["PQ"])
        cu = float(scores["CU"])
        return metric_result(metric, (pq + cu) / 20.0, details={"PQ": pq, "CU": cu})
    except Exception as exc:
        return metric_result(metric, None, str(exc))

