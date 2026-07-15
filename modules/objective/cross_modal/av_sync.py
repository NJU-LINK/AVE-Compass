import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from modules.objective.utils import is_silent_audio, metric_result, resolve_path, run_json_command


def compute_av_sync(sample: Dict[str, Any], config: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    metric = "av_sync"
    project_dir = Path(config["project_dir"])
    scripts = config.get("scripts", {})
    script = resolve_path(project_dir, scripts.get("synchformer_batch"))
    model_dir = resolve_path(project_dir, config.get("models", {}).get("synchformer_model_dir"))
    tau = float(config.get("settings", {}).get("av_sync_tau", 2.0))
    if not script or not script.exists():
        return metric_result(metric, None, f"missing Synchformer batch script: {script}")

    # Skip if edited video has silent audio (sync prediction is meaningless)
    if is_silent_audio(sample["edited_video_path"]):
        return metric_result(metric, 0.0, "edited video audio is silent")

    try:
        input_dir = (work_dir / "av_sync_input").resolve()
        input_dir.mkdir(parents=True, exist_ok=True)
        video_copy = input_dir / Path(sample["edited_video_path"]).name
        if not video_copy.exists():
            shutil.copy(sample["edited_video_path"], video_copy)
        output_json = (work_dir / "av_sync_result.json").resolve()
        cmd = [sys.executable, str(script), "--video_dir", str(input_dir), "--output_file", str(output_json), "--device", config.get("device", "cpu")]
        if model_dir:
            cmd.extend(["--model_base", str(model_dir)])
        run_json_command(cmd, cwd=script.parent)
        data = json.load(open(output_json, "r", encoding="utf-8"))
        first = (data.get("results") or [{}])[0]
        delta = first.get("predicted_offset_sec", first.get("delta_t"))
        if delta is None:
            return metric_result(metric, None, "Synchformer returned null offset (likely no detectable audio-visual sync)")
        delta = float(delta)
        return metric_result(metric, max(0.0, 1.0 - abs(delta) / tau), details={"delta_t": delta, "tau": tau})
    except Exception as exc:
        return metric_result(metric, None, str(exc))
