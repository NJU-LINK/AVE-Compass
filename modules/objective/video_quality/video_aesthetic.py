import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from modules.objective.utils import metric_result, resolve_path, run_json_command


def compute_video_aesthetic(sample: Dict[str, Any], config: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    metric = "video_aesthetic"
    project_dir = Path(config["project_dir"])
    models = config.get("models", {})
    scripts = config.get("scripts", {})
    model = resolve_path(project_dir, models.get("aesthetic_predictor"))
    script = resolve_path(project_dir, scripts.get("video_aesthetic_batch"))
    if not model or not model.exists():
        return metric_result(metric, None, f"missing Aesthetic Predictor v2.5 model: {model}")
    if not script or not script.exists():
        return metric_result(metric, None, f"missing video aesthetic batch script: {script}")
    try:
        input_dir = work_dir / "video_aesthetic_input"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(sample["edited_video_path"], input_dir / Path(sample["edited_video_path"]).name)
        output_json = work_dir / "video_aesthetic_result.json"
        cmd = [
            sys.executable,
            str(script),
            "--video_dir",
            str(input_dir),
            "--output",
            str(output_json),
            "--num_frames",
            str(config.get("settings", {}).get("video_aesthetic_num_frames", 10)),
            "--device",
            config.get("device", "cpu"),
            "--model_path",
            str(model),
        ]
        run_json_command(cmd)
        data = json.load(open(output_json, "r", encoding="utf-8"))
        first = (data.get("results") or [{}])[0]
        raw = first.get("score", first.get("video_aesthetic"))
        return metric_result(metric, float(raw) / 10.0, details={"raw_score": raw})
    except Exception as exc:
        return metric_result(metric, None, str(exc))
