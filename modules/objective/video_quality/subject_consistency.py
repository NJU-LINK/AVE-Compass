import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from modules.objective.utils import metric_result, resolve_path, run_json_command


def compute_subject_consistency(sample: Dict[str, Any], config: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    metric = "subject_consistency"
    project_dir = Path(config["project_dir"])
    model = resolve_path(project_dir, config.get("models", {}).get("dinov2_model"))
    script = resolve_path(project_dir, config.get("scripts", {}).get("subject_consistency_batch"))
    if not model or not model.exists():
        return metric_result(metric, None, f"missing DINOv2 model: {model}")
    if not script or not script.exists():
        return metric_result(metric, None, f"missing subject consistency batch script: {script}")

    try:
        input_dir = work_dir / "subject_consistency_input"
        input_dir.mkdir(parents=True, exist_ok=True)
        video_copy = input_dir / Path(sample["edited_video_path"]).name
        if not video_copy.exists():
            shutil.copy(sample["edited_video_path"], video_copy)
        output_json = work_dir / "subject_consistency_result.json"
        cmd = [
            sys.executable,
            str(script),
            "--video_dir",
            str(input_dir),
            "--output",
            str(output_json),
            "--model_path",
            str(model),
            "--num_frames",
            str(config.get("settings", {}).get("subject_consistency_num_frames", 50)),
            "--device",
            config.get("device", "cpu"),
        ]
        run_json_command(cmd)
        data = json.load(open(output_json, "r", encoding="utf-8"))
        first = (data.get("results") or [{}])[0]
        score = first.get("score")
        if score is None:
            return metric_result(metric, None, first.get("error", "subject consistency inference failed"))
        return metric_result(metric, float(score), details=first.get("details", first))
    except Exception as exc:
        return metric_result(metric, None, str(exc))
