import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from modules.objective.utils import metric_result, resolve_path, run_json_command


def compute_motion_smoothness(sample: Dict[str, Any], config: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    metric = "motion_smoothness"
    project_dir = Path(config["project_dir"])
    cfg = resolve_path(project_dir, config.get("models", {}).get("amt_s_config"))
    ckpt = resolve_path(project_dir, config.get("models", {}).get("amt_s_checkpoint"))
    script = resolve_path(project_dir, config.get("scripts", {}).get("motion_smoothness_batch"))
    if not script or not script.exists():
        return metric_result(metric, None, f"missing motion smoothness batch script: {script}")

    try:
        input_dir = work_dir / "motion_smoothness_input"
        input_dir.mkdir(parents=True, exist_ok=True)
        video_copy = input_dir / Path(sample["edited_video_path"]).name
        if not video_copy.exists():
            shutil.copy(sample["edited_video_path"], video_copy)
        output_json = work_dir / "motion_smoothness_result.json"
        if not cfg or not cfg.exists():
            return metric_result(metric, None, f"missing AMT config: {cfg}")
        if not ckpt or not ckpt.exists():
            return metric_result(metric, None, f"missing AMT checkpoint: {ckpt}")
        cmd = [
            sys.executable,
            str(script),
            "--video_dir",
            str(input_dir),
            "--output",
            str(output_json),
            "--config",
            str(cfg),
            "--checkpoint",
            str(ckpt),
            "--device",
            config.get("device", "cpu"),
        ]
        run_json_command(cmd)
        data = json.load(open(output_json, "r", encoding="utf-8"))
        first = (data.get("results") or [{}])[0]
        score = first.get("score")
        if score is None:
            return metric_result(metric, None, first.get("error", "motion smoothness inference failed"))
        return metric_result(metric, float(score), details=first.get("details", first))
    except Exception as exc:
        return metric_result(metric, None, str(exc))
