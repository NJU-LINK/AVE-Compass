import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from modules.objective.utils import metric_result, resolve_path, run_json_command


def compute_lip_sync(sample: Dict[str, Any], config: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    """Compute lip-sync confidence score using SyncNet.

    Only applicable to speech-category tasks. The SyncNet confidence
    (median_dist - min_dist) is normalized to [0, 1] via a configurable
    threshold (lip_sync_conf_threshold, default 2.0).
    """
    metric = "lip_sync"
    project_dir = Path(config["project_dir"])
    models = config.get("models", {})
    scripts = config.get("scripts", {})
    settings = config.get("settings", {})

    syncnet_model = resolve_path(project_dir, models.get("syncnet_model"))
    script = resolve_path(project_dir, scripts.get("lipsync_batch"))
    conf_threshold = float(settings.get("lip_sync_conf_threshold", 2.0))

    if not syncnet_model or not syncnet_model.exists():
        return metric_result(metric, None, f"missing SyncNet model: {syncnet_model}")
    if not script or not script.exists():
        return metric_result(metric, None, f"missing lip-sync batch script: {script}")

    try:
        input_dir = (work_dir / "lip_sync_input").resolve()
        input_dir.mkdir(parents=True, exist_ok=True)
        video_copy = input_dir / Path(sample["edited_video_path"]).name
        if not video_copy.exists():
            shutil.copy(sample["edited_video_path"], video_copy)

        output_json = (work_dir / "lip_sync_result.json").resolve()
        temp_dir = (work_dir / "temp_lipsync").resolve()

        cmd = [
            sys.executable,
            str(script),
            "--video_dir", str(input_dir),
            "--model_path", str(syncnet_model),
            "--output_file", str(output_json),
            "--device", config.get("device", "cpu"),
            "--temp_dir", str(temp_dir),
        ]
        run_json_command(cmd, cwd=script.parent)

        data = json.load(open(output_json, "r", encoding="utf-8"))
        first = (data.get("results") or [{}])[0]

        if first.get("error"):
            return metric_result(metric, None, f"SyncNet error: {first['error']}")

        confidence = first.get("confidence")
        offset = first.get("av_offset")
        min_dist = first.get("min_dist")

        if confidence is None:
            return metric_result(metric, None, "SyncNet returned null confidence")

        confidence = float(confidence)
        # Normalize confidence to [0, 1]: higher confidence = better lip sync
        score = max(0.0, min(1.0, confidence / conf_threshold))

        return metric_result(
            metric,
            score,
            details={
                "confidence": confidence,
                "av_offset": offset,
                "min_dist": min_dist,
                "conf_threshold": conf_threshold,
            },
        )
    except Exception as exc:
        return metric_result(metric, None, str(exc))
