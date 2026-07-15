import json
import sys
from pathlib import Path
from typing import Any, Dict

from modules.objective.utils import metric_result, resolve_path, run_json_command


def compute_visual_video_similarity(metric: str, sample: Dict[str, Any], config: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    project_dir = Path(config["project_dir"])
    model = resolve_path(project_dir, config.get("models", {}).get("videoclip_xl"))
    script = resolve_path(project_dir, config.get("scripts", {}).get("video_similarity_batch"))
    if not model or not model.exists():
        return metric_result(metric, None, f"missing VideoCLIP-XL model: {model}")
    if not script or not script.exists():
        return metric_result(metric, None, f"missing video similarity batch script: {script}")

    try:
        output_json = work_dir / f"{metric}_result.json"
        cmd = [
            sys.executable,
            str(script),
            "--source_video",
            str(sample["source_video_path"]),
            "--edited_video",
            str(sample["edited_video_path"]),
            "--output",
            str(output_json),
            "--model_path",
            str(model),
            "--metric",
            metric,
            "--num_frames",
            str(config.get("settings", {}).get("videoclip_num_frames", 8)),
            "--device",
            config.get("device", "cpu"),
        ]
        open_clip_model = config.get("settings", {}).get("open_clip_model")
        if open_clip_model:
            cmd.extend(["--open_clip_model", str(open_clip_model)])
        run_json_command(cmd)
        data = json.load(open(output_json, "r", encoding="utf-8"))
        first = (data.get("results") or [{}])[0]
        score = first.get("score")
        if score is None:
            return metric_result(metric, None, first.get("error", "video similarity inference failed"))
        value = max(-1.0, min(1.0, float(score)))
        return {"metric": metric, "score": value, "details": first.get("details", first)}
    except Exception as exc:
        return metric_result(metric, None, str(exc))
