import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from modules.objective.utils import extract_audio, is_silent_audio, metric_result, read_first_csv_row, resolve_path, run_json_command


def compute_speech_quality(sample: Dict[str, Any], config: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    metric = "speech_quality"
    project_dir = Path(config["project_dir"])
    models = config.get("models", {})
    scripts = config.get("scripts", {})
    nisqa_model = resolve_path(project_dir, models.get("nisqa_model"))
    nisqa_script = resolve_path(project_dir, scripts.get("nisqa_predict"))
    if not nisqa_model or not nisqa_model.exists():
        return metric_result(metric, None, f"missing NISQA model: {nisqa_model}")
    if not nisqa_script or not nisqa_script.exists():
        return metric_result(metric, None, f"missing NISQA run_predict.py: {nisqa_script}")

    # Skip if edited video has silent audio (speech quality is meaningless)
    if is_silent_audio(sample["edited_video_path"]):
        return metric_result(metric, 0.0, "edited video audio is silent")

    try:
        audio_path = extract_audio(sample["edited_video_path"], str(work_dir / "speech_quality_wav" / f"{Path(sample['edited_video_path']).stem}.wav"))
        input_dir = work_dir / "speech_quality_input"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(audio_path, input_dir / Path(audio_path).name)
        output_dir = work_dir / "speech_quality_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(nisqa_script),
            "--mode",
            "predict_dir",
            "--pretrained_model",
            str(nisqa_model),
            "--data_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--num_workers",
            "0",
            "--bs",
            "1",
        ]
        run_json_command(cmd)
        row = read_first_csv_row(output_dir / "NISQA_results.csv")
        mos = float(row.get("mos_pred") or row.get("mos") or row.get("MOS"))
        return metric_result(metric, (mos - 1.0) / 4.0, details={"MOS": mos})
    except Exception as exc:
        return metric_result(metric, None, str(exc))

