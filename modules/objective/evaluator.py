import importlib
from pathlib import Path
from typing import Any, Dict, List

from modules.objective.utils import OBJECTIVE_METRICS, aggregate_objective_scores, get_active_metrics, normalize_category


METRIC_MODULES = {
    "lip_sync": "modules.objective.cross_modal.lip_sync",
    "av_sync": "modules.objective.cross_modal.av_sync",
    "video_aesthetic": "modules.objective.video_quality.video_aesthetic",
    "subject_consistency": "modules.objective.video_quality.subject_consistency",
    "motion_smoothness": "modules.objective.video_quality.motion_smoothness",
    "audio_aesthetic": "modules.objective.audio_quality.audio_aesthetic",
    "speech_quality": "modules.objective.audio_quality.speech_quality",
    "videoclip_video_similarity": "modules.objective.preservation.videoclip_video_similarity",
    "audio_similarity": "modules.objective.preservation.audio_similarity",
}


class ObjectiveEvaluator:
    def __init__(self, project_dir: Path, output_dir: Path, config: Dict[str, Any], device: str = "cpu") -> None:
        self.project_dir = Path(project_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        objective_cfg = config.get("objective", {})
        self.config = {
            "project_dir": str(self.project_dir),
            "device": device,
            "models": objective_cfg.get("models", {}),
            "scripts": objective_cfg.get("scripts", {}),
            "settings": objective_cfg.get("settings", {}),
        }

    def evaluate_sample(self, sample: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        category = normalize_category(sample.get("category"))
        active_metrics = get_active_metrics(category)
        output_dir.mkdir(parents=True, exist_ok=True)

        scores = {metric: None for metric in OBJECTIVE_METRICS}
        metric_details: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []

        for metric in active_metrics:
            result = self._compute_metric(metric, sample, output_dir / metric)
            scores[metric] = result.get("score")
            metric_details[metric] = result
            if result.get("warning"):
                warnings.append(f"{metric}: {result['warning']}")

        return {
            "category": category,
            "active_metrics": active_metrics,
            "objective_scores": scores,
            "objective_metric_details": metric_details,
            "warnings": warnings,
        }

    def aggregate(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return aggregate_objective_scores(rows)

    def _compute_metric(self, metric: str, sample: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
        work_dir.mkdir(parents=True, exist_ok=True)
        module = importlib.import_module(METRIC_MODULES[metric])
        func = getattr(module, f"compute_{metric}")
        return func(sample, self.config, work_dir)
