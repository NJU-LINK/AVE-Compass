#!/usr/bin/env python3
"""Unified AVE-Compass evaluation entrypoint."""

import argparse
import csv
import datetime
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

try:
    import torch
except ImportError:
    torch = None


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from modules.objective.evaluator import ObjectiveEvaluator  # noqa: E402
from modules.objective.utils import OBJECTIVE_METRICS, aggregate_objective_scores, get_active_metrics, normalize_category  # noqa: E402


def _setup_logger(output_path: Path) -> logging.Logger:
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = output_path / f"{timestamp}_ave_compass.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("AVECompass")


def _convert_types(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_types(i) for i in obj]
    return obj


def _safe_mean(values: List[Any]) -> Optional[float]:
    valid = []
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isnan(float(value)):
            valid.append(float(value))
    return float(sum(valid) / len(valid)) if valid else None


class AVECompassEvaluator:
    def __init__(self, device: Any, output_path: str, config_path: str):
        self.device = device
        self.output_path = Path(output_path).resolve()
        self.logger = _setup_logger(self.output_path)
        self.config_path = Path(config_path).resolve()
        self.config = self._load_config()
        modules_cfg = self.config.get("modules", {})
        self.objective_enabled = bool(modules_cfg.get("objective", True))
        self.llm_enabled = bool(modules_cfg.get("llm", False))
        llm_cfg = self.config.get("llm", {})
        self.checklist_enabled = self.llm_enabled and bool(llm_cfg.get("checklist", True))
        self.realism_enabled = self.llm_enabled and bool(llm_cfg.get("realism", True))
        self.objective_evaluator = None
        if self.objective_enabled:
            self.objective_evaluator = ObjectiveEvaluator(
                project_dir=PROJECT_DIR,
                output_dir=self.output_path / "objective",
                config=self.config,
                device=str(device),
            )
        self.llm_evaluator = None
        self.realism_evaluator = None
        if self.checklist_enabled:
            from modules.llm.evaluator import VideoEditEvaluator

            self.llm_evaluator = VideoEditEvaluator(
                output_dir=str(self.output_path / "llm"),
                num_frames=int(llm_cfg.get("num_frames", 8)),
                use_full_video=bool(llm_cfg.get("use_full_video", True)),
            )
        if self.realism_enabled:
            from modules.llm.realism_evaluator import RealismEvaluator

            shared_client = self.llm_evaluator.llm_client if self.llm_evaluator is not None else None
            self.realism_evaluator = RealismEvaluator(
                llm_client=shared_client,
            )

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_json_file(self, path: Optional[Path], default: Any) -> Any:
        if not path or not path.exists():
            return default
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def _caption_text_from_data(self, caption_data: Any) -> str:
        if not isinstance(caption_data, dict):
            return ""
        if caption_data.get("overall"):
            return str(caption_data.get("overall") or "")
        parts = []
        for key in ("caption", "audio_caption", "speech_or_non_speech", "visual_style"):
            if caption_data.get(key):
                parts.append(f"{key}: {caption_data.get(key)}")
        return "\n".join(parts)

    def _normalize_edited_id(self, stem: str) -> str:
        match = re.match(r"^(.*)_(\d+)$", stem)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        return stem

    def _instruction_category(self, instruction: Dict[str, Any]) -> str:
        for key in ("category_label", "taxonomy_label", "task_category", "category_type", "objective_category"):
            if instruction.get(key):
                return normalize_category(instruction.get(key))
        return normalize_category(instruction.get("category"))

    def _build_samples_from_flat_model_dir(self, model_dir: Path, model_name: str) -> List[Dict[str, Any]]:
        prompt_dir = model_dir / "edit_instructions"
        source_dir = model_dir / "videos"
        edited_dir = model_dir / "edited_videos"
        checklist_dir = model_dir / "checklists"
        samples: List[Dict[str, Any]] = []

        for prompt_path in sorted(prompt_dir.glob("*.json")):
            prompt_data = self._load_json_file(prompt_path, {})
            if not isinstance(prompt_data, dict):
                continue

            instruction = prompt_data.get("instruction", {})
            if not isinstance(instruction, dict):
                instruction = {}

            video_name = str(prompt_data.get("video") or "")
            if not video_name:
                raise ValueError(f"Instruction file is missing the video field: {prompt_path.name}")
            task_id = str(prompt_data.get("task") or Path(video_name).stem or prompt_path.stem)
            source_video = source_dir / video_name
            edited_video = edited_dir / f"{prompt_path.stem}.mp4"
            checklist_path = checklist_dir / f"{prompt_path.stem}_checklist.json"
            missing_paths = [path for path in (source_video, edited_video, checklist_path) if not path.is_file()]
            if missing_paths:
                missing_text = ", ".join(str(path.relative_to(model_dir)) for path in missing_paths)
                raise FileNotFoundError(f"Missing files for {prompt_path.name}: {missing_text}")

            checklist_data = self._load_json_file(checklist_path, {})
            caption_data = {"overall": checklist_data.get("caption", "")} if isinstance(checklist_data, dict) else {}

            instruction_index = prompt_data.get("instruction_index")
            edit_prompt = str(
                instruction.get("prompt_en")
                or instruction.get("prompt")
                or prompt_data.get("prompt_en")
                or prompt_data.get("prompt")
                or ""
            )
            sample = {
                "id": f"{model_name}/{self._normalize_edited_id(edited_video.stem)}",
                "sample_id": f"{model_name}/{self._normalize_edited_id(edited_video.stem)}",
                "instruction_id": instruction.get("instruction_id", instruction.get("id", instruction_index)),
                "model_name": model_name,
                "case_name": task_id,
                "source_video_path": str(source_video.resolve()),
                "edited_video_path": str(edited_video.resolve()),
                "src_video_name": source_video.name,
                "edited_video_name": edited_video.name,
                "source_caption": self._caption_text_from_data(caption_data),
                "edit_prompt": edit_prompt,
                "prompt_index": instruction_index,
                "prompt_path": str(prompt_path.resolve()),
                "instruction": instruction,
                "category": self._instruction_category(instruction),
            }
            sample["checklist_path"] = str(checklist_path.resolve())
            samples.append(sample)
        return samples

    def _build_samples_from_input_root(self, input_root: Path, model_name: str) -> List[Dict[str, Any]]:
        if not input_root.exists():
            raise FileNotFoundError(f"Input root not found: {input_root}")
        required = ("videos", "edit_instructions", "checklists", "edited_videos")
        missing = [name for name in required if not (input_root / name).is_dir()]
        if missing:
            raise ValueError(f"Input root is missing required directories: {', '.join(missing)}")
        return self._build_samples_from_flat_model_dir(input_root, model_name)

    def evaluate_from_input_root(
        self,
        input_root: str,
        model_name: str,
        name: str = "ave_compass_eval",
    ) -> Dict[str, Any]:
        samples = self._build_samples_from_input_root(Path(input_root).resolve(), model_name)
        self.logger.info("Discovered %d edited videos", len(samples))
        if not samples:
            raise ValueError("No valid edited samples were found in the four input directories")
        return self._evaluate_samples(samples, name)

    def _evaluate_samples(self, samples: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        summary_name = f"{name}_{timestamp}"
        per_sample: List[Dict[str, Any]] = []
        by_model: Dict[str, List[Dict[str, Any]]] = {}

        for sample in samples:
            model_name = sample.get("model_name") or "unknown"
            edited_stem = self._normalize_edited_id(Path(sample["edited_video_path"]).stem)
            sample_dir = self.output_path / model_name / edited_stem
            sample_dir.mkdir(parents=True, exist_ok=True)

            existing_result_path = sample_dir / "eval_result.json"
            if existing_result_path.exists():
                try:
                    existing = json.load(open(existing_result_path, "r", encoding="utf-8"))
                    objective_done = not self.objective_enabled or bool(existing.get("objective_scores"))
                    checklist_done = not self.checklist_enabled or (
                        bool(existing.get("checklist", {}).get("success"))
                        and bool(existing.get("checklist", {}).get("metrics"))
                    )
                    realism_done = not self.realism_enabled or bool(existing.get("realism", {}).get("success"))
                    if objective_done and checklist_done and realism_done:
                        per_sample.append(existing)
                        by_model.setdefault(model_name, []).append(existing)
                        continue
                except Exception:
                    pass

            objective = self._evaluate_objective_sample(sample, sample_dir)
            checklist = self._evaluate_checklist_sample(sample, sample_dir)
            realism = self._evaluate_realism_sample(sample, sample_dir)

            row = {
                **sample,
                "category": objective["category"],
                "active_metrics": objective["active_metrics"],
                "objective_scores": objective["objective_scores"],
                "objective_metric_details": objective["objective_metric_details"],
                "checklist": checklist,
                "realism": realism,
                "warnings": objective["warnings"],
                "output_dir": str(sample_dir),
            }
            if checklist.get("warning"):
                row["warnings"].append(f"checklist: {checklist['warning']}")
            if realism.get("warning"):
                row["warnings"].append(f"realism: {realism['warning']}")
            row["scores"] = {
                "objective_score": None,
                "checklist_score": checklist.get("score"),
                "checklist_raw_score": checklist.get("raw_score"),
                "realism_score": realism.get("score"),
                "realism_raw_score": realism.get("raw_score"),
            }
            with open(sample_dir / "eval_result.json", "w", encoding="utf-8") as f:
                json.dump(_convert_types(row), f, ensure_ascii=False, indent=2)
            per_sample.append(row)
            by_model.setdefault(model_name, []).append(row)

        aggregation = {
            "objective": aggregate_objective_scores(per_sample),
            "checklist": self._aggregate_checklist_scores(per_sample),
            "realism": self._aggregate_realism_scores(per_sample),
        }
        self._save_outputs(summary_name, per_sample, aggregation)
        for model_name, rows in by_model.items():
            self._save_model_summary(
                model_name,
                rows,
                summary_name,
                {
                    "objective": aggregate_objective_scores(rows),
                    "checklist": self._aggregate_checklist_scores(rows),
                    "realism": self._aggregate_realism_scores(rows),
                },
            )

        return {
            "name": summary_name,
            "total_samples": len(per_sample),
            "aggregation": aggregation,
            "results": per_sample,
        }

    def _evaluate_objective_sample(self, sample: Dict[str, Any], sample_dir: Path) -> Dict[str, Any]:
        if self.objective_enabled and self.objective_evaluator is not None:
            return self.objective_evaluator.evaluate_sample(sample, sample_dir / "objective")
        category = normalize_category(sample.get("category"))
        return {
            "category": category,
            "active_metrics": get_active_metrics(category),
            "objective_scores": {metric: None for metric in OBJECTIVE_METRICS},
            "objective_metric_details": {},
            "warnings": [],
        }

    def _evaluate_checklist_sample(self, sample: Dict[str, Any], sample_dir: Path) -> Dict[str, Any]:
        if not self.checklist_enabled or self.llm_evaluator is None:
            return {"enabled": False, "score": None, "raw_score": None}

        llm_dir = sample_dir / "llm"
        llm_dir.mkdir(parents=True, exist_ok=True)
        provided_checklist = sample.get("checklist_path")

        try:
            if not provided_checklist or not Path(str(provided_checklist)).exists():
                return {
                    "enabled": True,
                    "success": False,
                    "score": None,
                    "raw_score": None,
                    "warning": "checklist file is required for MLLM checklist evaluation",
                }
            checklist_path = Path(str(provided_checklist)).resolve()

            result = self.llm_evaluator.evaluate_with_checklist(
                checklist_path=str(checklist_path),
                source_video_path=str(sample["source_video_path"]),
                edited_video_path=str(sample["edited_video_path"]),
                output_dir=str(llm_dir),
            )
            score_info = self._extract_checklist_score(result)
            metrics = self._extract_checklist_metrics(result)
            result_path = llm_dir / "llm_evaluation.json"
            result_path.write_text(json.dumps(_convert_types(result), ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "enabled": True,
                "success": bool(result.get("success")),
                "score": score_info["score"],
                "raw_score": score_info["raw_score"],
                "score_source": score_info["score_source"],
                "metrics": metrics,
                "checklist_path": str(checklist_path),
                "evaluation_path": str(result_path),
            }
        except Exception as exc:
            return {"enabled": True, "success": False, "score": None, "raw_score": None, "warning": str(exc)}

    def _evaluate_realism_sample(self, sample: Dict[str, Any], sample_dir: Path) -> Dict[str, Any]:
        if not self.realism_enabled or self.realism_evaluator is None:
            return {"enabled": False, "score": None, "raw_score": None}

        realism_dir = sample_dir / "llm" / "realism"
        realism_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = self.realism_evaluator.evaluate(
                edited_video_path=str(sample["edited_video_path"]),
                edit_prompt=str(sample.get("edit_prompt", "") or ""),
                caption=str(sample.get("source_caption", "") or ""),
            )
            result_path = realism_dir / "realism_evaluation.json"
            result_path.write_text(json.dumps(_convert_types(result), ensure_ascii=False, indent=2), encoding="utf-8")
            overall = (result.get("scores") or {}).get("overall")
            return {
                "enabled": True,
                **result,
                "score": float(overall) / 100.0 if isinstance(overall, (int, float)) else None,
                "raw_score": overall,
                "evaluation_path": str(result_path),
            }
        except Exception as exc:
            return {"enabled": True, "success": False, "score": None, "raw_score": None, "warning": str(exc)}

    def _extract_checklist_score(self, result: Dict[str, Any]) -> Dict[str, Any]:
        evaluation = result.get("evaluation", {}) if isinstance(result, dict) else {}
        checklist_summary = evaluation.get("checklist_summary", {}) if isinstance(evaluation, dict) else {}
        raw = checklist_summary.get("editing_intent")
        source = "checklist_summary.editing_intent"
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return {"raw_score": float(raw), "score": max(0.0, min(1.0, float(raw) / 100.0)), "score_source": source}
        return {"raw_score": None, "score": None, "score_source": source}

    def _extract_checklist_metrics(self, result: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
        evaluation = result.get("evaluation", {}) if isinstance(result, dict) else {}
        summary = evaluation.get("checklist_summary", {}) if isinstance(evaluation, dict) else {}
        dimension_scores = summary.get("dimension_scores", {}) if isinstance(summary, dict) else {}
        dimension_modality = evaluation.get("dimension_modality", {}) if isinstance(evaluation, dict) else {}

        def dimension_metric(name: str) -> Dict[str, Optional[float]]:
            return {
                "overall": (dimension_scores.get(name) or {}).get("score"),
                "video": ((dimension_modality.get(name) or {}).get("video") or {}).get("score"),
                "audio": ((dimension_modality.get(name) or {}).get("audio") or {}).get("score"),
            }

        response = summary.get("edit_response", {}) if isinstance(summary, dict) else {}
        intent_by_modality = summary.get("editing_intent_by_modality", {}) if isinstance(summary, dict) else {}
        return {
            "editing_intent": {
                "overall": summary.get("editing_intent"),
                "video": intent_by_modality.get("video"),
                "audio": intent_by_modality.get("audio"),
            },
            "instruction_following": dimension_metric("Instruction Following"),
            "fidelity_preserving": dimension_metric("Fidelity"),
            "edit_response": {
                "overall": response.get("score"),
                "video": (response.get("video") or {}).get("score"),
                "audio": (response.get("audio") or {}).get("score"),
            },
        }

    def _aggregate_checklist_scores(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        overall = _safe_mean([row.get("scores", {}).get("checklist_score") for row in rows])
        by_model: Dict[str, Optional[float]] = {}
        by_category: Dict[str, Optional[float]] = {}
        for model in sorted({row.get("model_name") or "unknown" for row in rows}):
            by_model[model] = _safe_mean(
                [row.get("scores", {}).get("checklist_score") for row in rows if (row.get("model_name") or "unknown") == model]
            )
        for category in sorted({row.get("category") or "unknown" for row in rows}):
            by_category[category] = _safe_mean(
                [row.get("scores", {}).get("checklist_score") for row in rows if (row.get("category") or "unknown") == category]
            )
        missing = sum(1 for row in rows if row.get("scores", {}).get("checklist_score") is None)

        metric_names = ("editing_intent", "instruction_following", "fidelity_preserving", "edit_response")
        dimensions = ("overall", "video", "audio")

        def metric_summary(selected: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
            return {
                metric: {
                    dimension: _safe_mean(
                        [
                            (((row.get("checklist") or {}).get("metrics") or {}).get(metric) or {}).get(dimension)
                            for row in selected
                        ]
                    )
                    for dimension in dimensions
                }
                for metric in metric_names
            }

        metrics = metric_summary(rows)
        metrics_by_model = {
            model: metric_summary([row for row in rows if (row.get("model_name") or "unknown") == model])
            for model in sorted({row.get("model_name") or "unknown" for row in rows})
        }
        metrics_by_category = {
            category: metric_summary([row for row in rows if (row.get("category") or "unknown") == category])
            for category in sorted({row.get("category") or "unknown" for row in rows})
        }
        return {
            "overall": overall,
            "by_model": by_model,
            "by_category": by_category,
            "missing_count": missing,
            "metrics": metrics,
            "metrics_by_model": metrics_by_model,
            "metrics_by_category": metrics_by_category,
        }

    def _aggregate_realism_scores(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def values(selected: List[Dict[str, Any]], dimension: str) -> List[Any]:
            return [((row.get("realism") or {}).get("scores") or {}).get(dimension) for row in selected]

        dimensions = ("overall", "video", "audio")
        overall = {dimension: _safe_mean(values(rows, dimension)) for dimension in dimensions}
        by_model = {}
        for model in sorted({row.get("model_name") or "unknown" for row in rows}):
            selected = [row for row in rows if (row.get("model_name") or "unknown") == model]
            by_model[model] = {dimension: _safe_mean(values(selected, dimension)) for dimension in dimensions}
        by_category = {}
        for category in sorted({row.get("category") or "unknown" for row in rows}):
            selected = [row for row in rows if (row.get("category") or "unknown") == category]
            by_category[category] = {dimension: _safe_mean(values(selected, dimension)) for dimension in dimensions}
        missing = {
            dimension: sum(value is None for value in values(rows, dimension))
            for dimension in dimensions
        }
        return {"overall": overall, "by_model": by_model, "by_category": by_category, "missing_count": missing}

    def _save_outputs(self, name: str, rows: List[Dict[str, Any]], aggregation: Dict[str, Any]) -> None:
        json_path = self.output_path / f"{name}_eval_results.json"
        csv_path = self.output_path / f"{name}_eval_results.csv"
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "routing": {category: get_active_metrics(category) for category in ["audio-only", "video-only", "joint", "speech"]},
            "aggregation": aggregation,
            "modules": {
                "objective": self.objective_enabled,
                "llm": self.llm_enabled,
                "checklist": self.checklist_enabled,
                "realism": self.realism_enabled,
            },
            "results": _convert_types(rows),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._write_csv(csv_path, rows)
        self.logger.info("Saved JSON: %s", json_path)
        self.logger.info("Saved CSV : %s", csv_path)

    def _save_model_summary(self, model_name: str, rows: List[Dict[str, Any]], summary_name: str, aggregation: Dict[str, Any]) -> None:
        model_dir = self.output_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        json_path = model_dir / f"{summary_name}_{model_name}_summary.json"
        csv_path = model_dir / f"{summary_name}_{model_name}_summary.csv"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"model_name": model_name, "aggregation": aggregation, "results": _convert_types(rows)}, f, ensure_ascii=False, indent=2)
        self._write_csv(csv_path, rows)

    def _write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        fields = [
            "id",
            "sample_id",
            "model_name",
            "instruction_id",
            "case_name",
            "src_video_name",
            "edited_video_name",
            "category",
            "active_metrics",
            *OBJECTIVE_METRICS,
            "checklist_score",
            "checklist_raw_score",
            "checklist_success",
            "editing_intent_overall",
            "editing_intent_video",
            "editing_intent_audio",
            "instruction_following_overall",
            "instruction_following_video",
            "instruction_following_audio",
            "fidelity_preserving_overall",
            "fidelity_preserving_video",
            "fidelity_preserving_audio",
            "edit_response_overall",
            "edit_response_video",
            "edit_response_audio",
            "realism_overall",
            "realism_video",
            "realism_audio",
            "realism_success",
            "warnings",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                scores = row.get("objective_scores", {})
                checklist_metrics = (row.get("checklist") or {}).get("metrics") or {}
                writer.writerow(
                    {
                        "id": row.get("id"),
                        "sample_id": row.get("sample_id"),
                        "model_name": row.get("model_name"),
                        "instruction_id": row.get("instruction_id"),
                        "case_name": row.get("case_name"),
                        "src_video_name": row.get("src_video_name"),
                        "edited_video_name": row.get("edited_video_name"),
                        "category": row.get("category"),
                        "active_metrics": ";".join(row.get("active_metrics", [])),
                        **{metric: scores.get(metric) for metric in OBJECTIVE_METRICS},
                        "checklist_score": row.get("scores", {}).get("checklist_score"),
                        "checklist_raw_score": row.get("scores", {}).get("checklist_raw_score"),
                        "checklist_success": row.get("checklist", {}).get("success"),
                        "editing_intent_overall": (checklist_metrics.get("editing_intent") or {}).get("overall"),
                        "editing_intent_video": (checklist_metrics.get("editing_intent") or {}).get("video"),
                        "editing_intent_audio": (checklist_metrics.get("editing_intent") or {}).get("audio"),
                        "instruction_following_overall": (checklist_metrics.get("instruction_following") or {}).get("overall"),
                        "instruction_following_video": (checklist_metrics.get("instruction_following") or {}).get("video"),
                        "instruction_following_audio": (checklist_metrics.get("instruction_following") or {}).get("audio"),
                        "fidelity_preserving_overall": (checklist_metrics.get("fidelity_preserving") or {}).get("overall"),
                        "fidelity_preserving_video": (checklist_metrics.get("fidelity_preserving") or {}).get("video"),
                        "fidelity_preserving_audio": (checklist_metrics.get("fidelity_preserving") or {}).get("audio"),
                        "edit_response_overall": (checklist_metrics.get("edit_response") or {}).get("overall"),
                        "edit_response_video": (checklist_metrics.get("edit_response") or {}).get("video"),
                        "edit_response_audio": (checklist_metrics.get("edit_response") or {}).get("audio"),
                        "realism_overall": ((row.get("realism") or {}).get("scores") or {}).get("overall"),
                        "realism_video": ((row.get("realism") or {}).get("scores") or {}).get("video"),
                        "realism_audio": ((row.get("realism") or {}).get("scores") or {}).get("audio"),
                        "realism_success": row.get("realism", {}).get("success"),
                        "warnings": " | ".join(row.get("warnings", [])),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AVE-Compass objective and MLLM evaluation")
    parser.add_argument("--output_path", type=str, default="./output", help="Output directory")
    parser.add_argument("--input_root", type=str, required=True, help="Directory containing the four evaluation input folders")
    parser.add_argument("--model_name", type=str, required=True, help="Name used for this model in outputs")
    parser.add_argument("--name", type=str, default="ave_compass_eval", help="Output file prefix")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--config", type=str, default=str((PROJECT_DIR / "configs" / "path.yml").resolve()), help="Config yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is not None:
        device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    bench = AVECompassEvaluator(device=device, output_path=args.output_path, config_path=args.config)
    result = bench.evaluate_from_input_root(args.input_root, args.model_name, args.name)
    print("=" * 80)
    print(f"Total samples: {result['total_samples']}")
    print(f"Aggregation: {json.dumps(_convert_types(result['aggregation']), ensure_ascii=False)}")
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
