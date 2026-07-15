import importlib
import math
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from metrics.evaluate import AVECompassEvaluator
from modules.objective.evaluator import METRIC_MODULES
from modules.objective.utils import OBJECTIVE_METRICS, aggregate_objective_scores, get_active_metrics


def test_routing():
    assert get_active_metrics("video-only") == [
        "video_aesthetic",
        "subject_consistency",
        "motion_smoothness",
        "audio_similarity",
    ]
    assert get_active_metrics("audio-only") == ["av_sync", "audio_aesthetic", "videoclip_video_similarity"]
    assert get_active_metrics("joint") == [
        "av_sync",
        "video_aesthetic",
        "subject_consistency",
        "motion_smoothness",
        "audio_aesthetic",
    ]
    assert get_active_metrics("speech") == [
        "lip_sync",
        "av_sync",
        "video_aesthetic",
        "subject_consistency",
        "motion_smoothness",
        "audio_aesthetic",
        "speech_quality",
    ]


def test_metric_modules_follow_paper_classification():
    assert METRIC_MODULES == {
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
    for metric, module_path in METRIC_MODULES.items():
        module = importlib.import_module(module_path)
        assert hasattr(module, f"compute_{metric}")


def test_preservation_and_consistency_formulas_match_implementation():
    from modules.objective.preservation.video_similarity_impl.scripts.batch_video_similarity import similarity_score
    from modules.objective.video_quality.subject_consistency_impl.scripts.batch_subject_consistency import consistency_score

    assert similarity_score(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == 0.0

    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    result = consistency_score(features)
    assert result["score"] == 0.5
    assert result["adjacent_cosine"] == 0.0
    assert result["pairwise_cosine"] == 1.0 / 3.0


def test_aggregate_ignores_null_and_nan():
    rows = [
        {"model_name": "m", "category": "video-only", "active_metrics": ["audio_similarity"], "objective_scores": {"audio_similarity": 0.5}},
        {"model_name": "m", "category": "video-only", "active_metrics": ["audio_similarity"], "objective_scores": {"audio_similarity": None}},
        {"model_name": "m", "category": "video-only", "active_metrics": ["audio_similarity"], "objective_scores": {"audio_similarity": math.nan}},
    ]
    for row in rows:
        for metric in OBJECTIVE_METRICS:
            row["objective_scores"].setdefault(metric, None)
    agg = aggregate_objective_scores(rows)
    assert agg["overall"]["audio_similarity"] == 0.5
    assert agg["overall"]["audio_aesthetic"] is None


def test_four_folder_input_discovery(tmp_path):
    input_root = tmp_path / "input"
    source_dir = input_root / "videos"
    instruction_dir = input_root / "edit_instructions"
    checklist_dir = input_root / "checklists"
    edited_dir = input_root / "edited_videos"
    for folder in (source_dir, instruction_dir, checklist_dir, edited_dir):
        folder.mkdir(parents=True)

    source_name = "sample.mp4"
    variant = "sample.1"
    (source_dir / source_name).write_bytes(b"")
    (edited_dir / f"{variant}.mp4").write_bytes(b"")
    (instruction_dir / f"{variant}.json").write_text(
        json.dumps(
            {
                "video": source_name,
                "task": "sample",
                "instruction_index": 1,
                "total_instructions_for_video": 1,
                "instruction": {
                    "category_label": "video_only",
                    "category": "V1",
                    "operation": "V1.1",
                    "prompt_en": "Change the object color to blue.",
                },
            }
        ),
        encoding="utf-8",
    )
    checklist_path = checklist_dir / f"{variant}_checklist.json"
    checklist_path.write_text(
        json.dumps(
            {
                "video": source_name,
                "instruction_index": 1,
                "caption": "Visual caption: a red object.",
                "edit_prompt": "Change the object color to blue.",
                "edit_category": "video_only",
                "questions": [],
                "success": True,
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yml"
    config_path.write_text("modules:\n  objective: false\n  llm: false\n", encoding="utf-8")
    bench = AVECompassEvaluator(device="cpu", output_path=str(tmp_path / "out"), config_path=str(config_path))

    samples = bench._build_samples_from_input_root(input_root, "agent")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["src_video_name"] == source_name
    assert sample["edited_video_name"] == f"{variant}.mp4"
    assert sample["model_name"] == "agent"
    assert sample["category"] == "video-only"
    assert sample["checklist_path"] == str(checklist_path.resolve())
    assert sample["source_caption"] == "Visual caption: a red object."


def test_extract_checklist_score_uses_public_summary():
    bench = AVECompassEvaluator.__new__(AVECompassEvaluator)
    score = bench._extract_checklist_score(
        {"evaluation": {"checklist_summary": {"editing_intent": 73.5}}}
    )

    assert score == {
        "raw_score": 73.5,
        "score": 0.735,
        "score_source": "checklist_summary.editing_intent",
    }


def test_checklist_summary_uses_per_case_product():
    from modules.llm.evaluator import VideoEditEvaluator

    evaluator = VideoEditEvaluator.__new__(VideoEditEvaluator)
    summary = evaluator._summarize(
        [
            {"dimension": "Edit Response", "modality_tag": "video", "answer": "Yes"},
            {"dimension": "Instruction Following", "modality_tag": "video", "answer": "Yes"},
            {"dimension": "Instruction Following", "modality_tag": "video", "answer": "No"},
            {"dimension": "Fidelity", "modality_tag": "video", "answer": "Yes"},
        ]
    )["checklist_summary"]

    assert summary["dimension_scores"]["Instruction Following"]["score"] == 50.0
    assert summary["dimension_scores"]["Fidelity"]["score"] == 100.0
    assert summary["editing_intent"] == 50.0
    assert summary["editing_intent_by_modality"]["video"] == 50.0


def test_extract_checklist_metrics_exposes_paper_dimensions():
    bench = AVECompassEvaluator.__new__(AVECompassEvaluator)
    metrics = bench._extract_checklist_metrics(
        {
            "evaluation": {
                "checklist_summary": {
                    "editing_intent": 50.0,
                    "editing_intent_by_modality": {"video": 40.0, "audio": 60.0},
                    "dimension_scores": {
                        "Instruction Following": {"score": 80.0},
                        "Fidelity": {"score": 62.5},
                    },
                    "edit_response": {
                        "score": 75.0,
                        "video": {"score": 100.0},
                        "audio": {"score": 50.0},
                    },
                },
                "dimension_modality": {
                    "Instruction Following": {"video": {"score": 80.0}, "audio": {"score": 80.0}},
                    "Fidelity": {"video": {"score": 50.0}, "audio": {"score": 75.0}},
                },
            }
        }
    )

    assert metrics["editing_intent"] == {"overall": 50.0, "video": 40.0, "audio": 60.0}
    assert metrics["instruction_following"] == {"overall": 80.0, "video": 80.0, "audio": 80.0}
    assert metrics["fidelity_preserving"] == {"overall": 62.5, "video": 50.0, "audio": 75.0}
    assert metrics["edit_response"] == {"overall": 75.0, "video": 100.0, "audio": 50.0}


def test_checklist_aggregation_averages_per_case_intent():
    bench = AVECompassEvaluator.__new__(AVECompassEvaluator)
    rows = [
        {
            "model_name": "m",
            "category": "joint",
            "scores": {"checklist_score": 1.0},
            "checklist": {
                "metrics": {
                    "editing_intent": {"overall": 100.0, "video": 100.0, "audio": 100.0},
                    "instruction_following": {"overall": 100.0, "video": 100.0, "audio": 100.0},
                    "fidelity_preserving": {"overall": 100.0, "video": 100.0, "audio": 100.0},
                    "edit_response": {"overall": 100.0, "video": 100.0, "audio": 100.0},
                }
            },
        },
        {
            "model_name": "m",
            "category": "joint",
            "scores": {"checklist_score": 0.0},
            "checklist": {
                "metrics": {
                    "editing_intent": {"overall": 0.0, "video": 0.0, "audio": 0.0},
                    "instruction_following": {"overall": 0.0, "video": 0.0, "audio": 0.0},
                    "fidelity_preserving": {"overall": 0.0, "video": 0.0, "audio": 0.0},
                    "edit_response": {"overall": 0.0, "video": 0.0, "audio": 0.0},
                }
            },
        },
    ]

    aggregated = bench._aggregate_checklist_scores(rows)

    assert aggregated["overall"] == 0.5
    assert aggregated["metrics"]["editing_intent"]["overall"] == 50.0


def test_realism_normalization_and_na_handling():
    from modules.llm.realism_evaluator import _mean, _normalize, _parse_score

    assert _parse_score("NA") is None
    assert _parse_score(6) == 5
    assert _mean([5, 4, 3]) == 4.0
    assert _mean([None, None]) is None
    assert _normalize(4.0) == 75.0


def test_realism_aggregation_ignores_missing_audio():
    bench = AVECompassEvaluator.__new__(AVECompassEvaluator)
    rows = [
        {
            "model_name": "m",
            "category": "video-only",
            "realism": {"scores": {"overall": 75.0, "video": 75.0, "audio": None}},
        },
        {
            "model_name": "m",
            "category": "joint",
            "realism": {"scores": {"overall": 50.0, "video": 25.0, "audio": 75.0}},
        },
    ]

    aggregated = bench._aggregate_realism_scores(rows)

    assert aggregated["overall"] == {"overall": 62.5, "video": 50.0, "audio": 75.0}
    assert aggregated["missing_count"]["audio"] == 1
