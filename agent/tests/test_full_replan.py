from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from av_editor.config import LLMConfig
from av_editor.core.mix_evaluator import MixEvaluator
from av_editor.core.planner import Planner
from av_editor.pipeline import EditingPipeline, OpResult, _StepContext
from av_editor.schema import AudioInventory, MixEvalResult


def test_audio_remove_uses_latest_edited_audio(tmp_path, monkeypatch):
    original_audio = tmp_path / "original.aac"
    current_audio = tmp_path / "after_first_remove.aac"
    original_video = tmp_path / "source.mp4"
    residual = tmp_path / "after_second_remove.aac"
    for path in (original_audio, current_audio, original_video):
        path.write_bytes(b"fixture")

    captured = {}
    pipeline = EditingPipeline.__new__(EditingPipeline)
    pipeline.cfg = SimpleNamespace(llm=LLMConfig(gemini_api_key="test"))

    async def fake_separate(video_path, prompt, output_dir, **kwargs):
        captured["video_path"] = video_path
        captured["audio_path"] = kwargs["audio_path"]
        residual.write_bytes(b"residual")
        return residual

    async def fake_retry(**kwargs):
        output = await kwargs["call"](kwargs["initial_prompt"])
        return OpResult(
            output_path=output,
            score=1.0,
            passed=True,
            passthrough=False,
            info={"reason": "pass"},
        )

    async def fake_post_eval(*args, **kwargs):
        return True, {"reason": "pass", "score": 1.0}

    pipeline._run_audio_separation = fake_separate
    pipeline._make_sam_evaluator = lambda *args, **kwargs: None
    pipeline._make_sam_improver = lambda *args, **kwargs: None
    pipeline._post_branch_eval = fake_post_eval
    monkeypatch.setattr("av_editor.pipeline._retry_op", fake_retry)

    context = _StepContext(
        session_dir=tmp_path,
        shots=[],
        duration=1.0,
        base_video=original_video,
        original_audio=original_audio,
        original_video=original_video,
        edited_audio=current_audio,
        audio_inventory=AudioInventory(preserve=["music"]),
    )
    subtask = SimpleNamespace(
        step=2,
        expect_prominent_target=False,
        deleted_sound="mechanical click",
        sam_prompt="mechanical click",
        sam_eval_criteria=[],
    )

    passed, _ = asyncio.run(pipeline._branch_audio_remove(
        subtask,
        context,
        attempt=0,
        step_dir=tmp_path / "step",
        video_for_audio=original_video,
        current_audio=current_audio,
    ))

    assert passed is True
    assert captured["audio_path"] == current_audio
    assert captured["video_path"] == original_video
    assert context.edited_audio == residual


def test_planner_dispatches_multimodal_messages_to_gemini(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return "[]"

    monkeypatch.setattr(
        "av_editor.core._gemini_client.generate_from_messages",
        fake_generate,
    )
    planner = Planner(LLMConfig(
        model="gemini-2.5-flash",
        gemini_api_key="test",
    ))
    user_content = [
        {"type": "text", "text": "instruction"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,AA=="},
        },
    ]
    result = asyncio.run(planner._call_llm("system", user_content))

    assert result == "[]"
    assert captured["model"] == "gemini-2.5-flash"
    assert captured["api_key"] == "test"
    assert captured["messages"][1]["content"] == user_content


def test_mix_evaluator_emits_global_replan_signal(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    target = tmp_path / "target.mp4"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    captured = {}

    def fake_gemini(**kwargs):
        captured.update(kwargs)
        return json.dumps({
            "instruction_score": 0.2,
            "fidelity_score": 0.7,
            "quality_score": 0.4,
            "volume_balance": 0.8,
            "volume_adjustment": None,
            "needs_regenerate": False,
            "needs_replan": True,
            "replan_confidence": 0.95,
            "replan_feedback": "The visual edit is absent while the audio changed.",
            "reason": "The requested visual change is absent.",
        })

    monkeypatch.setattr(
        "av_editor.core.mix_evaluator.gemini_with_fallback",
        fake_gemini,
    )
    evaluator = MixEvaluator(
        LLMConfig(gemini_api_key="test"),
        session_dir=tmp_path,
    )
    result = asyncio.run(evaluator.evaluate(
        target,
        AudioInventory(add=["rain"]),
        source_video=source,
        instruction="Make it rain and add rain sound.",
        source_caption="A dry street.",
        subtasks=[{"step": 1, "action": "weather_edit"}],
    ))

    assert result.needs_replan is True
    assert result.replan_confidence == 0.95
    assert result.needs_regenerate is False
    assert result.overall_score == (0.2 + 0.7 + 0.4) / 3.0
    assert "visual edit is absent" in result.replan_feedback
    assert captured["media_paths"] == [source, target]
    assert "original_instruction" in captured["user_text"]


def test_mix_evaluator_suppresses_low_confidence_replan(tmp_path, monkeypatch):
    target = tmp_path / "target.mp4"
    target.write_bytes(b"target")

    def fake_gemini(**kwargs):
        return json.dumps({
            "instruction_score": 0.3,
            "fidelity_score": 0.5,
            "quality_score": 0.4,
            "volume_balance": 0.8,
            "volume_adjustment": None,
            "needs_regenerate": False,
            "needs_replan": True,
            "replan_confidence": 0.89,
            "replan_feedback": "A possible structural issue was observed.",
            "reason": "The visual edit may be incomplete.",
        })

    monkeypatch.setattr(
        "av_editor.core.mix_evaluator.gemini_with_fallback",
        fake_gemini,
    )
    result = asyncio.run(MixEvaluator(
        LLMConfig(gemini_api_key="test"),
        session_dir=tmp_path,
    ).evaluate(target, AudioInventory()))

    assert result.replan_confidence == 0.89
    assert result.needs_replan is False


def test_planner_receives_full_replan_feedback(tmp_path):
    planner = Planner(LLMConfig(gemini_api_key="test"), session_dir=tmp_path)
    captured = {}

    async def fake_phase_a(instruction, *args):
        captured["phase_a_instruction"] = instruction
        return []

    async def fake_phase_b(intents, instruction, *args):
        captured["phase_b_instruction"] = instruction
        return []

    planner._phase_a = fake_phase_a
    planner._phase_b = fake_phase_b
    planner.validator = SimpleNamespace(
        validate_intents=lambda intents: [],
        validate_subtasks=lambda subtasks, **kwargs: [],
    )
    subtasks = asyncio.run(planner.plan(
        instruction="Replace the car with a bicycle.",
        replan_feedback="The car remained visible; regenerate the visual step.",
    ))

    assert subtasks == []
    assert "Replace the car with a bicycle." in captured["phase_a_instruction"]
    assert "FULL-REPLAN CONTEXT" in captured["phase_a_instruction"]
    assert "evidence, not as a prescribed solution" in captured["phase_a_instruction"]
    assert "car remained visible" in captured["phase_b_instruction"]
    saved = json.loads((tmp_path / "plan.json").read_text())
    assert saved["replan_feedback"].startswith("The car remained")


def test_mix_loop_bubbles_structural_failure_to_outer_pipeline(
    tmp_path, monkeypatch,
):
    output = tmp_path / "final.mp4"
    output.write_bytes(b"candidate")

    async def fake_evaluate(self, *args, **kwargs):
        return MixEvalResult(
            passed=False,
            overall_score=0.3,
            instruction_score=0.2,
            fidelity_score=0.4,
            quality_score=0.3,
            volume_balance=0.8,
            needs_replan=True,
            replan_confidence=0.95,
            replan_feedback="The plan omitted the required video operation.",
        )

    monkeypatch.setattr(MixEvaluator, "evaluate", fake_evaluate)
    pipeline = EditingPipeline.__new__(EditingPipeline)
    pipeline.cfg = SimpleNamespace(llm=LLMConfig(gemini_api_key="test"))
    context = _StepContext(
        session_dir=tmp_path,
        shots=[],
        duration=1.0,
        base_video=output,
        original_audio=None,
        instruction="Remove the object.",
        subtasks=[],
        allow_full_replan=True,
    )

    result_path = asyncio.run(pipeline._run_mix_eval_loop(
        context,
        output,
        output,
        AudioInventory(),
    ))

    assert result_path == output
    assert context.replan_request is not None
    assert context.mix_eval_result.overall_score == 0.3


def test_pipeline_replans_from_source_and_publishes_better_cycle(
    tmp_path, monkeypatch,
):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    video_only = tmp_path / "video_only.mp4"
    video_only.write_bytes(b"video")
    original_audio = tmp_path / "audio.aac"
    original_audio.write_bytes(b"audio")
    meta = SimpleNamespace(duration=2.0, width=1280, height=720, fps=24.0)

    monkeypatch.setattr(
        "av_editor.pipeline.preprocess",
        lambda *args, **kwargs: SimpleNamespace(
            video_path=video_only,
            audio_path=original_audio,
            original_video=source,
            has_audio=True,
            meta=meta,
        ),
    )
    monkeypatch.setattr(
        "av_editor.pipeline.extract_keyframes",
        lambda *args, **kwargs: [],
    )

    async def fake_caption(*args, **kwargs):
        return "A source clip with audio."

    monkeypatch.setattr("av_editor.pipeline.caption_video", fake_caption)
    monkeypatch.setattr("av_editor.pipeline.parse_shots", lambda *args, **kwargs: [])

    planner_calls = []

    class FakePlanner:
        session_dir = None
        last_inventory = AudioInventory()

        async def plan(self, **kwargs):
            planner_calls.append(kwargs)
            return [SimpleNamespace(is_audio=False)]

    pipeline = EditingPipeline.__new__(EditingPipeline)
    assert pipeline._MAX_FULL_REPLANS == 1
    pipeline.workspace = workspace
    pipeline.cfg = SimpleNamespace(llm=LLMConfig(gemini_api_key="test"))
    pipeline.planner = FakePlanner()
    pipeline.evaluator = SimpleNamespace(session_dir=None, _eval_records=[])

    async def fake_run_subtasks(subtasks, context):
        return None

    cycle_count = 0

    async def fake_assemble(context, output_path):
        nonlocal cycle_count
        cycle_count += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"cycle-{cycle_count}".encode())
        if cycle_count == 1:
            request = MixEvalResult(
                overall_score=0.3,
                needs_replan=True,
                replan_confidence=0.95,
                replan_feedback="The required dependent audio event is absent.",
            )
            context.mix_eval_result = request
            context.replan_request = request
        else:
            context.mix_eval_result = MixEvalResult(
                passed=True,
                overall_score=0.8,
            )
            context.replan_request = None
        return output_path

    pipeline._run_subtasks_ordered = fake_run_subtasks
    pipeline._assemble_final = fake_assemble

    final_output = asyncio.run(pipeline.run(source, "Make the scene rainy."))

    assert len(planner_calls) == 2
    assert planner_calls[0].get("replan_feedback") is None
    assert planner_calls[1]["replan_feedback"].startswith(
        "Final mixed-evaluation evidence:"
    )
    assert "dependent audio event is absent" in planner_calls[1]["replan_feedback"]
    assert final_output.read_bytes() == b"cycle-2"
    history = json.loads((final_output.parent / "full_replan.json").read_text())
    assert [item["needs_replan"] for item in history] == [True, False]
