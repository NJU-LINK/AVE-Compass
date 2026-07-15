"""
audio_evaluator.py - Audio quality evaluation after the audio editing stage.

Sends the FINAL video (with mixed audio) to Gemini 2.5 Flash and evaluates:
  1. instruction_score  — did the requested audio edits actually happen?
  2. sync_score         — does generated audio sync with video events?
  3. fidelity_score     — is the original audio (BGM/ambient) preserved?

Uses the same official Gemini API channel as video_captioner.py.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Any

from av_editor.config import LLMConfig
from av_editor.core._api_log import log_prompt
from av_editor.schema import AudioEvalResult, EditAction, SubTask

logger = logging.getLogger(__name__)

AUDIO_EVAL_MODEL = "gemini-2.5-flash"


def _audio_rms_db(audio_path: Path) -> float | None:
    """Return overall RMS (dBFS) of *audio_path* via ffmpeg astats,
    or None on failure. Used as an objective audibility floor
    before asking the VLM."""
    import subprocess as _sp
    try:
        r = _sp.run(
            [
                "ffmpeg", "-hide_banner", "-nostats", "-i", str(audio_path),
                "-af", "astats=measure_overall=RMS_level",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return None
    for line in r.stderr.splitlines():
        if "RMS level dB" in line and "Overall" not in line:
            try:
                return float(line.split("RMS level dB:")[-1].strip())
            except ValueError:
                continue
    return None
PASS_THRESHOLD = 0.55   # overall_score >= this → passed

# Weights for overall_score
W_INSTRUCTION = 0.45
W_SYNC        = 0.25
W_FIDELITY    = 0.30

AUDIO_EVAL_SYSTEM_PROMPT = """\
You are an expert audio evaluator for video editing.
You will receive:
  - A final edited video (watch AND listen carefully to the audio track).
  - A description of the audio edits that were requested.
  - A checklist of specific criteria to score.

IMPORTANT: Base your evaluation ONLY on what you actually HEAR.
Do NOT infer sounds from visuals — only report what is audible.

Score each checklist criterion from 0.0 to 1.0.
Then provide four overall dimension scores / flags:
  - instruction_score (0.0–1.0): Did all requested audio changes happen?
  - sync_score (0.0–1.0): Does the added/modified audio synchronize well with video events?
    (Set to 1.0 if the instruction only requested background/ambient sound
     with no specific per-event sync requirement.)
  - fidelity_score (0.0–1.0): Is the original audio (BGM, ambient, speech) that should
    be preserved still clearly audible?
    (Set to 1.0 if there was no original audio to preserve, or if the
     instruction was to fully replace/remove all audio.)
  - generated_audio_too_quiet (true/false): Is the newly added/generated audio too quiet
    or insufficiently prominent? Set true if EITHER:
    (a) the sound is barely audible — requires effort to hear, or
    (b) the sound is audible but clearly not prominent enough for its intended role
        (e.g. rain that sounds like a faint trickle when heavy rain was requested).
    Set false when: (a) the generated sound is absent entirely (wrong content issue),
    (b) volume is appropriate, or (c) the problem is content/sync rather than volume.
  - original_audio_too_quiet (true/false): Is the original audio that should be preserved
    (BGM, ambient, etc.) present but significantly too quiet or being overpowered?
    Set true ONLY when it IS audible but clearly too low in the mix.
  - generated_audio_contamination (true/false): Does the NEWLY GENERATED layer
    (the SFX/ambient/BGM that was added by the model — NOT the original audio
    that was preserved) contain sounds OTHER than what was requested?
    IMPORTANT: judge the generated layer alone. If the final mix contains
    human speech BECAUSE THE ORIGINAL AUDIO ALREADY HAD SPEECH, that is
    expected and NOT contamination. Only flag contamination when the speech
    is a NEW vocal that the model hallucinated on top — typically recognisable
    by overlapping a different timbre/cadence on the original speaker, or
    appearing in a moment when the original was silent. In particular check:
      * Voice/speech/humming/grunting/moaning bleeding into a request for a
        NON-VOCAL sound effect (cough, rain, wind, drum, engine, footsteps, etc.)
      * A different instrument or sound type than requested
      * Any clearly off-topic content mixed into the generated layer
    Set false when the generated audio cleanly matches the requested sound.
    Note: a cough itself CAN legitimately contain some voice-band energy from the
    vocal cords; only flag true when the voice content is recognisable as SPEECH
    or melodic HUMMING, not mere vocal timbre of the requested event.

Return ONLY valid JSON:
{
  "checklist_scores": {"<criterion text>": <float 0-1>, ...},
  "instruction_score": <float 0-1>,
  "sync_score": <float 0-1>,
  "fidelity_score": <float 0-1>,
  "generated_audio_too_quiet": <true|false>,
  "original_audio_too_quiet": <true|false>,
  "generated_audio_contamination": <true|false>,
  "reason": "<2-3 sentence summary of what you heard and your assessment>"
}
"""


def _build_checklist(
    audio_tasks: list[SubTask],
    original_audio_desc: str,
    has_original_audio: bool,
) -> list[str]:
    """
    Auto-generate per-criterion evaluation checklist from audio subtasks.
    """
    criteria: list[str] = []

    for task in audio_tasks:
        # New flat schema: pull the human-meaningful descriptions from
        # the dedicated fields. `mmaudio_prompt` is the new-sound
        # description; `deleted_sound` is the original-sound label.
        desc = (
            getattr(task, "mmaudio_prompt", "") or task.new_sound
            or task.intent
        ).strip()
        original_sound = (task.deleted_sound or "").strip()

        # ── instruction criteria ──────────────────────────────────────
        # Prefer planner-generated eval_criteria (anchored to original user intent).
        # Fall back to description-based criteria if planner left them empty.
        if task.eval_criteria:
            criteria.extend(task.eval_criteria)
        else:
            if task.action == EditAction.AUDIO_ADD_SFX:
                criteria.append(f"The sound effect '{desc}' is clearly audible in the output")
                criteria.append(f"The sound effect is prominent enough to be easily noticed without straining to hear it")

            elif task.action == EditAction.AUDIO_ADD_AMBIENT:
                criteria.append(f"The ambient sound '{desc}' is present and audible throughout the video")
                criteria.append(f"The ambient sound is prominent enough to be immediately noticeable, not merely a faint background trace")

            elif task.action == EditAction.AUDIO_REPLACE_BGM:
                criteria.append(f"The new sound '{desc}' is clearly present and prominent in the output")
                criteria.append(f"The new sound is at an appropriate volume — not too faint or buried in the mix")
                if original_sound:
                    criteria.append(f"The original '{original_sound}' is no longer dominant or has been replaced")

            elif task.action == EditAction.AUDIO_REPLACE_SFX:
                criteria.append(f"The replacement sound '{desc}' is clearly audible and prominent")
                criteria.append(f"The replacement sound is at an appropriate volume — not too faint or buried in the mix")
                if original_sound:
                    criteria.append(f"The original sound '{original_sound}' is absent or significantly reduced")

            elif task.action == EditAction.AUDIO_REMOVE:
                criteria.append(f"The sound '{desc}' has been removed or is no longer audible")

        # ── sync criteria (always added for actions that require sync) ─
        # Independent of eval_criteria — ensures sync is always checklist-guided.
        if task.action == EditAction.AUDIO_ADD_SFX:
            criteria.append(
                f"The sound effect synchronizes with the relevant visual events "
                f"(timing matches what is happening on screen)"
            )
        elif task.action == EditAction.AUDIO_REPLACE_SFX:
            criteria.append(
                f"The audio synchronizes naturally with the video content"
            )

    # ── fidelity criteria (always added when original audio should be preserved) ──
    has_replace = any(
        t.action in (EditAction.AUDIO_REPLACE_BGM, EditAction.AUDIO_REPLACE_SFX)
        for t in audio_tasks
    )
    has_remove_all = (
        len(audio_tasks) == 1
        and audio_tasks[0].action == EditAction.AUDIO_REMOVE
    )
    if has_original_audio and not has_remove_all and original_audio_desc:
        short_desc = original_audio_desc[:120]
        if has_replace:
            criteria.append(
                f"The portions of the original audio NOT targeted for replacement "
                f"are still clearly audible (e.g. non-replaced layers from: {short_desc})"
            )
        else:
            criteria.append(
                f"The original audio ({short_desc}) remains clear and prominent, "
                f"not overpowered by the newly added sound"
            )

    return criteria


def _video_to_base64_url(video_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _parse_json(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction from an LLM response.

    Handles: bare JSON, ```json fenced blocks, unfenced blocks preceded
    by prose ("Here is the evaluation: { ... }"), and blocks with
    trailing commentary after the closing ``}``.
    """
    raw = text.strip()
    # Strip single fenced block if present (most common case).
    fenced = re.sub(r"^```(?:json)?\s*", "", raw)
    fenced = re.sub(r"\s*```$", "", fenced)
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    # Fallback: find the FIRST balanced {...} block in the text.
    start = raw.find("{")
    if start < 0:
        raise json.JSONDecodeError(
            f"No JSON object found in response (len={len(raw)}): "
            f"{raw[:200]!r}",
            raw, 0,
        )
    depth = 0
    for i in range(start, len(raw)):
        c = raw[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])
    raise json.JSONDecodeError(
        f"Unterminated JSON object in response (len={len(raw)}): "
        f"{raw[:200]!r}",
        raw, start,
    )


class AudioEvaluator:
    """
    Evaluates audio editing quality by sending the final video to Gemini
    and scoring three dimensions: instruction, sync, and fidelity.
    """

    def __init__(self, llm_cfg: LLMConfig, session_dir: Path | None = None):
        self.llm_cfg = llm_cfg
        self.session_dir = session_dir
        self._attempt = 0

    # ── Unified persistence: one envelope, kind-specific details ──────
    #
    # All three audio-side checks (`evaluate`, `is_sound_still_present`,
    # `is_generated_contaminated`) write to the same `audio_eval.json`
    # via `_save_record`. The common envelope is identical across kinds
    # so downstream tooling can read it uniformly; per-kind specifics
    # go into `details`.

    def _save_record(
        self,
        *,
        kind: str,                        # "full_eval" | "remove_check" | "contam_check"
        passed: bool,
        reason: str,
        details: dict[str, Any],
        step: int | None = None,
        attempt: int | None = None,
        action: str | None = None,
        audio_path: Path | str | None = None,
        metrics: dict[str, Any] | None = None,
        llm_called: bool = True,
    ) -> None:
        """Append one evaluation record to <session>/audio_eval.json.

        Envelope (always present, every kind):
          kind, step, attempt, action, audio_path, passed, reason,
          llm_called, metrics

        Kind-specific:
          full_eval     → details = {scores, checklist_scores, flags, audio_tasks}
          remove_check  → details = {sound_description, still_audible, audibility}
          contam_check  → details = {expected_description, has_unwanted_speech, what_else}
        """
        if not self.session_dir:
            return
        try:
            out = self.session_dir / "audio_eval.json"
            records: list[dict] = []
            if out.exists():
                try:
                    records = json.loads(out.read_text())
                    if not isinstance(records, list):
                        records = [records]
                except Exception:
                    records = []
            records.append({
                "kind": kind,
                "step": step,
                "attempt": attempt,
                "action": action,
                "audio_path": str(audio_path) if audio_path else None,
                "passed": passed,
                "reason": reason,
                "llm_called": llm_called,
                "metrics": metrics or {},
                "details": details,
            })
            out.write_text(json.dumps(records, ensure_ascii=False, indent=2))
            logger.debug(
                "[AudioEval] saved %s record (step=%s attempt=%s) → %s",
                kind, step, attempt, out,
            )
        except Exception as exc:
            logger.warning("[AudioEval] failed to save record: %s", exc)

    # ── Focused contamination check on the RAW generated layer ────────
    #
    # Rationale: the main `evaluate()` looks at the final mixed video, so
    # the LLM cannot tell whether speech in the mix came from the
    # preserved original or from a hallucinated MMAudio output. By
    # sending it ONLY the generated layer (wrapped in a black silent
    # video so the same multimodal endpoint can ingest it) and asking a
    # narrow yes/no question, we get a much more reliable contamination
    # verdict.

    # ── Simplified intent-based scorers ──────────────────────────────
    # One LLM call per stage, scores 0–1 against the user's stated
    # intent for the step. The pipeline tracks each attempt's score
    # and falls back to the best attempt if retries don't yield a
    # PASS — instead of dropping all the way back to the unedited
    # original audio.

    # Structured per-stage scoring prompts. Each one decomposes the
    # quality verdict into the dimensions the user-spec actually cares
    # about, so retry improvers can act on concrete signals (what's
    # missing? what's extra?) instead of a single opaque number.

    _SEP_SCORE_PROMPT = (
        "You are an audio QA scorer evaluating a SAM Audio separation "
        "result. The video's visual track is intentionally blank — "
        "listen to the AUDIO ONLY (this is the RESIDUAL stem after the "
        "target sound was supposed to be removed).\n\n"
        "The intent text below names the TARGET sound (to be removed) "
        "and may optionally include an `[audio_inventory]` block with "
        "a `MUST preserve:` list — sounds that must remain audible.\n\n"
        "Score TWO dimensions:\n\n"
        "1. target_extraction (0.0–1.0): how completely is the TARGET "
        "   sound gone from this residual? Be TOLERANT — some leakage "
        "   is acceptable.\n"
        "     1.0 essentially gone, only mask leakage / faint trace\n"
        "     0.7 reduced to faint trace, single brief instance — "
        "         STILL ACCEPTABLE\n"
        "     0.4 still moderately audible at multiple points\n"
        "     0.0 unchanged or dominant\n\n"
        "2. residual_fidelity (0.0–1.0):\n"
        "   • If the intent provides a `MUST preserve:` list (or other "
        "     explicit preserve items), check those sounds are still "
        "     clearly audible:\n"
        "       1.0 all preserve items present and clear\n"
        "       0.7 preserve items present, slight attenuation\n"
        "       0.4 some preserve content lost\n"
        "       0.0 important preserve content stripped\n"
        "   • If NO preserve list is provided (no preserve constraint "
        "     stated), this dimension is N/A — set residual_fidelity = "
        "     1.0 (trivially satisfied).\n\n"
        "Combined score:\n"
        "  if residual_fidelity < 0.4: score = min(0.4, 0.5*target_extraction "
        "  + 0.5*residual_fidelity)  — losing preserve is the WORSE failure\n"
        "  else: score = 0.5*target_extraction + 0.5*residual_fidelity\n\n"
        "Return STRICT JSON. Every field REQUIRED:\n"
        "{\n"
        '  "what_you_hear": "<concrete, ≥5 words>",\n'
        '  "target_extraction": <float 0-1>,\n'
        '  "residual_fidelity": <float 0-1>,\n'
        '  "score": <float 0-1, combined per the rule above>,\n'
        '  "reason": "<one sentence>"\n'
        "}"
    )

    _GEN_SCORE_PROMPT = (
        "You are an audio QA scorer evaluating an MMAudio V2 generation "
        "result. The video's visual track is intentionally blank — "
        "listen to the AUDIO ONLY (this is the RAW generated layer "
        "before being mixed with any preserved track).\n\n"
        "The intent text names the REQUESTED sound (what MMAudio should "
        "produce) and a FORBIDDEN list (sounds MMAudio must NOT produce — "
        "typically items already in the original audio that we want to "
        "preserve, so MMAudio re-generating them on top would duplicate).\n\n"
        "Score THREE dimensions independently:\n\n"
        "1. content_present (0.0–1.0): is the REQUESTED sound actually "
        "   in this audio?\n"
        "     1.0 clearly present, prominent\n"
        "     0.7 audible but partial / missing one sub-element\n"
        "     0.4 only weakly suggested\n"
        "     0.0 absent or silence\n\n"
        "2. negative_violation (0.0–1.0): how well does the audio AVOID "
        "   re-generating sounds in the FORBIDDEN list (which are the "
        "   sounds we want to preserve from the original)? "
        "   This catches duplicate generation of preserved content.\n"
        "     1.0 none of the forbidden sounds appear\n"
        "     0.7 brief / faint forbidden bleed\n"
        "     0.4 forbidden sound clearly audible alongside requested\n"
        "     0.0 forbidden sound dominates the generated layer\n\n"
        "3. hallucination (0.0–1.0): are there COMPLETELY UNRELATED "
        "   sounds — content neither requested nor in the forbidden "
        "   list, just made-up by the model?\n"
        "     1.0 clean, on-topic only\n"
        "     0.7 minor uninvited bleed\n"
        "     0.4 noticeable unrelated content\n"
        "     0.0 heavily hallucinated, mostly unrelated\n\n"
        "DO NOT score audio-visual sync — sync is irrelevant for this "
        "stage and is handled (if at all) by the final MixEvaluator.\n\n"
        "Combined score (WEAKEST-LINK rule — the lowest of the three "
        "drives the result, so any single failing dimension forces a "
        "retry):\n"
        "  score = min(content_present, negative_violation, hallucination)\n"
        "Examples:\n"
        "  content=1.0 negative=0.7 hallucination=0.4 → score=0.4 (retry — "
        "    audio has the right sound but bleeds forbidden + invented "
        "    extras; rewriting the prompt or strengthening negatives "
        "    will improve the next attempt)\n"
        "  content=0.0 negative=1.0 hallucination=1.0 → score=0.0 (retry — "
        "    requested sound is missing entirely)\n"
        "  content=1.0 negative=0.9 hallucination=0.9 → score=0.9 (PASS — "
        "    all three dimensions are clean)\n\n"
        "ALSO surface short, concrete strings for the retry improver:\n"
        "  missing  — comma-separated REQUESTED elements you did NOT hear "
        "             (≤ 60 chars). Empty if all present.\n"
        "  unwanted — comma-separated FORBIDDEN or HALLUCINATED sounds "
        "             you DID hear (≤ 60 chars). Empty if clean.\n\n"
        "Return STRICT JSON. Every field REQUIRED:\n"
        "{\n"
        '  "what_you_hear": "<concrete, ≥5 words>",\n'
        '  "content_present": <float 0-1>,\n'
        '  "negative_violation": <float 0-1>,\n'
        '  "hallucination": <float 0-1>,\n'
        '  "score": <float 0-1>,\n'
        '  "missing": "<short list or empty>",\n'
        '  "unwanted": "<short list or empty>",\n'
        '  "reason": "<one sentence>"\n'
        "}"
    )

    # Fallback single-axis prompt for unknown evaluation kinds.
    _INTENT_SCORE_PROMPT = (
        "You are an audio QA scorer. The video's visual track is "
        "intentionally blank — listen to the AUDIO ONLY and judge how "
        "well it satisfies the editor's intent for this step.\n\n"
        "Return STRICT JSON. EVERY field is REQUIRED and non-empty:\n"
        "{\n"
        '  "what_you_hear": "<concrete description of what is in the '
        'clip, ≥ 5 words>",\n'
        '  "score": <float 0.0–1.0>,\n'
        '  "reason": "<one sentence explaining the score, grounded in '
        'what_you_hear>"\n'
        "}\n\n"
        "Scoring guide:\n"
        "  1.0 intent fully achieved\n"
        "  0.7 largely achieved, minor flaws\n"
        "  0.4 partial\n"
        "  0.0 not achieved\n"
        "Use the FULL range; do not default to 0 or 1."
    )

    async def evaluate_separation_intent(
        self,
        residual_audio: Path,
        intent: str,
        step: int | None = None,
        attempt: int | None = None,
        action: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Score how well a SAM-separated residual matches the editor's
        intent. Returns (score, info). Infra failures default to 0.5
        with infra_error set so the caller can downgrade verdicts."""
        return await self._intent_score(
            residual_audio, intent,
            kind="separation_check",
            step=step, attempt=attempt, action=action,
        )

    async def evaluate_generation_intent(
        self,
        generated_audio: Path,
        intent: str,
        step: int | None = None,
        attempt: int | None = None,
        action: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Score how well a raw MMAudio output matches the editor's
        intent. Returns (score, info)."""
        return await self._intent_score(
            generated_audio, intent,
            kind="generation_check",
            step=step, attempt=attempt, action=action,
        )

    async def _intent_score(
        self,
        audio: Path,
        intent: str,
        kind: str,
        step: int | None,
        attempt: int | None,
        action: str | None,
    ) -> tuple[float, dict[str, Any]]:
        rms = _audio_rms_db(audio)
        metrics = {"audio_rms_db": rms}

        # Objective floor: digital silence trivially scores 0.
        if rms is not None and rms < -45.0:
            info = {
                "score": 0.0,
                "what_you_hear": "(silence)",
                "reason": (
                    f"audio RMS {rms:.1f} dB < −45 dB (essentially silent)"
                ),
                "infra_error": None,
            }
            self._save_record(
                kind=kind,
                step=step, attempt=attempt, action=action,
                audio_path=audio, passed=False, reason=info["reason"],
                details={"intent": intent, **info},
                metrics=metrics, llm_called=False,
            )
            return 0.0, info

        wrapped = await self._wrap_audio_silent_video(audio)
        if wrapped is None:
            # Wrap failure → don't block; treat as neutral (0.5).
            info = {"score": 0.5, "what_you_hear": "", "reason": "wrap failure",
                    "infra_error": "wrap"}
            return 0.5, info
        # Pick the structured per-stage prompt based on kind.
        if kind == "separation_check":
            system_prompt = self._SEP_SCORE_PROMPT
        elif kind == "generation_check":
            system_prompt = self._GEN_SCORE_PROMPT
        else:
            system_prompt = self._INTENT_SCORE_PROMPT

        try:
            user_text = (
                f"EDITOR INTENT for this step:\n{intent}\n\n"
                "Listen to the attached audio and return the structured "
                "JSON verdict."
            )
            data = await self._ask_audio_only(
                wrapped, system_prompt, user_text,
            )
            infra = data.get("_infra_error")
            what = str(data.get("what_you_hear", "")).strip()
            llm_reason = str(data.get("reason", "")).strip()
            try:
                raw_score = float(data.get("score"))
            except (TypeError, ValueError):
                raw_score = None

            if infra or raw_score is None or not what:
                # Infra or empty → neutral 0.5 so we don't fail-loop.
                info = {
                    "score": 0.5,
                    "what_you_hear": what,
                    "reason": (
                        f"INFRA: {kind} unparseable "
                        f"({infra or 'missing fields'})"
                    ),
                    "infra_error": infra or "missing fields",
                }
                self._save_record(
                    kind=kind,
                    step=step, attempt=attempt, action=action,
                    audio_path=audio, passed=False, reason=info["reason"],
                    details={"intent": intent, **info},
                    metrics=metrics, llm_called=True,
                )
                return 0.5, info

            score = max(0.0, min(1.0, raw_score))
            # Pull stage-specific sub-scores into the info dict so
            # downstream improvers can act on concrete signals (what's
            # missing, what's unwanted, fidelity vs extraction split).
            sub: dict[str, Any] = {}
            if kind == "separation_check":
                for k in ("target_extraction", "residual_fidelity"):
                    if k in data:
                        try:
                            sub[k] = max(0.0, min(1.0, float(data[k])))
                        except (TypeError, ValueError):
                            pass
            elif kind == "generation_check":
                # Three sub-scores: content / negative-duplication /
                # hallucination. Sync is intentionally NOT scored here —
                # sync judgement belongs to the final mix evaluator
                # (when present at all) and was a noise source at the
                # raw-generation stage.
                for k in ("content_present", "negative_violation",
                         "hallucination"):
                    if k in data:
                        try:
                            sub[k] = max(0.0, min(1.0, float(data[k])))
                        except (TypeError, ValueError):
                            pass
                for k in ("missing", "unwanted"):
                    if k in data and isinstance(data[k], str):
                        sub[k] = data[k][:80]
            info = {
                "score": score,
                "what_you_hear": what,
                "reason": llm_reason,
                "infra_error": None,
                **sub,
            }
            # Compact log line that surfaces sub-scores for tuning.
            sub_str = " ".join(f"{k}={v}" for k, v in sub.items() if not isinstance(v, str))
            logger.info(
                "[AudioEval] %s: score=%.2f%s | hear=%r | %s",
                kind, score, (" " + sub_str) if sub_str else "",
                what[:80], llm_reason[:120],
            )
            self._save_record(
                kind=kind,
                step=step, attempt=attempt, action=action,
                audio_path=audio, passed=score >= 0.6, reason=llm_reason,
                details={"intent": intent, **info},
                metrics=metrics, llm_called=True,
            )
            return score, info
        finally:
            try:
                wrapped.unlink(missing_ok=True)
            except Exception:
                pass

    # ── Stage 1 helper: full SAM separation quality check ────────────

    _SEP_TARGET_STEM_PROMPT = (
        "You are an audio QA inspector verifying SAM Audio separation. "
        "The video's visual track is intentionally blank — listen to "
        "the AUDIO ONLY. This audio is the TARGET stem of a source-"
        "separation task. The separator was asked to capture a "
        "specific sound described in the user prompt.\n\n"
        "Score `target_extraction` (0.0–1.0) — how cleanly the "
        "requested sound was captured into THIS stem:\n"
        "  1.0  Requested sound is CLEARLY present and is the "
        "       PREDOMINANT content; only minor background bleed-"
        "       through (acceptable).\n"
        "  0.7  Requested sound dominates, but a non-trivial bleed of "
        "       another source is also audible. Acceptable for most "
        "       downstream uses, but flag in `contaminants`.\n"
        "  0.4  Requested sound IS present but a DIFFERENT, "
        "       equally-loud source is also captured — the separator "
        "       failed to discriminate. Examples: asked for "
        "       'male voice' but stem has BOTH male and female "
        "       speech; asked for 'background music' but stem has "
        "       music + speech. Quality cap — partial extraction at "
        "       this level is a serious quality issue.\n"
        "  0.0  Requested sound is NOT in this stem (silent stem, OR "
        "       completely wrong content like 'asked for tearing, "
        "       hear only music').\n\n"
        "Apply the user's task-specific eval criteria (provided "
        "below in the user message) when scoring — they may carry "
        "additional requirements like 'no female voice in the male "
        "target stem' or 'BGM must not contain dialogue'.\n\n"
        "Return STRICT JSON. EVERY field is REQUIRED and non-empty:\n"
        "{\n"
        '  "what_you_hear": "<concrete description of every sound '
        'audible in this stem, ≥ 5 words>",\n'
        '  "contaminants": "<comma-separated list of OTHER sounds '
        'present in this stem that the request did NOT ask for; '
        'leave empty string if none>",\n'
        '  "target_extraction": <float in [0.0, 1.0]>,\n'
        '  "reason": "<one-sentence justification>"\n'
        "}"
    )

    _SEP_RESIDUAL_STEM_PROMPT = (
        "You are an audio QA inspector verifying SAM Audio separation. "
        "The video's visual track is intentionally blank — listen to "
        "the AUDIO ONLY. This audio is the RESIDUAL stem of a source-"
        "separation task — everything that was NOT extracted into "
        "the target. The separator was asked to REMOVE a specific "
        "sound (described in the user prompt) and PRESERVE certain "
        "other sounds.\n\n"
        "Score `residual_fidelity` (0.0–1.0) — how well this residual "
        "serves its dual job (target removed AND preserved sounds "
        "intact):\n"
        "  1.0  Target sound is ABSENT or only a faint trace AND every "
        "       expected preserved sound is fully audible and intact "
        "       (no cuts, no clipped words, no missing layers).\n"
        "  0.7  Target faint; preserved sounds mostly intact with "
        "       only minor degradation (a slight cut at a word "
        "       boundary, mild quality loss, brief gap).\n"
        "  0.4  EITHER the target is still CLEARLY audible (separator "
        "       under-removed) OR a preserved sound is significantly "
        "       degraded (a chunk of speech missing, BGM cut to half "
        "       length, several words clipped). Quality cap — either "
        "       failure makes the residual unusable.\n"
        "  0.0  Target DOMINATES the residual (separator did nothing) "
        "       OR a preserved sound is entirely missing (e.g. all "
        "       speech gone). Worst case.\n\n"
        "Apply the user's task-specific eval criteria (provided "
        "below in the user message) when scoring — they typically "
        "include 'speech remains fully intelligible' or 'no byte-"
        "level mis-extraction of speech segments'.\n\n"
        "Return STRICT JSON. EVERY field is REQUIRED and non-empty:\n"
        "{\n"
        '  "what_you_hear": "<concrete description of every sound '
        'audible in this residual, ≥ 5 words>",\n'
        '  "target_audibility": "<absent | faint_trace | clearly_audible | dominant>",\n'
        '  "preserved_sounds_missing": "<comma-separated list of '
        'expected preserved sounds you did NOT hear, or empty if all '
        'present / nothing was expected>",\n'
        '  "residual_fidelity": <float in [0.0, 1.0]>,\n'
        '  "reason": "<one-sentence justification>"\n'
        "}\n\n"
        "Notes:\n"
        "- If no preserved sounds were specified, the preserved-side "
        "  contribution is trivially satisfied — score from target "
        "  removal alone.\n"
        "- Speech is the most listener-noticeable layer. Even a 0.5-"
        "  second clip in dialogue is a serious failure → score "
        "  ≤ 0.4."
    )

    async def evaluate_separation(
        self,
        target_audio: Path,
        residual_audio: Path,
        target_description: str,
        expected_preserved: str = "",
        extra_eval_criteria: list[str] | None = None,
        threshold: float = 0.6,
        floor: float = 0.4,
        step: int | None = None,
        attempt: int | None = None,
        action: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Unified 2-dim SAM separation evaluator. Listens to BOTH
        target and residual stems and returns graded scores.

        Dimensions (each 0.0-1.0):
          target_extraction  — how cleanly the requested sound was
                                captured in the TARGET stem (and only
                                that sound; other foreground sources
                                penalise this score).
          residual_fidelity  — how well the RESIDUAL serves its dual
                                job: target removed AND preserved
                                sounds intact. A graded score covers
                                partial speech loss / partial BGM loss
                                / target leakage.

        Pass rule:
          combined = 0.5*target_extraction + 0.5*residual_fidelity
          passed = (combined >= threshold) AND
                   (target_extraction >= floor) AND
                   (residual_fidelity >= floor)

        Dual-floor enforcement: a 1.0/0.3 split fails even though the
        average is 0.65 — both halves must be at least minimally OK.

        `extra_eval_criteria` from Phase B is forwarded to BOTH stem
        prompts so task-specific concerns (e.g. 'no female voice in
        the male target stem', 'every spoken word must remain
        intelligible') affect the LLM's score directly.

        Infra failures default to score 0.5 (neutral) so a broken
        evaluator doesn't single-handedly fail or pass a step.
        """
        target_rms = _audio_rms_db(target_audio)
        residual_rms = _audio_rms_db(residual_audio)
        metrics = {
            "target_rms_db": target_rms,
            "residual_rms_db": residual_rms,
        }

        target_info = await self._check_separation_target_stem(
            target_audio, target_description, target_rms,
            extra_eval_criteria=extra_eval_criteria,
        )
        residual_info = await self._check_separation_residual_stem(
            residual_audio, target_description, expected_preserved,
            residual_rms,
            extra_eval_criteria=extra_eval_criteria,
        )

        tex = float(target_info.get("target_extraction", 0.5))
        fid = float(residual_info.get("residual_fidelity", 0.5))
        combined = 0.5 * tex + 0.5 * fid
        passed = (
            combined >= threshold
            and tex >= floor
            and fid >= floor
        )

        info = {
            "score": combined,
            "target_extraction": tex,
            "residual_fidelity": fid,
            "target_contaminants": target_info.get("contaminants", ""),
            "target_audibility": residual_info.get("target_audibility", ""),
            "preserved_sounds_missing": residual_info.get(
                "preserved_sounds_missing", ""
            ),
            "preserved_required": bool((expected_preserved or "").strip()),
            "target_what_you_hear": target_info.get("what_you_hear", ""),
            "residual_what_you_hear": residual_info.get("what_you_hear", ""),
            "target_reason": target_info.get("reason", ""),
            "residual_reason": residual_info.get("reason", ""),
            "target_infra_error": target_info.get("infra_error"),
            "residual_infra_error": residual_info.get("infra_error"),
            "passed_target_floor": tex >= floor,
            "passed_residual_floor": fid >= floor,
            "passed_combined": combined >= threshold,
        }

        # Build a one-line summary reason
        why = []
        if tex < floor:
            why.append(
                f"target_extraction={tex:.2f} below floor {floor}"
            )
        if fid < floor:
            why.append(
                f"residual_fidelity={fid:.2f} below floor {floor}"
            )
        if combined < threshold and tex >= floor and fid >= floor:
            why.append(
                f"combined={combined:.2f} below threshold {threshold}"
            )
        summary = (
            "; ".join(why) if why
            else f"both dimensions pass (tex={tex:.2f}, fid={fid:.2f})"
        )
        info["reason"] = summary

        logger.info(
            "[AudioEval] sep-quality: passed=%s score=%.2f tex=%.2f "
            "fid=%.2f | %s | target_hear=%r | residual_hear=%r",
            passed, combined, tex, fid, summary,
            info["target_what_you_hear"][:80],
            info["residual_what_you_hear"][:80],
        )
        self._save_record(
            kind="separation_check",
            step=step, attempt=attempt, action=action,
            audio_path=residual_audio,
            passed=passed, reason=summary,
            details={
                "target_description": target_description,
                "expected_preserved": expected_preserved,
                "extra_eval_criteria": extra_eval_criteria or [],
                **info,
            },
            metrics=metrics,
            llm_called=True,
        )
        return passed, info

    # Compatibility alias for older speech_tts/swap branches.
    check_separation_quality = evaluate_separation

    async def _check_separation_target_stem(
        self,
        target_audio: Path,
        target_description: str,
        target_rms: float | None,
        extra_eval_criteria: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sub-check 1 of separation: ask LLM about the target stem.
        Returns graded `target_extraction` (0-1) plus contaminants
        info."""
        if target_rms is not None and target_rms < -45.0:
            return {
                "target_extraction": 0.0,
                "contaminants": "",
                "what_you_hear": "(silence)",
                "reason": (
                    f"target stem RMS {target_rms:.1f} dB < −45 dB "
                    f"(stem is essentially silent — separation captured nothing)"
                ),
                "infra_error": None,
            }
        wrapped = await self._wrap_audio_silent_video(target_audio)
        if wrapped is None:
            return {
                "target_extraction": 0.5,        # neutral on infra
                "contaminants": "",
                "what_you_hear": "",
                "reason": "wrap failure — defaulting neutral",
                "infra_error": "wrap",
            }
        criteria_block = ""
        if extra_eval_criteria:
            criteria_block = (
                "\nTask-specific eval criteria (apply when scoring):\n"
                + "\n".join(f"  - {c}" for c in extra_eval_criteria)
                + "\n"
            )
        try:
            data = await self._ask_audio_only(
                wrapped, self._SEP_TARGET_STEM_PROMPT,
                f"This is an audio-separation task.\n"
                f"The separator was asked to ISOLATE this sound: "
                f"\"{target_description}\".\n"
                f"{criteria_block}"
                f"Listen to the attached target stem and grade "
                f"target_extraction.",
            )
            infra = data.get("_infra_error")
            what = str(data.get("what_you_hear", "")).strip()
            if infra or "target_extraction" not in data or not what:
                return {
                    "target_extraction": 0.5,    # neutral on infra
                    "contaminants": "",
                    "what_you_hear": what,
                    "reason": f"INFRA: target sub-check unparseable ({infra})",
                    "infra_error": infra or "missing fields",
                }
            try:
                tex = max(0.0, min(1.0, float(data.get("target_extraction"))))
            except (TypeError, ValueError):
                tex = 0.5
            return {
                "target_extraction": tex,
                "contaminants": str(data.get("contaminants", "")).strip(),
                "what_you_hear": what,
                "reason": str(data.get("reason", "")).strip(),
                "infra_error": None,
            }
        finally:
            try:
                wrapped.unlink(missing_ok=True)
            except Exception:
                pass

    async def _check_separation_residual_stem(
        self,
        residual_audio: Path,
        target_description: str,
        expected_preserved: str,
        residual_rms: float | None,
        extra_eval_criteria: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sub-check 2 of separation: ask LLM about the residual stem.
        Returns graded `residual_fidelity` (0-1) covering BOTH target-
        removal cleanliness AND preserved-sound integrity."""
        if residual_rms is not None and residual_rms < -45.0:
            return {
                "residual_fidelity": 0.0 if expected_preserved else 1.0,
                "target_audibility": "absent",
                "preserved_sounds_missing": expected_preserved or "",
                "what_you_hear": "(silence)",
                "reason": (
                    f"residual RMS {residual_rms:.1f} dB < −45 dB "
                    f"(residual is essentially silent — "
                    f"{'over-separation, preserved sounds lost' if expected_preserved else 'no preserved sounds expected → trivially clean'})"
                ),
                "infra_error": None,
            }
        wrapped = await self._wrap_audio_silent_video(residual_audio)
        if wrapped is None:
            return {
                "residual_fidelity": 0.5,        # neutral on infra
                "target_audibility": "absent",
                "preserved_sounds_missing": "",
                "what_you_hear": "",
                "reason": "wrap failure — defaulting neutral",
                "infra_error": "wrap",
            }
        preserved = (expected_preserved or "").strip() or "(none specified)"
        criteria_block = ""
        if extra_eval_criteria:
            criteria_block = (
                "\nTask-specific eval criteria (apply when scoring):\n"
                + "\n".join(f"  - {c}" for c in extra_eval_criteria)
                + "\n"
            )
        try:
            data = await self._ask_audio_only(
                wrapped, self._SEP_RESIDUAL_STEM_PROMPT,
                f"This is an audio-separation task.\n"
                f"TARGET sound (should be GONE from residual): "
                f"\"{target_description}\"\n"
                f"OTHER sounds expected to STILL BE PRESENT in the "
                f"residual (BGM / speech / ambience / other notable "
                f"sounds): \"{preserved}\"\n"
                f"{criteria_block}"
                f"Listen to the attached residual and grade "
                f"residual_fidelity.",
            )
            infra = data.get("_infra_error")
            what = str(data.get("what_you_hear", "")).strip()
            if infra or "residual_fidelity" not in data or not what:
                return {
                    "residual_fidelity": 0.5,    # neutral on infra
                    "target_audibility": "absent",
                    "preserved_sounds_missing": "",
                    "what_you_hear": what,
                    "reason": (
                        f"INFRA: residual sub-check unparseable ({infra})"
                    ),
                    "infra_error": infra or "missing fields",
                }
            try:
                fid = max(0.0, min(1.0, float(data.get("residual_fidelity"))))
            except (TypeError, ValueError):
                fid = 0.5
            return {
                "residual_fidelity": fid,
                "target_audibility": str(
                    data.get("target_audibility", "absent")
                ).lower().strip(),
                "preserved_sounds_missing": str(
                    data.get("preserved_sounds_missing", "")
                ).strip(),
                "what_you_hear": what,
                "reason": str(data.get("reason", "")).strip(),
                "infra_error": None,
            }
        finally:
            try:
                wrapped.unlink(missing_ok=True)
            except Exception:
                pass

    async def is_separation_target_captured(
        self,
        target_audio: Path,
        expected_sound: str,
        step: int | None = None,
        attempt: int | None = None,
        action: str | None = None,
    ) -> bool:
        """Stage-1 check (audio_replace_* only): did SAM Audio's target
        stem actually capture the requested sound? If not, the residual
        we feed downstream still contains the target, and we should
        retry SAM with a different prompt."""
        target_rms = _audio_rms_db(target_audio)

        # Objective floor: a digital-silent target stem is unambiguous
        # — separation missed entirely.
        if target_rms is not None and target_rms < -45.0:
            reason = (
                f"target stem RMS {target_rms:.1f} dB < −45 dB "
                f"(stem is essentially silent — separation captured nothing)"
            )
            logger.info("[AudioEval] sep-target-check: %s", reason)
            self._save_record(
                kind="separation_check",
                step=step, attempt=attempt, action=action,
                audio_path=target_audio, passed=False, reason=reason,
                details={
                    "expected_sound": expected_sound,
                    "captured": False,
                    "what_you_hear": "(silence)",
                },
                metrics={"target_rms_db": target_rms},
                llm_called=False,
            )
            return False

        wrapped = await self._wrap_audio_silent_video(target_audio)
        if wrapped is None:
            return True       # don't block on infra failure
        try:
            user_text = (
                f"The separator was asked to ISOLATE this sound: "
                f"\"{expected_sound}\".\n"
                "Listen to the attached stem and judge whether the "
                "intended sound is dominant in it."
            )
            data = await self._ask_audio_only(
                wrapped, self._SEP_TARGET_SYSTEM_PROMPT, user_text,
            )
            captured = bool(data.get("captured", False))
            what = str(data.get("what_you_hear", ""))
            llm_reason = str(data.get("reason", ""))
            logger.info(
                "[AudioEval] sep-target-check: captured=%s | hear=%r | %s",
                captured, what, llm_reason[:120],
            )
            self._save_record(
                kind="separation_check",
                step=step, attempt=attempt, action=action,
                audio_path=target_audio,
                passed=captured, reason=llm_reason,
                details={
                    "expected_sound": expected_sound,
                    "captured": captured,
                    "what_you_hear": what,
                },
                metrics={"target_rms_db": target_rms},
                llm_called=True,
            )
            return captured
        finally:
            try:
                wrapped.unlink(missing_ok=True)
            except Exception:
                pass

    # ── Stage 2: combined generation acceptability check ─────────────
    # Replaces the older speech-only contam-check + dominant-content
    # match check. The model only needs to answer two things on the
    # raw generated layer:
    #   (1) is the requested sound actually produced?
    #   (2) is there any UNWANTED sound — especially anything the
    #       planner listed in `negative_prompt`?
    # Either failure → retry generation.

    _GEN_ACCEPT_SYSTEM_PROMPT = (
        "You are an audio QA inspector verifying that a text-conditioned "
        "sound generator produced acceptable output. The video's visual "
        "track is intentionally blank — listen to the AUDIO ONLY.\n\n"
        "Return STRICT JSON. EVERY field is REQUIRED and non-empty:\n"
        "{\n"
        '  "what_you_hear": "<concrete description of every sound you '
        'actually hear in the clip, ≥ 8 words; mention each distinct '
        'sound source>",\n'
        '  "target_present": <true|false>,\n'
        '  "forbidden_present": <true|false>,\n'
        '  "forbidden_heard": "<comma-separated list of forbidden '
        'sounds you actually heard, or empty string if none>",\n'
        '  "reason": "<one short sentence grounded in what_you_hear>"\n'
        "}\n\n"
        "Workflow (do in order):\n"
        "1. List EVERYTHING you hear in `what_you_hear` first.\n"
        "2. Set `target_present` true only if the REQUESTED sound is "
        "   recognisably present. A vague generic ambient noise that "
        "   doesn't sound like the request → false.\n"
        "3. Set `forbidden_present` true if you hear ANY of the items "
        "   in the FORBIDDEN list (case-insensitive, semantic match — "
        "   e.g. 'engine' matches 'car engine sound'). List them in "
        "   `forbidden_heard`.\n"
        "4. The output is acceptable only when target_present=true "
        "   AND forbidden_present=false.\n\n"
        "Notes:\n"
        "- Speech / vocals are usually in the FORBIDDEN list for SFX "
        "  requests; treat them like any other forbidden item.\n"
        "- Texture variation / minor bleed of the requested sound is "
        "  fine; only flag forbidden_present for clearly distinct "
        "  off-topic sources."
    )

    async def is_generation_acceptable(
        self,
        generated_audio: Path,
        expected_description: str,
        negative_prompt: str = "",
        step: int | None = None,
        attempt: int | None = None,
        action: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Stage-2 unified acceptability check on raw generated audio.

        Returns (passed, info) where:
          passed = target_present AND not forbidden_present
          info = {
            "target_present": bool,
            "forbidden_present": bool,
            "forbidden_heard": str,   # comma-separated names
            "what_you_hear": str,
            "reason": str,
            "infra_error": str | None,
          }

        On infra failure we return (True, info) so a broken evaluator
        does not block the pipeline; the caller sees `infra_error` in
        info and can log accordingly."""
        gen_rms = _audio_rms_db(generated_audio)

        # Objective floor: silent output trivially fails (no target).
        if gen_rms is not None and gen_rms < -45.0:
            reason = (
                f"generated stem RMS {gen_rms:.1f} dB < −45 dB "
                f"(generation is essentially silent)"
            )
            logger.info("[AudioEval] gen-accept: %s", reason)
            info = {
                "target_present": False,
                "forbidden_present": False,
                "forbidden_heard": "",
                "what_you_hear": "(silence)",
                "reason": reason,
                "infra_error": None,
            }
            self._save_record(
                kind="generation_check",
                step=step, attempt=attempt, action=action,
                audio_path=generated_audio, passed=False, reason=reason,
                details={
                    "expected_description": expected_description,
                    "negative_prompt": negative_prompt,
                    **info,
                },
                metrics={"generated_rms_db": gen_rms},
                llm_called=False,
            )
            return False, info

        wrapped = await self._wrap_audio_silent_video(generated_audio)
        if wrapped is None:
            info = {"target_present": True, "forbidden_present": False,
                    "forbidden_heard": "", "what_you_hear": "",
                    "reason": "wrap failure", "infra_error": "wrap"}
            return True, info     # don't block on infra failure
        try:
            forbidden = (negative_prompt or "").strip() or "(none specified)"
            user_text = (
                f"REQUESTED sound: \"{expected_description}\"\n"
                f"FORBIDDEN sounds (must NOT appear in the output): "
                f"\"{forbidden}\"\n\n"
                "Listen to the attached audio and decide:\n"
                "- target_present: did the generator produce the "
                "REQUESTED sound?\n"
                "- forbidden_present: are any of the FORBIDDEN sounds "
                "audibly present in the output?"
            )
            data = await self._ask_audio_only(
                wrapped, self._GEN_ACCEPT_SYSTEM_PROMPT, user_text,
            )

            infra_err = data.get("_infra_error")
            what = str(data.get("what_you_hear", "")).strip()
            llm_reason = str(data.get("reason", "")).strip()

            # Empty fields / parse failure → treat as infra-fail
            # (accept the generation rather than fail-loop on broken
            # plumbing).
            required = ("target_present", "forbidden_present", "what_you_hear")
            missing = [k for k in required if k not in data]
            if infra_err or missing or not what:
                logger.warning(
                    "[AudioEval] gen-accept infra-fail "
                    "(infra_err=%s missing=%s what=%r) — accepting.",
                    infra_err, missing, what,
                )
                info = {
                    "target_present": True,
                    "forbidden_present": False,
                    "forbidden_heard": "",
                    "what_you_hear": what,
                    "reason": (
                        f"INFRA: gen-accept unparseable, accepting. "
                        f"err={infra_err or f'missing {missing}'}"
                    ),
                    "infra_error": infra_err or f"missing keys {missing}",
                }
                self._save_record(
                    kind="generation_check",
                    step=step, attempt=attempt, action=action,
                    audio_path=generated_audio,
                    passed=True, reason=info["reason"],
                    details={
                        "expected_description": expected_description,
                        "negative_prompt": negative_prompt,
                        **info,
                    },
                    metrics={"generated_rms_db": gen_rms},
                    llm_called=True,
                )
                return True, info

            target_present = bool(data.get("target_present", False))
            forbidden_present = bool(data.get("forbidden_present", False))
            forbidden_heard = str(data.get("forbidden_heard", "")).strip()
            passed = target_present and not forbidden_present

            info = {
                "target_present": target_present,
                "forbidden_present": forbidden_present,
                "forbidden_heard": forbidden_heard,
                "what_you_hear": what,
                "reason": llm_reason,
                "infra_error": None,
            }
            logger.info(
                "[AudioEval] gen-accept: passed=%s target=%s forbidden=%s%s | hear=%r | %s",
                passed, target_present, forbidden_present,
                f" ({forbidden_heard})" if forbidden_heard else "",
                what, llm_reason[:120],
            )
            self._save_record(
                kind="generation_check",
                step=step, attempt=attempt, action=action,
                audio_path=generated_audio,
                passed=passed, reason=llm_reason,
                details={
                    "expected_description": expected_description,
                    "negative_prompt": negative_prompt,
                    **info,
                },
                metrics={"generated_rms_db": gen_rms},
                llm_called=True,
            )
            return passed, info
        finally:
            try:
                wrapped.unlink(missing_ok=True)
            except Exception:
                pass

    _CONTAM_SYSTEM_PROMPT = (
        "You are an audio QA inspector. You will be given a short video "
        "whose visual track is intentionally blank — listen to the AUDIO "
        "ONLY. The audio is the raw output of a sound-effect generator "
        "and should contain ONLY the requested sound.\n\n"
        "Return STRICT JSON:\n"
        "{\n"
        '  "has_unwanted_speech": <true|false>,\n'
        '  "what_else": "<short description of any extra/off-topic sound, '
        'or empty string>",\n'
        '  "reason": "<one-sentence justification>"\n'
        "}\n\n"
        "Set has_unwanted_speech=true if you hear ANY of: human speech, "
        "dialogue, words, singing, melodic humming, muttering. "
        "Vocal-timbre energy from a clearly non-speech event (e.g. a cough, "
        "grunt, scream, laugh that was explicitly requested) does NOT count.\n"
        "Set has_unwanted_speech=false if the audio is purely the "
        "requested non-vocal sound (rustling, rain, engine, footsteps, "
        "drum, wind, water, etc.) or close to it."
    )

    _REMOVE_CHECK_SYSTEM_PROMPT = (
        "You are an audio QA inspector verifying that a sound has been "
        "removed. The video's visual track is intentionally blank — "
        "listen to the AUDIO ONLY.\n\n"
        "Return STRICT JSON:\n"
        "{\n"
        '  "still_audible": <true|false>,\n'
        '  "audibility": "<absent | residual | clearly audible>",\n'
        '  "reason": "<one-sentence justification>"\n'
        "}\n\n"
        "Set still_audible=true if the requested sound is clearly audible "
        "anywhere in the clip — even a single distinct instance counts. "
        "Faint smears or isolated low-energy traces under broader ambience "
        '("residual") still count as removed and should set still_audible=false. '
        "Set still_audible=false only when the requested sound is gone or "
        "reduced to inaudible residue."
    )

    async def is_sound_still_present(
        self,
        audio: Path,
        sound_description: str,
        reference_audio: Path | None = None,
        step: int | None = None,
        attempt: int | None = None,
        action: str | None = None,
    ) -> bool:
        """Yes/no check: is *sound_description* still audible in *audio*?
        Used by audio_remove to verify SAM Audio actually removed the
        target. On any infra failure returns False (don't block the
        pipeline; the previous behavior was to skip eval entirely).

        Objective pre-check: if *audio*'s overall RMS is essentially
        digital silence (or ≥35 dB below *reference_audio*), no sound
        is audibly present and we skip the LLM. This guards against
        Gemini hallucinating the prompt sound from low-level mask
        leakage / click artefacts that aren't actually perceptible.
        """
        residual_rms = _audio_rms_db(audio)
        ref_rms = _audio_rms_db(reference_audio) if reference_audio else None
        metrics = {
            "residual_rms_db": residual_rms,
            "reference_rms_db": ref_rms,
        }

        # Objective silence floor — skip LLM if residual is essentially
        # digital silence, otherwise the VLM tends to hallucinate the
        # prompt sound from sub-perceptible mask leakage.
        if residual_rms is not None and residual_rms < -45.0:
            reason = (
                f"residual RMS {residual_rms:.1f} dB < −45 dB "
                f"(objective silence floor)"
            )
            logger.info("[AudioEval] remove-check: %s", reason)
            self._save_record(
                kind="remove_check",
                step=step, attempt=attempt, action=action,
                audio_path=audio, passed=True, reason=reason,
                details={
                    "sound_description": sound_description,
                    "still_audible": False,
                    "audibility": "silence-floor",
                },
                metrics=metrics, llm_called=False,
            )
            return False
        if (
            residual_rms is not None and ref_rms is not None
            and ref_rms - residual_rms >= 35.0
        ):
            drop = ref_rms - residual_rms
            reason = (
                f"residual {residual_rms:.1f} dB is {drop:.1f} dB below "
                f"original {ref_rms:.1f} dB (≥35 dB drop)"
            )
            logger.info("[AudioEval] remove-check: %s", reason)
            self._save_record(
                kind="remove_check",
                step=step, attempt=attempt, action=action,
                audio_path=audio, passed=True, reason=reason,
                details={
                    "sound_description": sound_description,
                    "still_audible": False,
                    "audibility": "dropped-vs-original",
                },
                metrics=metrics, llm_called=False,
            )
            return False

        wrapped = await self._wrap_audio_silent_video(audio)
        if wrapped is None:
            return False
        try:
            user_text = (
                f"The audio was supposed to have this sound REMOVED: "
                f"\"{sound_description}\".\n"
                "Listen carefully and judge whether it is still audible."
            )
            data = await self._ask_audio_only(
                wrapped, self._REMOVE_CHECK_SYSTEM_PROMPT, user_text,
            )
            verdict = bool(data.get("still_audible", False))
            audibility = str(data.get("audibility", ""))
            llm_reason = str(data.get("reason", ""))
            logger.info(
                "[AudioEval] remove-check: still_audible=%s | audibility=%s | %s",
                verdict, audibility, llm_reason[:120],
            )
            self._save_record(
                kind="remove_check",
                step=step, attempt=attempt, action=action,
                audio_path=audio,
                passed=not verdict, reason=llm_reason,
                details={
                    "sound_description": sound_description,
                    "still_audible": verdict,
                    "audibility": audibility,
                },
                metrics=metrics, llm_called=True,
            )
            return verdict
        finally:
            try:
                wrapped.unlink(missing_ok=True)
            except Exception:
                pass

    # ── shared helpers for focused audio-only checks ──────────────────

    async def _wrap_audio_silent_video(self, audio: Path) -> Path | None:
        """Wrap a WAV/AAC into a tiny black silent-video MP4 so we can
        send it through the same multimodal endpoint that takes video.
        Returns the wrapped path or None on ffmpeg failure."""
        import subprocess as _sp
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False,
        ) as tmp:
            wrapped = Path(tmp.name)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=black:s=64x64:r=1",
            "-i", str(audio),
            "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "128k",
            str(wrapped),
        ]
        r = _sp.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            logger.warning(
                "[AudioEval] audio-wrap failed: %s", r.stderr[:200],
            )
            try:
                wrapped.unlink(missing_ok=True)
            except Exception:
                pass
            return None
        return wrapped

    async def _ask_audio_only(
        self, wrapped_video: Path, system_prompt: str, user_text: str,
    ) -> dict[str, Any]:
        """Single-purpose audio-only LLM call through the official Gemini API.
        Retries on parse error; infra failures surface via `_infra_error`."""
        from av_editor.core._gemini_client import gemini_with_fallback

        api_key = self.llm_cfg.gemini_api_key
        model = self.llm_cfg.gemini_model

        def _call() -> tuple[dict[str, Any] | None, str | None, str]:
            """3 tries against `gemini_with_fallback`, which routes both
            primary and fallback Gemini models through the official API."""
            last_err: str | None = None
            last_raw: str = ""
            for try_n in range(3):
                try:
                    raw = gemini_with_fallback(
                        gemini_api_key=api_key,
                        primary_model=model,
                        fallback_model="gemini-2.5-flash",
                        system_prompt=system_prompt,
                        user_text=user_text,
                        media_paths=[wrapped_video],
                        json_response=True,
                        max_output_tokens=9999,
                        component="AudioEvaluator(audio-only)",
                    )
                except Exception as net_exc:
                    last_err = f"network/api error try {try_n + 1}: {net_exc}"
                    logger.warning("[AudioEval] %s", last_err)
                    continue
                last_raw = raw
                try:
                    return _parse_json(raw), None, raw
                except Exception as parse_exc:
                    last_err = (
                        f"parse error on try {try_n + 1}: "
                        f"{parse_exc} | raw[:200]={raw[:200]!r}"
                    )
                    logger.warning("[AudioEval] %s", last_err)
            return None, last_err, last_raw

        try:
            data, err, raw = await asyncio.to_thread(_call)
        except Exception as exc:
            logger.warning("[AudioEval] audio-only ask failed: %s", exc)
            return {"_infra_error": f"thread: {exc}"}
        if data is None:
            return {"_infra_error": err or "unknown failure",
                    "_raw_response": raw}
        # Guard: occasionally the LLM returns a JSON array at the top
        # level (`[{...}]`) instead of a single object. All callers
        # treat the response as a dict (`data.get(...)`) — coerce to
        # the first dict element so downstream `.get` calls don't
        # crash with `'list' object has no attribute 'get'`.
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                data = data[0]
            else:
                return {
                    "_infra_error": "LLM returned non-dict array",
                    "_raw_response": str(data)[:200],
                }
        if not isinstance(data, dict):
            return {
                "_infra_error": f"LLM returned {type(data).__name__}",
                "_raw_response": str(data)[:200],
            }
        return data

    async def is_generated_contaminated(
        self,
        generated_audio: Path,
        expected_description: str,
        step: int | None = None,
        attempt: int | None = None,
        action: str | None = None,
    ) -> bool:
        """Run a focused yes/no check on whether the raw generated audio
        contains hallucinated human speech / vocals.

        Returns True if contaminated, False otherwise. On any infra
        failure returns False (don't block the pipeline on this check)."""
        gen_rms = _audio_rms_db(generated_audio)
        wrapped = await self._wrap_audio_silent_video(generated_audio)
        if wrapped is None:
            return False
        try:
            user_text = (
                f"The generator was instructed to produce ONLY: "
                f"\"{expected_description}\".\n"
                "Listen to the audio in the attached video and judge."
            )
            data = await self._ask_audio_only(
                wrapped, self._CONTAM_SYSTEM_PROMPT, user_text,
            )
            verdict = bool(data.get("has_unwanted_speech", False))
            what_else = str(data.get("what_else", ""))
            llm_reason = str(data.get("reason", ""))
            logger.info(
                "[AudioEval] contam-check on generated layer: "
                "has_unwanted_speech=%s | what_else=%r | %s",
                verdict, what_else, llm_reason[:120],
            )
            self._save_record(
                kind="contam_check",
                step=step, attempt=attempt, action=action,
                audio_path=generated_audio,
                passed=not verdict, reason=llm_reason,
                details={
                    "expected_description": expected_description,
                    "has_unwanted_speech": verdict,
                    "what_else": what_else,
                },
                metrics={"generated_rms_db": gen_rms},
                llm_called=True,
            )
            return verdict
        finally:
            try:
                wrapped.unlink(missing_ok=True)
            except Exception:
                pass

    async def evaluate(
        self,
        final_video: Path,
        audio_tasks: list[SubTask],
        original_audio_desc: str = "",
        has_original_audio: bool = True,
    ) -> AudioEvalResult:
        """
        Parameters
        ----------
        final_video         : Path to the postprocessed output (video + mixed audio).
        audio_tasks         : Audio subtasks that were executed.
        original_audio_desc : Short description of the original audio (from caption).
        has_original_audio  : Whether the original video had audio.
        """
        if not audio_tasks:
            logger.info("[AudioEval] no audio tasks — skipping evaluation")
            return AudioEvalResult(
                instruction_score=1.0, sync_score=1.0, fidelity_score=1.0,
                overall_score=1.0, passed=True, reason="No audio edits requested.",
            )

        checklist = _build_checklist(audio_tasks, original_audio_desc, has_original_audio)
        # New flat schema: prefer the natural-language `intent` (Phase
        # A) plus the modality-specific prompt as a tail.
        def _desc_for(t):
            tail = t.mmaudio_prompt or t.sam_prompt or ""
            return (t.intent or tail or "").strip()

        task_descriptions = "\n".join(
            f"- [{t.action.value}] {_desc_for(t)}" for t in audio_tasks
        )

        checklist_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(checklist))
        user_text = (
            f"## Audio edits that were applied\n{task_descriptions}\n\n"
            f"## Evaluation checklist\n{checklist_text}\n\n"
            "Watch and listen to the video carefully. "
            "Score each checklist item and the three overall dimensions."
        )

        from av_editor.core._gemini_client import gemini_with_fallback

        api_key = self.llm_cfg.gemini_api_key
        model = self.llm_cfg.gemini_model

        def _call() -> dict[str, Any]:
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    raw = gemini_with_fallback(
                        gemini_api_key=api_key,
                        primary_model=model,
                        fallback_model="gemini-2.5-flash",
                        system_prompt=AUDIO_EVAL_SYSTEM_PROMPT,
                        user_text=user_text,
                        media_paths=[final_video],
                        json_response=True,
                        max_output_tokens=9999,
                        component=f"AudioEvaluator(full-mix eval, {len(audio_tasks)} tasks)",
                    )
                    if not raw:
                        raise ValueError("Empty response from model")
                    try:
                        return _parse_json(raw)
                    except Exception as parse_exc:
                        logger.warning(
                            "[AudioEval] JSON parse failed: %s | "
                            "raw[:300]=%r",
                            parse_exc, raw[:300],
                        )
                        raise
                except Exception as e:
                    last_err = e
                    wait = min(2 ** attempt, 8)
                    logger.warning(
                        "[AudioEval] attempt %d/3 failed: %s — retry in %ds",
                        attempt + 1, e, wait,
                    )
                    time.sleep(wait)
            raise last_err or RuntimeError("Audio eval API failed")

        try:
            data = await asyncio.to_thread(_call)

            checklist_scores: dict[str, float] = {
                k: float(v) for k, v in data.get("checklist_scores", {}).items()
            }
            i_score   = float(data.get("instruction_score", 0.0))
            s_score   = float(data.get("sync_score", 1.0))
            f_score   = float(data.get("fidelity_score", 1.0))
            gen_quiet = bool(data.get("generated_audio_too_quiet", False))
            ori_quiet = bool(data.get("original_audio_too_quiet", False))
            contaminated = bool(data.get("generated_audio_contamination", False))
            reason    = data.get("reason", "")

            overall = W_INSTRUCTION * i_score + W_SYNC * s_score + W_FIDELITY * f_score
            # A good overall score can't rescue a catastrophic instruction
            # failure — if the requested edit wasn't done, fidelity/sync
            # being 1.0 is meaningless (they score the *untouched*
            # original audio). Require a minimum instruction score.
            # Contamination (unwanted voice/humming leaking into an SFX
            # generation) is also disqualifying — we'd rather retry with
            # a stronger negative prompt than ship dirty audio.
            speech_actions = {EditAction.SPEECH_TTS, EditAction.SPEECH_SWAP}
            has_speech_edit = any(t.action in speech_actions for t in audio_tasks)
            INSTRUCTION_FLOOR = 0.7 if has_speech_edit else 0.4
            passed = (
                overall >= PASS_THRESHOLD
                and i_score >= INSTRUCTION_FLOOR
                and not contaminated
            )

            # Log per-criterion scores
            for criterion, score in checklist_scores.items():
                logger.info("  [AudioChecklist] %.2f  %s", score, criterion)
            logger.info(
                "[AudioEval] instruction=%.2f sync=%.2f fidelity=%.2f overall=%.2f → %s"
                " | gen_quiet=%s ori_quiet=%s contaminated=%s | %s",
                i_score, s_score, f_score, overall,
                "PASS" if passed else "FAIL",
                gen_quiet, ori_quiet, contaminated, reason,
            )

            result = AudioEvalResult(
                instruction_score=i_score,
                sync_score=s_score,
                fidelity_score=f_score,
                overall_score=round(overall, 3),
                passed=passed,
                reason=reason,
                checklist_scores=checklist_scores,
                generated_audio_too_quiet=gen_quiet,
                original_audio_too_quiet=ori_quiet,
                generated_audio_contamination=contaminated,
            )
            self._save(result, audio_tasks, attempt=self._attempt)
            self._attempt += 1
            return result

        except Exception as exc:
            logger.error("[AudioEval] evaluation failed: %s — skipping", exc)
            return AudioEvalResult(
                instruction_score=0.0, sync_score=0.0, fidelity_score=0.0,
                overall_score=0.0, passed=False,
                reason=f"Evaluation error: {exc}",
            )

    def _save(self, result: AudioEvalResult, audio_tasks: list[SubTask], attempt: int = 0) -> None:
        # Step / action come from the audio task(s) being evaluated.
        step = audio_tasks[0].step if audio_tasks else None
        action = audio_tasks[0].action.value if audio_tasks else None
        self._save_record(
            kind="full_eval",
            step=step, attempt=attempt, action=action,
            audio_path=None,
            passed=result.passed,
            reason=result.reason,
            details={
                "audio_tasks": [t.to_dict() for t in audio_tasks],
                "checklist_scores": result.checklist_scores,
                "scores": {
                    "instruction": result.instruction_score,
                    "sync": result.sync_score,
                    "fidelity": result.fidelity_score,
                    "overall": result.overall_score,
                },
                "flags": {
                    "generated_audio_too_quiet": result.generated_audio_too_quiet,
                    "original_audio_too_quiet": result.original_audio_too_quiet,
                    "generated_audio_contamination": result.generated_audio_contamination,
                },
            },
            llm_called=True,
        )
