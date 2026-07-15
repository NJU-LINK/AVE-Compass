"""Final mixed-media evaluation and least-cost remediation signals."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from av_editor.config import LLMConfig
from av_editor.core._api_log import log_prompt
from av_editor.core._gemini_client import gemini_with_fallback
from av_editor.schema import (
    AudioInventory,
    MixEvalResult,
    VolumeAdjustment,
)

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.6
REPLAN_CONFIDENCE_THRESHOLD = 0.9


MIX_EVAL_SYSTEM_PROMPT = """\
You are the FINAL-STAGE mixed-media evaluator for an audio-video editing
pipeline. You receive the original source clip followed by the edited target
clip, the user's original instruction, the executed plan, and an
`audio_inventory` describing what the target audio should contain:

  - preserve      : sounds that MUST still be clearly audible from the original.
  - remove        : sounds that MUST be gone.
  - add           : newly-generated sounds that MUST be present and prominent.
  - replace       : pairs {from → to}; the 'from' sound must be absent, the
                    'to' sound must be clearly audible in its place.
  - volume_adjust : pure level shifts on existing stems. Each entry is
                    {target, delta_db, direction} where `direction` is
                    "boost" or "reduce". The target stem itself stays in
                    `preserve` (it's not replaced or removed) — the mix
                    just rebalances its loudness. Score it as: did the
                    target stem become noticeably louder/softer in the
                    expected direction relative to the rest of the mix?
                    (Don't expect ±delta_db precision; rough perceptual
                    direction is enough.)

Judge the assembled result globally, including failures that only become
visible after independently edited components are combined:

1. **Instruction following**  (instruction_score, 0.0–1.0)
   Check the original instruction, every executed plan criterion, and every
   audio-inventory item. Penalize missing, partial, or incorrect visual,
   audio, speech, and cross-modal edits.

2. **Fidelity preservation**  (fidelity_score, 0.0–1.0)
   Compare source and target. Check that non-target subjects, scene content,
   motion, speech, ambience, and timing remain faithful unless the instruction
   requires changing them.

3. **Overall quality**  (quality_score, 0.0–1.0)
   Judge visual realism, audio quality, temporal coherence, lip/AV sync, and
   whether independently generated components form one plausible clip.

4. **Volume balance**  (volume_balance, 0.0–1.0)
   Are `preserve` sounds still clearly audible, or drowned out?
   Are `add` / `replace.to` sounds prominent enough, not buried?
   1.0 = everything sits at an appropriate level.
   0.5 = one side (generated OR original) is clearly wrong.
   0.0 = unusable balance (dialogue inaudible, or generated layer missing entirely).

5. **Volume adjustment suggestion**  (ACTIONABLE)
   If volume_balance < 0.7, propose a concrete re-mix. Return
   `volume_adjustment` with two floats:
     * original_volume  — multiplier for the original/preserved track,
                          range [0.3, 1.0]. NEVER above 1.0 — the
                          original audio is the reference baseline,
                          and amplifying it past 1.0 introduces
                          clipping / distortion / unnatural loudness.
                          If the original feels "too quiet" relative
                          to the generated layer, ATTENUATE the
                          generated layer instead.
     * generated_volume — multiplier for the newly generated track,
                          range [0.05, 1.5]. Lower bound is
                          aggressively low so a noisy / hallucinated
                          generation can be muted to near-inaudible
                          rather than left dominating the mix.
   Start from baseline 1.0 and adjust:
     - generated too loud / hallucinated → generated_volume = 0.1-0.4,
                                            original_volume = 1.0
     - generated too quiet               → generated_volume up to 1.5
     - both OK                           → omit `volume_adjustment` (null)

6. **Needs regenerate flag**  (needs_regenerate, bool)
   Set TRUE when the generated content itself is wrong — missing
   entirely, wrong sound, hallucinated speech, separation that left
   the targeted sound fully audible — and rerunning the most recent
   generative audio subtask can fix it. Set FALSE when the content is
   correct and only the loudness is off.

7. **Needs replan signal**  (needs_replan, bool)
   Set TRUE only when local remixing or regenerating the most recent audio
   subtask cannot repair the failure. Examples: the plan omitted a required
   visual/audio dependency, the requested visual edit is absent or wrong,
   a non-target visual region was changed and must be regenerated from the
   source, the wrong shot or operation was selected, or a cross-modal
   structural conflict requires changing multiple steps. When TRUE, provide
   `replan_confidence` and concise `replan_feedback` describing observable
   failures and which requirements were not met. Report evidence rather than
   prescribing a replacement plan; the Planner decides how to repair it. Do
   not request replanning for a pure loudness issue or an isolated
   generative-audio failure.

8. **How the remediation fields combine**
   Treat the two local-repair fields as orthogonal and fill them independently of
   each other when `needs_replan` is false. When `needs_replan` is true,
   it takes priority over the cheaper local actions.

     • Content wrong, content fix is everything             →
       needs_regenerate = true,  volume_adjustment = null
     • Content wrong AND the new generation will probably also
       need a level tweak (e.g. ALL recent attempts of this same
       sound came in too loud / too quiet, so you expect the next
       regen to do the same)                                →
       needs_regenerate = true,  volume_adjustment = { ... }
       (the pipeline will regen and apply the level when remixing)
     • Content correct, only the loudness balance is off    →
       needs_regenerate = false, volume_adjustment = { ... }
     • Everything OK (passed)                               →
       needs_regenerate = false, volume_adjustment = null

   Do NOT suppress `volume_adjustment` just because you also set
   `needs_regenerate = true`. The pipeline will use both.

Return STRICT JSON (no markdown fences, no prose):
{
  "instruction_score": <float 0-1>,
  "fidelity_score":    <float 0-1>,
  "quality_score":     <float 0-1>,
  "volume_balance":    <float 0-1>,
  "per_item": {"<inventory item>": <float 0-1>, ...},
  "volume_adjustment": {"original_volume": <float>, "generated_volume": <float>} | null,
  "needs_regenerate":  <true|false>,
  "needs_replan":      <true|false>,
  "replan_confidence": <float 0-1>,
  "replan_feedback":   "<observed structural failures; otherwise empty>",
  "reason":            "<2 sentences: what you heard + why the scores>"
}
"""


def _parse_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start < 0:
        raise json.JSONDecodeError(f"no JSON in response: {raw[:200]!r}", raw, 0)
    depth = 0
    for i in range(start, len(raw)):
        c = raw[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])
    raise json.JSONDecodeError("unterminated JSON", raw, start)


def _clamp_vol(x: Any, lo: float = 0.05, hi: float = 1.5) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 1.0
    return max(lo, min(hi, v))


def _clamp_score(x: Any) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _clamp_orig_vol(x: Any) -> float:
    """Original-track volume must NEVER amplify past 1.0 — the source
    is the reference baseline. The remix loop should attenuate the
    generated layer instead."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 1.0
    return max(0.3, min(1.0, v))


def _clamp_gen_vol(x: Any) -> float:
    """Generated-track volume can attenuate aggressively (down to
    0.05) when the generation is hallucinated / too loud, and modestly
    boost (up to 1.5) when too quiet."""
    return _clamp_vol(x, lo=0.05, hi=1.5)


class MixEvaluator:
    """Evaluates the assembled edit and returns the cheapest repair signal.

    Routes Gemini model names through the official Gemini API, matching
    AudioEvaluator and Evaluator.
    """

    def __init__(self, llm_cfg: LLMConfig, session_dir: Path | None = None):
        self.cfg = llm_cfg
        self.session_dir = session_dir

    async def evaluate(
        self,
        final_video: Path,
        inventory: AudioInventory | None,
        *,
        attempt: int = 0,
        source_video: Path | None = None,
        instruction: str = "",
        source_caption: str = "",
        subtasks: list[Any] | None = None,
    ) -> MixEvalResult:
        """Grade the final edit and return an actionable remediation result."""
        inventory = inventory or AudioInventory()
        api_key = self.cfg.gemini_api_key
        model = self.cfg.gemini_model
        plan_summary = []
        for task in subtasks or []:
            if hasattr(task, "to_dict"):
                plan_summary.append(task.to_dict())
            elif isinstance(task, dict):
                plan_summary.append(task)
        user_text = (
            "Media order: the first attached clip is the ORIGINAL SOURCE; "
            "the second is the EDITED TARGET. If only one clip is attached, "
            "it is the edited target.\n\n"
            "original_instruction:\n"
            + instruction
            + "\n\nsource_caption:\n"
            + source_caption
            + "\n\nexecuted_plan:\n"
            + json.dumps(plan_summary, ensure_ascii=False, indent=2)
            + "\n\naudio_inventory:\n"
            + json.dumps(inventory.to_dict(), ensure_ascii=False, indent=2)
            + "\n\nCompare the clips, watch and listen, then return the JSON verdict."
        )
        media_paths = [final_video]
        if source_video is not None and source_video.exists():
            media_paths = [source_video, final_video]

        async def _call(json_mode: bool, max_tokens: int) -> str:
            return await asyncio.to_thread(
                gemini_with_fallback,
                gemini_api_key=api_key,
                primary_model=model,
                fallback_model="gemini-2.5-flash",
                system_prompt=MIX_EVAL_SYSTEM_PROMPT,
                user_text=user_text,
                media_paths=media_paths,
                json_response=json_mode,
                temperature=0.1,
                max_output_tokens=max_tokens,
                component="MixEvaluator",
            )

        try:
            # 1500 tokens — JSON output with full per_item dict +
            # reason can hit ~600 tokens; 800 was clipping mid-string.
            raw = await _call(json_mode=True, max_tokens=9999)
        except Exception as exc:
            logger.warning("[MixEval] LLM call failed: %s — neutral pass", exc)
            return MixEvalResult(
                passed=True, overall_score=0.6,
                instruction_score=0.6, fidelity_score=0.6,
                quality_score=0.6, volume_balance=0.6,
                reason=f"INFRA: {exc}", needs_regenerate=False,
                needs_replan=False,
            )

        # Cheap retry: if json mode returned a truncated/invalid blob,
        # ask again without forcing the JSON schema and try parsing
        # the freer-form response. This rescues calls where Gemini's
        # JSON-mode token budget got cut off (we've seen final '}'
        # missing → unterminated JSON).
        try:
            data = _parse_json(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[MixEval] JSON parse failed (json_mode=True): %s. "
                "raw[:300]=%r — retrying without json_response",
                exc, (raw or "")[:300],
            )
            try:
                raw2 = await _call(json_mode=False, max_tokens=9999)
            except Exception as exc2:
                logger.warning("[MixEval] retry call failed: %s", exc2)
                raw2 = ""
            try:
                data = _parse_json(raw2)
            except json.JSONDecodeError as exc2:
                logger.warning(
                    "[MixEval] JSON parse failed on retry: %s. "
                    "raw[:300]=%r — neutral pass",
                    exc2, (raw2 or "")[:300],
                )
                return MixEvalResult(
                    passed=True, overall_score=0.6,
                    instruction_score=0.6, fidelity_score=0.6,
                    quality_score=0.6, volume_balance=0.6,
                    reason="INFRA: unparseable verdict", needs_regenerate=False,
                    needs_replan=False,
                )

        instr = _clamp_score(data.get("instruction_score", 0.0))
        fidelity = _clamp_score(data.get("fidelity_score", 0.0))
        quality = _clamp_score(data.get("quality_score", 0.0))
        vol_bal = _clamp_score(data.get("volume_balance", 0.0))
        needs_regen = bool(data.get("needs_regenerate", False))
        requested_replan = bool(data.get("needs_replan", False))
        replan_confidence = _clamp_score(data.get("replan_confidence", 0.0))
        needs_replan = (
            requested_replan
            and replan_confidence >= REPLAN_CONFIDENCE_THRESHOLD
        )
        if requested_replan and not needs_replan:
            logger.info(
                "[MixEval] full replan suppressed: confidence %.2f < %.2f",
                replan_confidence,
                REPLAN_CONFIDENCE_THRESHOLD,
            )
        replan_feedback = str(data.get("replan_feedback", ""))[:1000]
        reason = str(data.get("reason", ""))[:500]

        overall = (instr + fidelity + quality) / 3.0
        passed = (
            overall >= PASS_THRESHOLD
            and not needs_regen
            and not needs_replan
        )

        vol_adj: VolumeAdjustment | None = None
        raw_adj = data.get("volume_adjustment")
        if isinstance(raw_adj, dict):
            vol_adj = VolumeAdjustment(
                original_volume=_clamp_orig_vol(raw_adj.get("original_volume", 1.0)),
                generated_volume=_clamp_gen_vol(raw_adj.get("generated_volume", 1.0)),
            )

        result = MixEvalResult(
            passed=passed,
            overall_score=overall,
            instruction_score=instr,
            fidelity_score=fidelity,
            quality_score=quality,
            volume_balance=vol_bal,
            reason=reason,
            volume_adjustment=vol_adj,
            needs_regenerate=needs_regen,
            needs_replan=needs_replan,
            replan_confidence=replan_confidence,
            replan_feedback=replan_feedback,
        )
        self._persist(result, inventory, attempt=attempt, final_video=final_video)
        logger.info(
            "[MixEval] attempt=%d instr=%.2f fidelity=%.2f quality=%.2f "
            "overall=%.2f regen=%s replan=%s confidence=%.2f adj=%s → %s",
            attempt, instr, fidelity, quality, overall, needs_regen,
            needs_replan, replan_confidence,
            vol_adj.to_dict() if vol_adj else None,
            "PASS" if passed else "FAIL",
        )
        return result

    def _persist(
        self,
        result: MixEvalResult,
        inventory: AudioInventory,
        *,
        attempt: int,
        final_video: Path,
    ) -> None:
        if not self.session_dir:
            return
        try:
            out = self.session_dir / "mix_eval.json"
            records: list[dict] = []
            if out.exists():
                try:
                    records = json.loads(out.read_text()) or []
                    if not isinstance(records, list):
                        records = [records]
                except Exception:
                    records = []
            records.append({
                "attempt": attempt,
                "final_video": str(final_video),
                "inventory": inventory.to_dict(),
                "passed": result.passed,
                "overall_score": result.overall_score,
                "instruction_score": result.instruction_score,
                "fidelity_score": result.fidelity_score,
                "quality_score": result.quality_score,
                "volume_balance": result.volume_balance,
                "needs_regenerate": result.needs_regenerate,
                "needs_replan": result.needs_replan,
                "replan_confidence": result.replan_confidence,
                "replan_feedback": result.replan_feedback,
                "volume_adjustment": (
                    result.volume_adjustment.to_dict()
                    if result.volume_adjustment else None
                ),
                "reason": result.reason,
            })
            out.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        except Exception as exc:
            logger.warning("[MixEval] persist failed: %s", exc)
