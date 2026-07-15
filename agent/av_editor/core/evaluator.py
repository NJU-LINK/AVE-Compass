"""
evaluator.py - Quality & consistency evaluation between editing steps.

After each atomic edit, the Evaluator checks:
  1. **Quality**      : Did the edit achieve the intended goal?
  2. **Consistency**  : Were non-target regions left unchanged (over-edit guard)?

Both checks are performed by a VLM via API. The full BEFORE and AFTER
clips are sent as short videos so motion-dependent criteria
(suppress an action, change a behaviour, lipsync) are actually
verifiable — sampling JPEG frames misses sub-second events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from av_editor.config import LLMConfig
from av_editor.core._api_log import log_prompt
from av_editor.schema import EvalResult, EvalVerdict, SubTask

logger = logging.getLogger(__name__)

# Video evaluation goes through Gemini because Gemini natively accepts
# video clips for this evaluator path.
VIDEO_EVAL_MODEL = "gemini-2.5-flash"

# ── Prompt templates ───────────────────────────────────────────────────────

EVAL_SYSTEM_PROMPT = """\
You are a video editing quality evaluator.
You will receive TWO short video clips, in order:
  1. The BEFORE clip — the source going into this edit.
  2. The AFTER clip — what the editor produced.

WATCH BOTH CLIPS as continuous video — not as static stills. Motion,
timing, and sub-second events (a quick mouth open, a head tilt, an
impact frame) are part of the evaluation.

Evaluate TWO dimensions:
1. **quality_score** (0.0 – 1.0): How well does the edit match the instruction?
   1.0 = perfect, 0.0 = completely wrong.
2. **consistency_score** (0.0 – 1.0): How well are the NON-TARGET areas preserved?
   1.0 = identical to before, 0.0 = everything changed.

Return ONLY a JSON object:
{
  "quality_score": <float>,
  "consistency_score": <float>,
  "reason": "<ONE concise sentence, ≤ 30 words>"
}
The `reason` MUST stay under 30 words so the JSON does not get
truncated by the response budget — keep it brief and factual.
"""

EVAL_CHECKLIST_SYSTEM_PROMPT = """\
You are a video editing quality evaluator.
You will receive:
  - A numbered list of specific evaluation criteria for this edit.
  - TWO short video clips, in order: the BEFORE clip, then the AFTER clip.

WATCH BOTH CLIPS as continuous video — not as static stills. Motion,
timing, and sub-second events (a quick mouth open, a head tilt, an
impact frame) are part of the evaluation.

## Mandatory grounding step (do this FIRST, internally)
Before scoring, identify the dominant visible subjects/objects in
each clip — the ones a viewer would name if asked "what is this
video showing?". Limit to 5 per clip; pick the most prominent.
Output them in `before_subjects` and `after_subjects`.

This is REQUIRED because criteria like "replace X with Y" can only
be verified by checking whether Y is actually in `after_subjects`
and X is actually absent from it. Do not skip this step.

## Temporal consistency check (CRITICAL — do not skip)

V2V editors frequently apply the requested change ONLY to the early
frames and let the original content reappear later. To catch this,
sample THREE moments in the AFTER clip — START (~10% in), MIDDLE
(~50%), END (~90%) — and describe what dominates each. Output them
in `after_temporal_samples` as three short strings.

When the edit is supposed to be SUSTAINED across the clip
(replace_object, recolor, scene_edit, style_transfer, etc.), the
target should appear in ALL THREE samples. If the original subject
re-emerges in the MIDDLE or END, the edit FAILED partial-temporally:
score the criterion proportionally (e.g. ~0.3 if only START matches,
~0.5 if START+MIDDLE match, ~1.0 only when all three match) and
note "edit reverts after Xs" in `reason`.

Examples of partial-temporal failure to catch:
- "replace squash with watermelon" — START shows watermelon, but at
  ~3s the elephant grabs it and an ORANGE squash reappears.
- "remove dog" — first frame is empty grass, but the dog appears
  again at ~5s when it walks back into frame.
- "make it cyberpunk style" — first 2s look stylised, then it
  reverts to the natural colours of the source.

## Scoring rules

Score EACH criterion from 0.0 to 1.0 based on what is ACTUALLY in
the AFTER clip (compared to BEFORE), not what you assume should be
there from the criterion text:
- For "X should be replaced with Y" / "X should be Y": score 1.0
  ONLY if Y appears in `after_subjects` AND X does not. If you wrote
  X in `after_subjects`, the replacement did NOT happen → score 0.0.
- For preservation criteria ("Z should remain unchanged"): score
  based on whether Z still looks the same.
- For consistency_score: how well non-target areas are preserved.

**Negation criteria are BINARY, not averaged.** Whenever a criterion
asks for the ABSENCE of an action ("should NOT move", "remain still",
"mouth should remain closed", "no longer visible", "is removed",
etc.), score 0.0 if the forbidden event occurs AT ANY POINT in the
AFTER clip — even a single frame counts. Do not score it 1.0 just
because the action is absent in most frames; one violation is
enough to fail. Score 1.0 only when the forbidden event is fully
absent throughout the AFTER clip.

Return ONLY a JSON object:
{
  "before_subjects": ["<subject1>", "<subject2>", ...],
  "after_subjects": ["<subject1>", "<subject2>", ...],
  "after_temporal_samples": ["<start>", "<middle>", "<end>"],
  "checklist_scores": {"<criterion_text>": <float>, ...},
  "consistency_score": <float>,
  "reason": "<ONE concise sentence, ≤ 40 words; cite timestamp of any negation/partial-temporal violation if present>"
}
Keep `reason` SHORT — long answers get truncated by the response
budget and the JSON fails to parse. One factual sentence is enough.
"""


class Evaluator:
    """
    Evaluates the result of a single editing step by comparing
    before/after keyframes via a VLM.
    """

    def __init__(
        self,
        llm_cfg: LLMConfig,
        quality_threshold: float = 0.6,
        consistency_threshold: float = 0.7,
        session_dir: Path | None = None,
    ):
        # Keep llm_cfg for the official Gemini path. We no longer
        # construct an OpenAI client here — multimodal eval calls go
        # straight to Google via _gemini_client.
        self.llm_cfg = llm_cfg
        self.model = llm_cfg.gemini_model or VIDEO_EVAL_MODEL
        self.quality_threshold = quality_threshold
        self.consistency_threshold = consistency_threshold
        self.session_dir = session_dir
        self._eval_records: list[dict[str, Any]] = []

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _video_to_base64_url(video_path: Path) -> str:
        mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
        b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def _parse_response(self, text: str) -> dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    # ── public API ─────────────────────────────────────────────────────

    async def evaluate(
        self,
        subtask: SubTask,
        before_video: Path,
        after_video: Path,
        workspace: Path,
    ) -> EvalResult:
        """
        Compare before/after videos for one editing step.
        Uses checklist mode when subtask.eval_criteria is available,
        otherwise falls back to generic quality/consistency scoring.
        """
        # We now send the BEFORE/AFTER clips directly via the official
        # Gemini API (see _gemini_client). No need to base64-encode
        # them up-front — the helper picks inline-vs-File-API by size.

        # Choose prompt mode based on eval_criteria
        use_checklist = bool(subtask.eval_criteria)
        clips_note = (
            "You will receive TWO video clips in order: the FIRST is the "
            "BEFORE clip, the SECOND is the AFTER clip. Watch each end-"
            "to-end as continuous video — do not treat them as stills."
        )
        if use_checklist:
            system_prompt = EVAL_CHECKLIST_SYSTEM_PROMPT
            criteria_text = "\n".join(
                f"{i+1}. {c}" for i, c in enumerate(subtask.eval_criteria)
            )
            text_content = (
                f"Evaluation criteria:\n{criteria_text}\n\n{clips_note}"
            )
        else:
            system_prompt = EVAL_SYSTEM_PROMPT
            text_content = (
                f"Editing instruction for this step:\n"
                f'"{subtask.intent or subtask.video_prompt}"\n'
                f"Action: {subtask.action.value}\n"
                f"Target scope: {subtask.target.value}\n\n{clips_note}"
            )

        try:
            from av_editor.core._gemini_client import gemini_with_fallback
            api_key = self.llm_cfg.gemini_api_key
            text = await asyncio.to_thread(
                gemini_with_fallback,
                gemini_api_key=api_key,
                primary_model=self.model,
                fallback_model="gemini-2.5-flash",
                system_prompt=system_prompt,
                user_text=text_content,
                media_paths=[before_video, after_video],
                json_response=True,
                max_output_tokens=9999,
                component=(
                    f"Evaluator(video, step={subtask.step}, "
                    f"action={subtask.action.value}, "
                    f"use_checklist={use_checklist})"
                ),
            )
            data = self._parse_response(text or "{}")

            if use_checklist:
                checklist = data.get("checklist_scores", {})
                q_score = (
                    sum(checklist.values()) / len(checklist) if checklist else 0.0
                )
                c_score = float(data.get("consistency_score", 0))
                reason = data.get("reason", "")
                before_subjects = list(data.get("before_subjects", []) or [])
                after_subjects = list(data.get("after_subjects", []) or [])
                after_temporal = list(data.get("after_temporal_samples", []) or [])

                # Log per-criterion scores
                for criterion, score in checklist.items():
                    logger.info("  [Checklist] %.2f  %s", score, criterion)
                if after_subjects:
                    logger.info(
                        "  [Subjects] before=%s after=%s",
                        before_subjects, after_subjects,
                    )
                if after_temporal:
                    logger.info(
                        "  [Temporal] start=%r middle=%r end=%r",
                        *(after_temporal + ["", "", ""])[:3],
                    )
            else:
                checklist = {}
                q_score = float(data.get("quality_score", 0))
                c_score = float(data.get("consistency_score", 0))
                reason = data.get("reason", "")
                before_subjects = []
                after_subjects = []
                after_temporal = []

            # Hard constraint: combined-threshold gate + per-dimension
            # floors at 0.3 + per-criterion floor at 0.3.
            # Rationale: an averaged `quality_score` can mask the case
            # where ONE criterion (e.g. "camera angle changed to low-
            # angle") scored 0 while peer criteria scored 1. The
            # checklist average looks acceptable but the user's actual
            # request was not realised.
            DIM_FLOOR = 0.3
            CRIT_FLOOR = 0.3
            passed_dim_thresholds = (
                q_score >= self.quality_threshold
                and c_score >= self.consistency_threshold
            )
            passed_dim_floor = (
                q_score >= DIM_FLOOR and c_score >= DIM_FLOOR
            )
            failed_criteria = [
                (name, s) for name, s in (checklist or {}).items()
                if s < CRIT_FLOOR
            ]
            passed_per_criterion = not failed_criteria
            passed = (
                passed_dim_thresholds
                and passed_dim_floor
                and passed_per_criterion
            )
            if passed_dim_thresholds and not passed_per_criterion:
                # Override: a criterion came in below the floor — even
                # if averaging hid it, this is a hard fail.
                fc = "; ".join(
                    f"{name!r}={s:.2f}" for name, s in failed_criteria
                )
                reason = (
                    f"Dimension averages pass (quality={q_score:.2f}, "
                    f"consistency={c_score:.2f}) but the following "
                    f"criterion/criteria scored below floor "
                    f"{CRIT_FLOOR}: {fc}. → FAIL"
                )

            # ── #3 Second-opinion: independent listing on AFTER ────────
            # Run a focused, criterion-blind call on the AFTER clip and
            # compare with the main evaluator's claim. If it contradicts
            # a "should be replaced with X" / "should not contain X"
            # criterion, downgrade to FAIL — the main evaluator has a
            # known habit of hallucinating PASS by parroting the
            # criterion text instead of actually looking at pixels.
            second_opinion = await self._second_opinion_subjects(after_video)
            override_reason: str | None = None
            if passed and second_opinion is not None:
                # Cross-check against the union of independent subjects
                # AND the main eval's per-timestamp samples — catches
                # both "wrong overall content" and "right at start but
                # reverts later" failure modes.
                pool = list(second_opinion) + [
                    str(t).strip().lower() for t in (after_temporal or [])
                    if str(t).strip()
                ]
                override_reason = self._cross_check_criteria_vs_subjects(
                    subtask.eval_criteria, pool,
                )
                if override_reason:
                    logger.warning(
                        "[Evaluator] step %d: main eval said PASS but "
                        "independent second-opinion contradicts: %s — "
                        "downgrading to FAIL.",
                        subtask.step, override_reason,
                    )
                    passed = False
                    reason = (
                        f"Downgraded: {override_reason}. "
                        f"Main reason was: {reason}"
                    )

            result = EvalResult(
                verdict=EvalVerdict.PASS if passed else EvalVerdict.FAIL,
                quality_score=q_score,
                consistency_score=c_score,
                reason=reason,
                checklist_scores=checklist,
            )
            logger.info(
                "[Evaluator] step %d: quality=%.2f consistency=%.2f → %s | %s",
                subtask.step, q_score, c_score, result.verdict.value, reason,
            )

            # ── persist eval record ───────────────────────────────────
            self._eval_records.append({
                "step": subtask.step,
                "intent": subtask.intent,
                "video_prompt": subtask.video_prompt,
                "quality_score": q_score,
                "consistency_score": c_score,
                "verdict": result.verdict.value,
                "reason": reason,
                "checklist_scores": checklist if checklist else None,
                "before_subjects": before_subjects,
                "after_subjects": after_subjects,
                "after_temporal_samples": after_temporal,
                "second_opinion_subjects": second_opinion,
                "downgrade_reason": override_reason,
            })
            self._flush_eval_log()

            return result

        except Exception as exc:
            logger.error("[Evaluator] failed: %s — defaulting to FAIL", exc)
            result = EvalResult(
                verdict=EvalVerdict.FAIL,
                quality_score=0.0,
                consistency_score=0.0,
                reason=f"Evaluation error: {exc}",
            )
            self._eval_records.append({
                "step": subtask.step,
                "intent": subtask.intent,
                "video_prompt": subtask.video_prompt,
                "quality_score": 0.0,
                "consistency_score": 0.0,
                "verdict": "fail",
                "reason": f"Evaluation error: {exc}",
                "checklist_scores": None,
            })
            self._flush_eval_log()
            return result

    # ── second-opinion + cross-check helpers ───────────────────────────

    _SECOND_OPINION_SYSTEM_PROMPT = (
        "You are a video object inspector. List the dominant visible "
        "subjects/objects in the attached clip — the things a viewer "
        "would name if asked 'what is this video showing?'.\n\n"
        "RULES:\n"
        "- Use 1-2 word common nouns (e.g. 'watermelon', 'dog', "
        "  'silver tape', 'acrylic sheet', 'man', 'puffer vest').\n"
        "- Up to 5 entries, in order of prominence.\n"
        "- Look at PIXELS only. Do NOT guess from context or assume "
        "  what should be there.\n\n"
        "Return STRICT JSON:\n"
        '{"subjects": ["...", "...", ...]}'
    )

    async def _second_opinion_subjects(self, after_video: Path) -> list[str] | None:
        """Independent listing of dominant subjects in AFTER clip.
        Criterion-blind so it cannot be biased by the edit description.
        Returns None on infra failure."""
        try:
            from av_editor.core._gemini_client import gemini_with_fallback
            api_key = self.llm_cfg.gemini_api_key
            text = await asyncio.to_thread(
                gemini_with_fallback,
                gemini_api_key=api_key,
                primary_model=self.model,
                fallback_model="gemini-2.5-flash",
                system_prompt=self._SECOND_OPINION_SYSTEM_PROMPT,
                user_text="Inspect the attached clip and list its dominant subjects.",
                media_paths=[after_video],
                json_response=True,
                max_output_tokens=9999,
                component="Evaluator(second-opinion subjects)",
            )
            data = self._parse_response((text or "").strip())
            subjects = [
                str(s).strip().lower() for s in data.get("subjects", [])
                if str(s).strip()
            ]
            logger.info("[Evaluator] second-opinion subjects: %s", subjects)
            return subjects[:5] or None
        except Exception as exc:
            logger.warning(
                "[Evaluator] second-opinion failed: %s — skipping cross-check",
                exc,
            )
            return None

    @staticmethod
    def _cross_check_criteria_vs_subjects(
        criteria: list[str], after_subjects: list[str],
    ) -> str | None:
        """Conservative cross-check: parse each criterion for
        "should be replaced with X" / "should NOT be X" / "X should
        be absent" patterns, and flag contradictions when X appears
        verbatim in the independent AFTER subject list.

        Returns a reason string if a contradiction is found, else None.
        Heuristic — only triggers on clear, literal mismatches; falls
        through silently on ambiguous criteria (better to trust main
        eval than cause false FAILs)."""
        if not criteria or not after_subjects:
            return None

        subjects_lc = [s.lower() for s in after_subjects]

        def _contains(term: str) -> bool:
            t = term.strip().lower()
            return any(t in s or s in t for s in subjects_lc) if t else False

        for c in criteria:
            text = (c or "").lower().strip()
            if not text:
                continue

            # Pattern A: "X should be replaced with Y" / "X is replaced
            # with Y" / "replace X with Y"
            #   → if X still in subjects AND Y not in subjects → fail
            for pat in (
                r"(?:replace|change|swap)\s+(?:the\s+)?([\w\s]+?)\s+(?:with|to|for|into)\s+(?:a\s+|an\s+|the\s+)?([\w\s]+?)(?:\.|$|,| should| that)",
                r"([\w\s]+?)\s+should\s+be\s+replaced\s+(?:with|by)\s+(?:a\s+|an\s+|the\s+)?([\w\s]+?)(?:\.|$|,)",
                r"([\w\s]+?)\s+is\s+replaced\s+(?:with|by)\s+(?:a\s+|an\s+|the\s+)?([\w\s]+?)(?:\.|$|,)",
            ):
                m = re.search(pat, text)
                if m:
                    old, new = m.group(1).strip(), m.group(2).strip()
                    if old and new and _contains(old) and not _contains(new):
                        return (
                            f"criterion '{c.strip()}' says '{old}' "
                            f"should be replaced with '{new}', but "
                            f"AFTER still shows '{old}' and not '{new}' "
                            f"(independent subjects: {after_subjects})"
                        )
                    break

            # Pattern B: "X should not be visible / present / audible /
            # in the scene" / "X is removed" / "X should be absent"
            for pat in (
                r"([\w\s]+?)\s+should\s+(?:not\s+be|no\s+longer\s+be)\s+(?:visible|present|in)",
                r"([\w\s]+?)\s+(?:is|has\s+been)\s+removed",
                r"([\w\s]+?)\s+should\s+be\s+(?:absent|removed|gone)",
                r"remove\s+(?:the\s+)?([\w\s]+?)(?:\.|$|,| from)",
            ):
                m = re.search(pat, text)
                if m:
                    forbidden = m.group(1).strip()
                    if forbidden and _contains(forbidden):
                        return (
                            f"criterion '{c.strip()}' says '{forbidden}' "
                            f"should be absent, but AFTER subjects "
                            f"still include it: {after_subjects}"
                        )
                    break

        return None

    # ── persistence ────────────────────────────────────────────────────

    def _flush_eval_log(self) -> None:
        """Write all eval records to session_dir/eval_log.json."""
        if self.session_dir is None:
            return
        try:
            out_path = self.session_dir / "eval_log.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(self._eval_records, ensure_ascii=False, indent=2)
            )
            logger.debug("[Evaluator] eval log saved → %s", out_path)
        except Exception as exc:
            logger.warning("[Evaluator] failed to save eval log: %s", exc)
