"""
pipeline.py - Top-level orchestrator that wires all modules together.

Usage:
    from av_editor.pipeline import EditingPipeline
    from av_editor.config import AppConfig

    pipeline = EditingPipeline(AppConfig())
    result = await pipeline.run("input.mp4", "Make the clip cyberpunk")
    print(result)   # path to final video with audio

Flow:
    Preprocess -> Plan -> Execute video -> Execute audio -> Postprocess
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from av_editor.config import AppConfig
from av_editor.core.audio_evaluator import AudioEvaluator
from av_editor.core.evaluator import Evaluator
from av_editor.core.planner import (
    Planner,
    _intent_overlaps_any,
    build_audio_inventory,
)

# Minimum shot duration for video tools. Shots shorter
# than this are padded by freezing the last frame, edited, then trimmed.
V2V_MIN_DURATION_SEC = 3.0
from av_editor.core.postprocessor import postprocess
from av_editor.core.preprocessor import extract_keyframes, preprocess
from av_editor.core.state_tracker import StateTracker
from av_editor.core.video_captioner import caption_video, parse_shots
from av_editor.schema import EditAction, EditSession
from av_editor.tools.tool_registry import build_default_registry

logger = logging.getLogger(__name__)


# ── Helper: extract existing audio from video caption ─────────────────────

def _extract_existing_audio(video_caption: str) -> list[str]:
    """
    Parse the video caption (produced by Gemini) and extract a list of
    sounds/audio elements already present in the original video.

    These are used as ``negative_prompt`` for MMAudio so it does NOT
    regenerate sounds that will already be in the original audio track.

    Returns a list of short sound descriptions, e.g.
        ["dog barking", "background music", "wind noise"]
    """
    if not video_caption:
        return []

    import re

    sounds: list[str] = []

    # Keywords use regex word boundaries to avoid substring false matches
    # (e.g. "voice" in "invoice", "bell" in "embellish", "speech" in
    # "speechless"). Keys are case-insensitive regex patterns; values are
    # canonical negative-prompt phrases.
    #
    # NOTE: "silence" / "silent" are deliberately excluded — they are NOT
    # sounds, and feeding them to a generator's negative prompt would tell
    # it to avoid producing quiet passages.
    audio_keywords: dict[str, str] = {
        r"bark(?:ing|s|ed)?": "dog barking",
        r"meow(?:ing|s|ed)?": "cat meowing",
        r"speech|speaking|spoken|dialogue": "human speech",
        r"talk(?:ing|s|ed)?": "human talking",
        r"voices?": "human voice",
        r"music|bgm|soundtrack": "background music",
        r"wind": "wind",
        r"rain(?:ing|drops?)?": "rain",
        r"thunder": "thunder",
        r"traffic": "traffic noise",
        r"birds?|birdsong|chirp(?:ing|s|ed)?": "birdsong",
        r"water|splash(?:ing|es|ed)?|drip(?:ping|s|ped)?": "water sounds",
        r"footsteps?|footstep": "footsteps",
        r"engines?": "engine sound",
        r"sirens?": "siren",
        r"click(?:ing|s|ed)?": "clicking sounds",
        r"knock(?:ing|s|ed)?": "knocking",
        r"laugh(?:ing|s|ed|ter)?": "laughter",
        r"crying|sob(?:bing|s|bed)?": "crying",
        r"scream(?:ing|s|ed)?": "screaming",
        r"whistl(?:e|ing|es|ed)": "whistling",
        r"clap(?:ping|s|ped)?": "clapping",
        r"snor(?:e|ing|es|ed)": "snoring",
        r"cough(?:ing|s|ed)?": "coughing",
        r"chew(?:ing|s|ed)?": "chewing sounds",
        r"typing": "typing sounds",
        r"horns?": "horn honking",
        r"bells?": "bell ringing",
        r"drums?": "drums",
        r"guitars?": "guitar",
        r"pianos?": "piano",
        r"singing|sings?": "singing",
        r"humming": "humming",
        r"breathing|breaths?": "breathing",
        r"pant(?:ing|s|ed)?": "panting",
        r"growl(?:ing|s|ed)?": "growling",
        r"whin(?:e|ing|es|ed)": "whining",
        r"buzz(?:ing|es|ed)?": "buzzing",
        r"rustl(?:e|ing|es|ed)": "rustling",
        r"creak(?:ing|s|ed)?": "creaking",
        r"slam(?:ming|s|med)?": "door slamming",
    }

    # Prefer the "Audio analysis" section if present.
    audio_match = re.search(
        r"(?:audio\s*analysis|what\s*sounds)(.+?)(?:visual\s*analysis|##\s*visual|\Z)",
        video_caption,
        re.IGNORECASE | re.DOTALL,
    )
    search_text = (audio_match.group(1) if audio_match else video_caption).lower()

    # Drop explicit negations so we don't add "human speech" when the caption
    # says "no speech" / "silent" / "no music" etc. Simple but effective:
    # remove short windows around negation cues before scanning.
    search_text = re.sub(
        r"\b(?:no|not|without|absent|lack\s+of|zero|silent|silence)\s+[\w\s-]{0,40}",
        " ",
        search_text,
    )

    for pattern, description in audio_keywords.items():
        if re.search(rf"\b(?:{pattern})\b", search_text, re.IGNORECASE):
            if description not in sounds:
                sounds.append(description)

    return sounds


_SHOT_REGEX_PATTERNS = [
    # "shot 1", "Shot 2", "shot #3"
    (r"\b[Ss]hots?\s*#?\s*\d+\b", ""),
    # "the wide/medium/full/close-up shot", "this shot", "each shot"
    (r"\b(?:the|this|that|each|every|another|previous|current|next|first|second|third|wide|medium|full|close[-\s]?up|cutaway|establishing|master|reverse|over[-\s]?the[-\s]?shoulder)\s+[Ss]hots?\b", ""),
    # standalone "Shot"/"shot" if still slipping through
    (r"\b[Ss]hots?\b", ""),
]


def _sanitize_tool_prompt(text: str) -> str:
    """Remove any mention of shot-level terminology from a prompt
    string before it is sent to an editing model. The model only sees
    one clip at a time and has no access to other shots."""
    import re
    out = text or ""
    for pat, repl in _SHOT_REGEX_PATTERNS:
        out = re.sub(pat, repl, out)
    # Collapse any double spaces / dangling punctuation caused by the
    # removals.
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out.strip()


def _measure_rms_db(audio_path: Path) -> float | None:
    """Return the overall RMS level of *audio_path* in dBFS, or None on
    failure. Uses ffmpeg's ``astats`` filter."""
    import subprocess as _sp
    try:
        r = _sp.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(audio_path),
             "-af", "astats=measure_overall=RMS_level", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
        for line in r.stderr.splitlines():
            if "RMS level dB" in line and "Overall" not in line:
                try:
                    tail = line.split("RMS level dB:")[-1].strip()
                    return float(tail)
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def _compute_ambient_gen_vol(
    original_audio: Path,
    generated_audio: Path,
    headroom_db: float = 10.0,
    min_vol: float = 0.03,
    max_vol: float = 0.5,
) -> tuple[float, str]:
    """Pick a ``generated_volume`` so the ambient layer sits at least
    *headroom_db* BELOW the original audio's RMS. Ambient is a continuous
    broadband layer while dialogue is sparse; a fixed gen_vol like 0.3
    still ends up perceptually louder than sparse speech. Measuring and
    matching loudness keeps the ambient a clean underlay.

    Returns ``(gen_vol, explanation_str)`` for logging.
    """
    orig_db = _measure_rms_db(original_audio)
    gen_db = _measure_rms_db(generated_audio)
    if orig_db is None or gen_db is None:
        return 0.15, f"loudness probe failed (orig={orig_db}, gen={gen_db}); using fallback 0.15"
    target_ambient_db = orig_db - headroom_db
    gain_db = target_ambient_db - gen_db
    # Convert gain in dB to a linear multiplier.
    vol = 10 ** (gain_db / 20.0)
    vol = max(min_vol, min(max_vol, vol))
    return vol, (
        f"orig={orig_db:.1f}dB gen_raw={gen_db:.1f}dB "
        f"→ target≤{target_ambient_db:.1f}dB → gen_vol={vol:.3f}"
    )


def _caption_audio_section(video_caption: str) -> str:
    """Extract a short description of the original audio for the
    fidelity check in AudioEvaluator."""
    if not video_caption:
        return ""
    lines = video_caption.splitlines()
    audio_lines: list = []
    in_audio = False
    for line in lines:
        low = line.lower()
        if "audio" in low and ("##" in line or "**" in line):
            in_audio = True
        elif line.startswith("##") and in_audio:
            break
        if in_audio:
            audio_lines.append(line)
    return " ".join(audio_lines[:8]).strip()


def _norm_transcript_text(text: str | None) -> str:
    text = (text or "").lower()
    text = re.sub(r"[“”\"'`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find_transcript_window(
    caption: str | None,
    reference_text: str | None,
) -> tuple[float, float] | None:
    """Find the timestamped transcript span matching a spoken line."""
    ref = _norm_transcript_text(reference_text)
    if not ref or not caption:
        return None
    pattern = re.compile(
        r"\[(?P<start>\d+(?:\.\d+)?)\s*[–-]\s*"
        r"(?P<end>\d+(?:\.\d+)?)\][^\n\"“”]*[\"“](?P<text>.*?)[\"”]",
        re.DOTALL,
    )
    fuzzy: tuple[float, float] | None = None
    for m in pattern.finditer(caption):
        spoken = _norm_transcript_text(m.group("text"))
        if not spoken:
            continue
        start = float(m.group("start"))
        end = float(m.group("end"))
        if end <= start:
            continue
        if spoken == ref or spoken in ref or ref in spoken:
            return (start, end)
        a = set(ref.split())
        b = set(spoken.split())
        if (
            fuzzy is None and a and b
            and len(a & b) / max(1, min(len(a), len(b))) >= 0.6
        ):
            fuzzy = (start, end)
    return fuzzy


def _resolve_speech_splice_window(
    subtask,
    context: "_StepContext",
) -> tuple[float, float] | None:
    policy = getattr(subtask, "audio_splice", None) or {}
    mode = str(policy.get("mode", "localized_replace") or "localized_replace")
    if mode not in {"localized_replace", "local_replace"}:
        return None

    try:
        start = policy.get("start")
        end = policy.get("end")
        if start is not None and end is not None:
            s = float(start)
            e = float(end)
            if e > s:
                return (max(0.0, s), min(float(context.duration), e))
    except (TypeError, ValueError):
        pass

    def _pad_value(name: str, default: float) -> float:
        try:
            return max(0.0, float(policy.get(name, default)))
        except (TypeError, ValueError):
            return default

    ref = (
        policy.get("reference_text")
        or getattr(subtask, "speech_reference_text", "")
        or ""
    )
    win = _find_transcript_window(context.video_caption, ref)
    if win:
        pre_pad = _pad_value("pre_pad", 0.08)
        post_pad = _pad_value("post_pad", 0.45)
        max_snap_shot = _pad_value("max_snap_shot_duration", 2.0)
        start = max(0.0, win[0] - pre_pad)
        end = min(float(context.duration), win[1] + post_pad)
        ref_words = [w for w in re.split(r"\s+", ref.strip()) if w]
        # Captions sometimes align only the first few words of a long line.
        # A too-short speech window leaves most of the original speaker intact,
        # so widen long utterances to the containing shot when needed.
        if ref_words and len(ref_words) >= 5:
            min_reasonable = min(4.0, max(1.5, len(ref_words) * 0.18))
            if (end - start) < min_reasonable:
                for shot in getattr(context, "shots", []) or []:
                    shot_start = float(getattr(shot, "start", -1.0))
                    shot_end = float(getattr(shot, "end", -1.0))
                    if shot_start <= win[0] <= shot_end and shot_end > shot_start:
                        start = max(0.0, min(start, shot_start))
                        end = min(float(context.duration), max(end, shot_end))
                        break
                else:
                    end = float(context.duration)
        for shot in getattr(context, "shots", []) or []:
            if (
                getattr(shot, "duration", 999.0) <= max_snap_shot
                and float(getattr(shot, "start", -1.0)) <= win[0] <= float(getattr(shot, "end", -1.0))
                and win[0] - float(getattr(shot, "start", 0.0)) <= 0.25
            ):
                end = max(end, min(float(context.duration), float(shot.end)))
                break
        return (
            start,
            end,
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Unified step-runner scaffolding
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _StepContext:
    """Mutable state shared across SubTasks during ordered execution.

    Set once per session. Each handler reads inputs from here and
    writes its output back, so later steps can consume it.
    """
    session_dir: Path
    shots: list                         # list[Shot]
    duration: float
    base_video: Path                    # original video-only (no audio)
    original_audio: Optional[Path]
    original_video: Optional[Path] = None      # original input video (with audio)
    video_caption: str = ""
    instruction: str = ""
    subtasks: list = field(default_factory=list)
    allow_full_replan: bool = False
    # Full-video video state — set when a global video edit runs.
    current_global_video: Optional[Path] = None
    # Per-shot edited videos — set by per-shot video steps.
    shot_videos: dict = field(default_factory=dict)
    # Full edited audio — set by audio/speech_tts steps.
    edited_audio: Optional[Path] = None
    # Speech clone reference stem (kept for potential reuse/inspection).
    speech_stem: Optional[Path] = None
    # Last Qwen TTS / Voice Design generated speech audio. The lipsync
    # handler uses this as the SOLE driving audio (padded to shot
    # duration) — feeding the residual+cloned mix to Sync Lipsync 2
    # confuses the model when the cloned line is shorter than the
    # shot, since the tail is BGM-only and the model can't decide
    # what the mouth should do.
    last_cloned_voice: Optional[Path] = None
    # Pre-mix generated audio from the last audio step — used by the
    # "remix only" retry branch when the evaluator flags gen_quiet
    # without content failure. Skipping MMAudio saves ~60s per retry.
    last_generated_audio: Optional[Path] = None
    last_gen_vol: float = 0.7
    # Most recent successfully-completed audio SubTask. Used by the
    # mix evaluator's needs_regenerate path to re-run the offending
    # generation step when the final mix flags content (not volume)
    # breakage. None when no audio step has run.
    last_audio_subtask: Any = None   # SubTask | None
    # Track whether any per-shot video edit landed (for assembly).
    any_per_shot_video_edit: bool = False
    # Track whether any global video edit landed.
    any_global_video_edit: bool = False
    # Completed step id → arbitrary artifact record (for debugging).
    completed: dict = field(default_factory=dict)
    # step_id → (best_score, best_audio_path, best_reason). Updated
    # after each audio attempt so that if all retries fail-but-don't-
    # error, the runner can fall back to the highest-scoring attempt
    # instead of dropping the audio edit entirely.
    best_audio_per_step: dict = field(default_factory=dict)
    # step_id → (best_score, best_video_path, shot_idx_or_None, best_info).
    # Mirror of best_audio_per_step but for video-edit steps. After
    # each retry of `_run_step_video`, the highest-scoring attempt's
    # output is tracked here. The runner uses this to commit the
    # best-scoring attempt to context.shot_videos / current_global_video
    # at the end of the retry loop, even when no attempt PASSes.
    best_video_per_step: dict = field(default_factory=dict)
    # Planner-derived ground-truth for the final-mix evaluator; None
    # when the plan has no audio edits.
    audio_inventory: Any = None   # AudioInventory | None
    # Final mixed-media evaluation state. `replan_request` is populated
    # only when the evaluator identifies a structural failure that local
    # remixing or regeneration cannot repair.
    mix_eval_result: Any = None   # MixEvalResult | None
    replan_request: Any = None    # MixEvalResult | None


# ─────────────────────────────────────────────────────────────────────────
# Tool-unit retry helper (V2): each external audio tool (SAM, MMAudio) is
# wrapped in a small "op" with self-contained retry logic. Branch handlers
# (audio_remove / audio_replace / audio_add) call these ops and don't
# reimplement retry semantics.
#
# Op semantics:
#   - call(prompt) → output_path | None       (the actual tool invocation)
#   - evaluate(output_path) → (score, info)   (per-attempt eval, score 0-1)
#   - improve(prompt, info, history) → str    (minimal-change rewriter)
# Loop:
#   for attempt in [0..max_retries]:
#       out = call(prompt)
#       if out: score, info = evaluate(out); track best
#       if score >= threshold: return success
#       prompt = improve(prompt, info, history)
#   if best_score > 0: return best (sub-threshold but useful)
#   if best_score == 0: return passthrough (input as output)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class OpResult:
    """Result of one tool-unit retry op (e.g. SAM or MMAudio call+retry).

    `output_path` is None ONLY when (a) every attempt's tool call
    crashed AND no fallback_input was provided, or (b) the caller
    explicitly disabled passthrough. Otherwise it's always a usable
    audio file: either a passing/best attempt, or the passthrough input.

    `passthrough` distinguishes "we couldn't make progress, returning
    the input unchanged" from "we found a non-passing but workable
    candidate". Useful for branch handlers that need to know whether
    the op actually did anything.

    `attempts` records every attempt's (score, prompt, info-reason) for
    audit / log inspection.
    """
    output_path: Optional[Path]
    score: float                   # best score across attempts; 0.0 if nothing usable
    passed: bool                   # True iff at least one attempt scored >= threshold
    passthrough: bool              # True iff fell back to fallback_input
    attempts: list[tuple[float, str, str]] = field(default_factory=list)
    final_prompt: str = ""
    info: dict[str, Any] = field(default_factory=dict)   # info from best attempt


async def _retry_op(
    *,
    name: str,
    initial_prompt: str,
    call: Any,                     # async (prompt) -> Path | None
    evaluate: Any,                 # async (Path) -> (float, dict)
    improve: Any,                  # async (prompt: str, info: dict, history: list[str]) -> str
    fallback_input: Optional[Path],
    max_retries: int = 2,
    threshold: float = 0.6,
) -> OpResult:
    """Run *call* up to (max_retries + 1) attempts, evaluating each, and
    return the best result. See module-level comment for semantics.

    Per-attempt failures (call returns None) count as score 0.0 with a
    synthetic "tool crashed" reason — the next attempt still gets a
    rewritten prompt via *improve*.
    """
    history: list[str] = []
    attempts_record: list[tuple[float, str, str]] = []
    best_path: Optional[Path] = None
    best_score = -1.0
    best_info: dict[str, Any] = {}
    prompt = (initial_prompt or "").strip()

    for attempt in range(max_retries + 1):
        try:
            out = await call(prompt)
        except Exception as exc:
            logger.warning("[%s] attempt %d call raised: %s", name, attempt + 1, exc)
            out = None

        if out is None:
            score = 0.0
            info = {"reason": "tool call returned no output", "infra_error": "call_failed"}
        else:
            try:
                score, info = await evaluate(out)
            except Exception as exc:
                logger.warning("[%s] attempt %d eval raised: %s", name, attempt + 1, exc)
                # Eval crash → treat as neutral 0.5 so we don't loop forever.
                score, info = 0.5, {"reason": f"eval crashed: {exc}", "infra_error": "eval_failed"}

        attempts_record.append((score, prompt, str(info.get("reason", ""))[:200]))
        if out is not None and score > best_score:
            best_score = score
            best_path = out
            best_info = dict(info)

        logger.info(
            "[%s] attempt %d/%d score=%.2f (best=%.2f) | %s",
            name, attempt + 1, max_retries + 1, score, max(best_score, 0.0),
            str(info.get("reason", ""))[:120],
        )

        if out is not None and score >= threshold:
            return OpResult(
                output_path=out, score=score, passed=True,
                passthrough=False, attempts=attempts_record,
                final_prompt=prompt, info=dict(info),
            )

        if attempt == max_retries:
            break

        # Improve and retry
        if prompt and prompt not in history:
            history.append(prompt)
        try:
            prompt = await improve(prompt, info, history)
        except Exception as exc:
            logger.warning("[%s] improver raised: %s — keeping current prompt", name, exc)

    # No attempt passed. Return best if it has any positive score.
    if best_path is not None and best_score > 0.0:
        logger.warning(
            "[%s] no attempt reached threshold %.2f; returning best score=%.2f",
            name, threshold, best_score,
        )
        return OpResult(
            output_path=best_path, score=best_score, passed=False,
            passthrough=False, attempts=attempts_record,
            final_prompt=prompt, info=best_info,
        )

    # All attempts scored 0 OR the tool crashed every time.
    # User rule: fall back to this step's INPUT as its OUTPUT.
    logger.warning(
        "[%s] all %d attempts scored 0 — falling back to input passthrough%s",
        name, max_retries + 1,
        "" if fallback_input else " (no fallback input provided)",
    )
    return OpResult(
        output_path=fallback_input, score=0.0, passed=False,
        passthrough=True, attempts=attempts_record,
        final_prompt=prompt, info=best_info,
    )


def _cap_words(text: str, max_words: int) -> str:
    """Hard-truncate *text* to at most *max_words* whitespace-separated
    words. Used on prompts that must stay short regardless of what the
    LLM did — retry improvers drift longer over attempts unless capped."""
    if not text:
        return text
    parts = text.split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])


# SAM Audio prompt shape enforcement — a single `[adjective(s)] noun`
# phrase. Multi-fragment / negated / clausal prompts dilute SAM's
# attention and degrade separation. Used both by the SAM improver and
# by the audio branches when ingesting Phase B's `sam_prompt`.
_SAM_NEG_TOKENS = (
    "excluding", "exclude", "without", "except", "avoid",
    "no ", "not ", "instead of", "but not", "rather than",
    "ignore", "skip",
)


def _sam_strip_negative(text: str) -> str:
    """Cut everything from the first occurrence of a negation token
    onward (whichever is earliest)."""
    if not text:
        return text
    t = text
    low = t.lower()
    for tok in _SAM_NEG_TOKENS:
        idx = low.find(tok)
        if idx > 0:
            t = t[:idx].rstrip(", ")
            low = t.lower()
    return t.strip(", ").strip()


def _sam_to_single_phrase(text: str) -> str:
    """Collapse a multi-fragment SAM prompt into a single noun phrase.

    Strategy:
      1. Drop "and" / "or" connectors → keep the longer side (richer
         modifier set, typically holds the head noun).
      2. Split on comma; pick the longest fragment.
      3. Strip stray punctuation.

    Empirically SAM Audio's separation degrades when several
    comma-separated fragments compete for the encoder's attention; a
    single `[adjective(s)] noun` form is the most reliable shape.
    """
    import re as _re
    t = (text or "").strip().strip('"').strip("'").rstrip(".")
    if not t:
        return t
    # Drop "and" / "or" → take the longer side
    for conj in (" and ", " or "):
        if conj in t.lower():
            parts = _re.split(conj, t, flags=_re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip()]
            if parts:
                t = max(parts, key=lambda p: len(p.split()))
    # Split on commas → pick the longest fragment
    if "," in t:
        fragments = [f.strip() for f in t.split(",") if f.strip()]
        if fragments:
            t = max(fragments, key=lambda f: len(f.split()))
    return t.strip(", .;:").strip()


def _sanitise_sam_prompt(text: str, max_words: int = 8) -> str:
    """SAM prompt cleanup: strip negation → collapse to single phrase →
    cap words. Idempotent — safe to call multiple times."""
    if not text:
        return text
    return _cap_words(
        _sam_to_single_phrase(_sam_strip_negative(text)),
        max_words,
    )


def _generate_silent_aac(out_path: Path, duration: float) -> Path:
    """Generate a silent AAC audio file of the given duration. Used as
    the audio_remove fallback when SAM exhausts retries — for a pure
    "delete sound X" operation, returning silence is closer to the
    user's intent than returning the un-modified original audio."""
    import subprocess as _sp
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t", f"{max(0.05, float(duration)):.3f}",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    ]
    r = _sp.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"silent AAC generation failed: {r.stderr[:200]}")
    return out_path


def _wav_to_aac(src: Path, dst: Path) -> Path:
    """Re-encode a (PCM) WAV file to AAC. SAM Audio returns
    `pcm_s16le` WAV stems; some downstream paths assume AAC for
    `context.edited_audio`. Doing the conversion at the source keeps
    the rest of the pipeline format-uniform."""
    import subprocess as _sp
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:a", "aac", "-b:a", "192k",
        str(dst),
    ]
    r = _sp.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"WAV→AAC conversion failed: {r.stderr[:200]}")
    return dst


# Keyword → canonical negative phrase. Covers the sounds MMAudio most
# commonly hallucinates onto food / impact / animal scenes. Matched as
# whole words (case-insensitive) against the evaluator's free-form
# "what I heard" + reason text, so the contamination retry branch can
# reinforce negatives even when the eval did not populate
# `forbidden_heard` structurally.
_CONTAMINANT_KEYWORDS: list[tuple[str, str]] = [
    ("footstep", "footsteps"), ("footsteps", "footsteps"),
    ("stepping", "footsteps"), ("walking", "footsteps"),
    ("hiss", "hissing"), ("hissing", "hissing"),
    ("crunch", "crunching"), ("crunching", "crunching"),
    ("squish", "squishing"), ("squishing", "squishing"),
    ("crushing", "crushing"), ("thud", "thumping"),
    ("thumping", "thumping"), ("knock", "knocking"),
    ("music", "music"), ("bgm", "music"),
    ("speech", "speech"), ("talking", "talking"),
    ("humming", "humming"), ("singing", "singing"),
    ("whisper", "whispering"), ("moan", "moaning"),
]


def _extract_contaminant_phrases(text: str) -> str:
    """Pull concrete hallucinated-sound nouns out of a free-form eval
    sentence. Returns a comma-joined short list of canonical phrases
    suitable for appending to `mmaudio_negative_prompt`. Limited to
    at most 6 hits to keep the negative list focused."""
    if not text:
        return ""
    import re as _re
    low = text.lower()
    found: list[str] = []
    for kw, canon in _CONTAMINANT_KEYWORDS:
        if _re.search(rf"\b{_re.escape(kw)}\w*\b", low) and canon not in found:
            found.append(canon)
        if len(found) >= 6:
            break
    return ", ".join(found)


def _cap_negative_list(neg: str, max_items: int = 8) -> str:
    """Keep *neg* (comma-separated items) to the first *max_items*
    unique lowercased phrases. MMAudio's negative prompt loses
    effectiveness when it balloons — stay focused."""
    if not neg:
        return ""
    out: list[str] = []
    for token in neg.split(","):
        t = token.strip().rstrip(".;").lower()
        if t and t not in out:
            out.append(t)
        if len(out) >= max_items:
            break
    return ", ".join(out)


_NON_MUSIC_SFX_NEGATIVES = (
    "music",
    "background music",
    "soundtrack",
    "melody",
    "singing",
    "vocals",
)


def _with_non_music_sfx_negatives(items: list[str], max_items: int = 12) -> str:
    """Negative prompt for concrete SFX generation.

    MMAudio can turn repetitive SFX descriptions into a background track,
    so non-music SFX always explicitly forbids music-like outputs even
    when the planner did not mention music in the preserve inventory.
    """
    return _cap_negative_list(
        ", ".join([*items, *_NON_MUSIC_SFX_NEGATIVES]),
        max_items=max_items,
    )


def _reason_has_music_hallucination(reason: str) -> bool:
    text = (reason or "").lower()
    return any(
        token in text
        for token in (
            "music", "bgm", "soundtrack", "melody", "singing",
            "vocal", "hallucinat", "unwanted", "unrelated",
        )
    )


def _inventory_annotation(context: "_StepContext") -> str:
    """Format the planner-derived audio inventory as a short annotation
    to append to separation/generation intent strings. Gives the stage
    evaluators a concrete preserve/remove/add check-list instead of
    inferring from the edit prompt alone. Returns "" when unavailable."""
    inv = getattr(context, "audio_inventory", None)
    if inv is None:
        return ""
    parts: list[str] = []
    if getattr(inv, "preserve", None):
        parts.append("MUST preserve: " + ", ".join(inv.preserve))
    if getattr(inv, "remove", None):
        parts.append("MUST remove: " + ", ".join(inv.remove))
    if getattr(inv, "add", None):
        parts.append("MUST add: " + ", ".join(inv.add))
    if getattr(inv, "replace", None):
        rep = ", ".join(
            f"{p.get('from','?')} → {p.get('to','?')}" for p in inv.replace
        )
        parts.append("MUST replace: " + rep)
    if not parts:
        return ""
    return (
        "\n\n[audio_inventory — ground-truth check-points]\n" + "\n".join(parts)
    )


def _mm_criteria_annotation(subtask) -> str:
    """Append the planner's `mmaudio_eval_criteria` to the MMAudio
    evaluator's intent string. The MMAudio evaluator scores raw
    generated audio before mixing, so its criteria should be scoped
    to the generated layer only — these are emitted by Phase B and
    forwarded verbatim."""
    crits = list(getattr(subtask, "mmaudio_eval_criteria", []) or [])
    if not crits:
        return ""
    body = "\n".join(f"  - {c}" for c in crits)
    return (
        "\n\n[mmaudio_eval_criteria — task-specific checks for the "
        "generated audio]\n" + body
    )


def _topo_order(subtasks: list) -> list:
    """Return SubTasks in (deps-respecting, step-ascending) order. Cycles
    fall back to step order."""
    by_step = {t.step: t for t in subtasks}
    visited: set = set()
    ordered: list = []

    def visit(s, stack):
        if s.step in visited:
            return
        if s.step in stack:
            logger.warning("[Runner] cyclic deps at step %d — ignoring", s.step)
            return
        stack.add(s.step)
        for dep in s.depends_on:
            if dep in by_step:
                visit(by_step[dep], stack)
        stack.discard(s.step)
        visited.add(s.step)
        ordered.append(s)

    for t in sorted(subtasks, key=lambda x: x.step):
        visit(t, set())
    return ordered


class EditingPipeline:
    """
    Full editing pipeline:
        preprocess → plan → execute video → execute audio → postprocess

    Audio branch:
      - Audio subtasks (is_audio=True) are collected from the plan.
      - Their descriptions are combined into a single prompt.
      - MMAudio V2 generates synchronised audio from the *edited* video.
      - The generated audio is mixed with the original audio in postprocess.
    """

    def __init__(self, config: AppConfig | None = None):
        self.cfg = config or AppConfig()
        self.workspace = self.cfg.ensure_workspace()

        # Build components. Planner needs the active video backend so
        # Phase B emits backend-specific video prompts and validates
        # the right word cap.
        self.planner = Planner(
            self.cfg.llm,
            video_backend=getattr(self.cfg.tools, "video_backend", "wan"),
        )
        self.evaluator = Evaluator(
            self.cfg.llm,
            quality_threshold=self.cfg.pipeline.eval_quality_threshold,
            consistency_threshold=self.cfg.pipeline.eval_consistency_threshold,
        )
        self.registry = build_default_registry(self.cfg.tools)

    # ── audio helpers ─────────────────────────────────────────────────

    async def _run_audio_separation(
        self,
        video_path: Path,
        description: str,
        output_dir: Path,
        audio_path: Path | None = None,
        expect_prominent_target: bool = False,
    ) -> Path | None:
        """Run audio separation tool to remove *description* and return the residual.

        Parameters
        ----------
        video_path  : Video file (used to extract audio if audio_path is None).
        description : Text description of the sound to separate/remove.
        output_dir  : Directory for intermediate and output files.
        audio_path  : If provided, pass this audio alongside video_path.
                      The fal SAM Audio visual tool muxes it with the current
                      visual snapshot before calling visual-separate.
        """
        sep_tool = self.registry.find_audio_separation_tool()
        if sep_tool is None:
            logger.warning("[Pipeline] No audio separation tool — skipping")
            return None

        params: dict = {
            "description": description,
            "mode": "remove",
            "expect_prominent_target": expect_prominent_target,
        }
        if audio_path is not None:
            params["audio_path"] = str(audio_path)

        result = await sep_tool.execute(
            video_path=video_path,
            action="audio_remove",
            params=params,
            output_dir=output_dir,
        )
        if result.success and result.output_path:
            logger.info("[Pipeline] Separated '%s' → residual: %s",
                        description, result.output_path)
            # Stash the target stem path for callers that need both
            # (e.g. audio_replace's stage-1 verification of WHAT got
            # captured into the target stem). The fal_samaudio tool
            # writes target_path into raw_response.
            target_str = (result.raw_response or {}).get("target_path")
            if target_str:
                # Attach as a side-channel attribute so we don't break
                # the existing return type. Callers can read it via
                # `self._last_separation_target`.
                self._last_separation_target = Path(target_str)
            else:
                self._last_separation_target = None
            return result.output_path
        self._last_separation_target = None
        logger.warning("[Pipeline] Audio separation failed: %s", result.error_msg)
        return None

    async def _run_mmaudio_generation(
        self,
        edited_video: Path,
        prompt: str,
        session_dir: Path,
        duration: float,
        negative_prompt: str = "",
        mask_away_clip: bool = False,
        guidance_scale: float = 4.5,
    ) -> Path | None:
        """Run MMAudio to generate audio for the edited video."""
        audio_tool = self.registry.find_audio_tool()
        if audio_tool is None:
            logger.warning("[Pipeline] No audio generation tool — skipping")
            return None

        audio_dir = session_dir / "audio_gen"
        result = await audio_tool.execute(
            video_path=edited_video,
            action="audio_generate",
            params={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "mask_away_clip": mask_away_clip,
                "guidance_scale": guidance_scale,
                "duration": duration,
            },
            output_dir=audio_dir,
        )
        if result.success and result.output_path:
            logger.info("[Pipeline] audio generated → %s", result.output_path)
            return result.output_path
        logger.warning("[Pipeline] audio generation failed: %s", result.error_msg)
        return None

    async def _run_lipsync(
        self,
        video_path: Path,
        audio_path: Path,
        output_dir: Path,
        sync_mode: str = "cut_off",
    ) -> Path | None:
        """Run Sync Lipsync 2 to re-animate mouth movement in *video_path*
        so it matches the new *audio_path*. Returns the output video
        path, or None on failure.
        """
        tool = self.registry.find_lipsync_tool()
        if tool is None:
            logger.warning("[Pipeline] No lipsync tool — skipping")
            return None
        result = await tool.execute(
            video_path=video_path,
            action="speech_lipsync",
            params={"audio_path": str(audio_path), "sync_mode": sync_mode},
            output_dir=output_dir,
        )
        if result.success and result.output_path:
            logger.info("[Pipeline] lipsync → %s", result.output_path)
            return result.output_path
        logger.warning("[Pipeline] lipsync failed: %s", result.error_msg)
        return None

    async def _lipsync_target_shots_with_audio(
        self,
        source_video: Path,
        edited_audio: Path,
        shots: list,
        target_indices: list[int],
        output_dir: Path,
    ) -> Path | None:
        """Lipsync the shots named in *target_indices* using per-shot
        slices of *edited_audio* (the complete, already-edited audio
        track), then concat with the untouched shots. Returns the
        full-length video-only result, or None if there is nothing to do.

        The per-shot audio slice aligns speech timing with the shot,
        so the generated mouth movement matches what the final audio
        will contain at that moment.
        """
        import subprocess
        from av_editor.core.shot_slicer import (
            concat_shots, slice_audio, slice_shot,
        )
        if not target_indices or not shots:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        target_set = set(target_indices)
        pieces: list[Path] = []

        def _normalize_shot(src: Path, dst: Path) -> Path:
            """Re-encode to our canonical per-shot format: H.264 yuv420p,
            no audio (audio is handled globally in postprocess). Ensures
            the concat demuxer can -c copy the pieces together."""
            dst.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "ffmpeg", "-y", "-i", str(src),
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-an", str(dst),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"normalize failed: {r.stderr}")
            return dst

        for shot in shots:
            src = output_dir / f"src_shot_{shot.index:03d}.mp4"
            slice_shot(source_video, shot, src)
            if shot.index not in target_set:
                pieces.append(src)
                continue

            # Slice the matching audio segment out of the complete
            # edited audio — this is what the mouth should animate to.
            shot_audio = output_dir / f"shot_{shot.index:03d}_audio.wav"
            slice_audio(edited_audio, shot.start, shot.end, shot_audio)

            ls_out = await self._run_lipsync(
                video_path=src,
                audio_path=shot_audio,
                output_dir=output_dir / f"shot_{shot.index:03d}_ls",
                sync_mode="cut_off",  # audio slice already matches shot duration
            )
            if ls_out:
                normalized = output_dir / f"shot_{shot.index:03d}_ls_norm.mp4"
                try:
                    _normalize_shot(ls_out, normalized)
                    pieces.append(normalized)
                except Exception as exc:
                    logger.warning(
                        "[Pipeline] normalize lipsync output failed: %s — "
                        "using raw lipsync and forcing concat re-encode", exc,
                    )
                    pieces.append(ls_out)
            else:
                pieces.append(src)

        from av_editor.core.shot_slicer import _probe_fps, _probe_size
        src_fps = _probe_fps(source_video)
        src_size = _probe_size(source_video)
        final = output_dir / "lipsync_concat.mp4"
        concat_shots(
            pieces, final, reencode=True,
            target_fps=src_fps, target_size=src_size,
        )
        return final

    async def _separate_target_and_residual(
        self,
        video_path: Path,
        output_dir: Path,
        speaker_description: str = "human speech voice dialogue",
        audio_path: Path | None = None,
    ) -> tuple[Path | None, Path | None]:
        """Run audio separation ONCE and return both stems as
        ``(target, residual)``. The underlying SAM Audio API always
        produces both files; this helper avoids paying for two calls
        when the speech branch needs both the clone reference (target)
        and the preserved background (residual).

        If *audio_path* is given, it is passed alongside *video_path*.
        The fal SAM Audio visual tool muxes both streams before calling
        visual-separate, which is useful when *video_path* is a video-only
        snapshot after visual edits.
        """
        sep_tool = self.registry.find_audio_separation_tool()
        if sep_tool is None:
            logger.warning("[Pipeline] No audio separation tool — skipping")
            return None, None

        params: dict = {
            "description": speaker_description,
            "mode": "remove",
        }
        if audio_path is not None:
            params["audio_path"] = str(audio_path)

        try:
            result = await sep_tool.execute(
                video_path=video_path,
                action="audio_remove",
                params=params,
                output_dir=output_dir,
            )
        except Exception as exc:
            logger.warning("[Pipeline] speaker separation failed: %s", exc)
            return None, None

        if not result.success:
            logger.warning(
                "[Pipeline] speaker separation returned no output: %s",
                result.error_msg,
            )
            return None, None

        raw = result.raw_response or {}
        target_path = raw.get("target_path")
        residual_path = raw.get("residual_path")
        target = Path(target_path) if target_path else None
        residual = Path(residual_path) if residual_path else None
        logger.info(
            "[Pipeline] separated '%s' → target: %s | residual: %s",
            speaker_description,
            target.name if target else "None",
            residual.name if residual else "None",
        )
        return target, residual

    async def _run_speech_clone(
        self,
        reference_audio: Path,
        text: str,
        output_dir: Path,
        reference_text: str = "",
        language: str = "auto",
    ) -> Path | None:
        """Run Qwen3 TTS Voice Clone to synthesise *text* in the voice of
        *reference_audio*. Returns the generated speech WAV path, or None
        on failure.
        """
        speech_tool = self.registry.find_speech_tool()
        if speech_tool is None:
            logger.warning("[Pipeline] No speech tool — skipping speech clone")
            return None

        result = await speech_tool.execute(
            video_path=Path(),                     # unused by speech tool
            action="speech_replace_full",
            params={
                "reference_audio": str(reference_audio),
                "text": text,
                "reference_text": reference_text,
                "language": language,
            },
            output_dir=output_dir,
        )
        if result.success and result.output_path:
            logger.info("[Pipeline] speech cloned → %s", result.output_path)
            return result.output_path
        logger.warning("[Pipeline] speech clone failed: %s", result.error_msg)
        return None

    async def _improve_video_prompt(
        self,
        current_desc: str,
        eval_reason: str,
        action: str,
    ) -> str:
        """Rewrite a video edit's description based on VLM evaluator
        feedback. Used by the unified step runner when a video step
        fails its per-step eval.

        LLM: official Gemini API, primary gemini-3.1-pro-preview with fallback
        gemini-2.5-flash.
        """
        # Seedance is reference-to-video generation: it regenerates the whole
        # clip and cannot honor pixel-level preservation clauses.
        is_seedance = getattr(self.cfg.tools, "video_backend", "wan") == "seedance"
        preserve_block = (
            "Preservation clause: DO NOT add any 'keep/same/unchanged' "
            "preservation clause — this backend regenerates the whole clip "
            "and cannot honor pixel-level preservation, so such clauses add "
            "nothing. If the edit drifted, just describe the intended result "
            "more concretely.\n\n"
            if is_seedance else
            "Preservation clause — CONDITIONAL:\n"
            "Only when the evaluator explicitly flags a FIDELITY/"
            "CONSISTENCY problem (the subject's identity, pose, clothing, "
            "nearby props, or background changed unintentionally), append "
            "a COMMA clause with the imperative 'keep': "
            "'..., keep the background unchanged.' Name the 1-2 things "
            "that drifted; use 'keep' (not the participle 'keeping') and a "
            "comma, not a run-on. If the feedback only complains the edit "
            "itself was too weak/unclear, DO NOT add a preservation "
            "clause — strengthen the edit instead.\n\n"
        )
        system = (
            "You are an expert prompt engineer for a video-to-video "
            "editing model. The prompt must be an IMPERATIVE EDIT "
            "INSTRUCTION, not a scene caption — the model already sees "
            "the source clip and does not need it described.\n\n"
            "Style rules:\n"
            "- Start with a COMMON edit verb. Prefer ONLY: Change, "
            "Replace, Add, Remove, Make.\n"
            "- Avoid rarer/colloquial/director-style verbs such as "
            "shift, tilt, swap, turn into, give it. Rewrite them with "
            "common verbs: 'Change the camera to a low angle', "
            "'Replace the bin with a pot', 'Change her dress to red'.\n"
            "- For motion_edit/action changes, use this canonical form: "
            "'Change <subject>'s action to <new action>'. Do not use "
            "'Make <subject> ...' for motion_edit.\n"
            "- NO HEDGING. Don't write 'look like' / 'looks like' / "
            "'as if'. The model treats hedges as soft suggestions, "
            "not edits — write the edit as a direct change.\n"
            "- For STYLE TRANSFER use the canonical form 'Change the "
            "video to <X> style' (or 'Change to <X> style'). Don't "
            "be colloquial — no 'make it cartoony' / 'give it an "
            "anime vibe'. Use a concrete style name.\n"
            "- Keep it CONCISE but a NATURAL, grammatically complete "
            "sentence — keep articles and prepositions, normal sentence "
            "structure. Do NOT clip into a telegraphic fragment. Aim "
            "≤ 16 words including any preservation clause.\n"
            "- Adjectives are fine when they carry the user's intent "
            "('low-angle', 'deep red', 'empty pot'). Keep them short "
            "and concrete.\n"
            "- NO 'do not ...' clauses. NO 'shot' / 'keyframe' / "
            "'segment'.\n\n"
            + preserve_block +
            "Given the current prompt and the evaluator feedback, "
            "rewrite the prompt so the next attempt addresses the "
            "failure. Return ONLY the new prompt — no quotes, no "
            "explanation."
        )
        user = (
            f"Action: {action}\n"
            f"Current prompt:\n\"{current_desc}\"\n\n"
            f"Evaluator feedback:\n{eval_reason}\n\n"
            "Rewrite the prompt."
        )
        try:
            from av_editor.core._gemini_client import gemini_with_fallback
            raw = await asyncio.to_thread(
                gemini_with_fallback,
                gemini_api_key=self.cfg.llm.gemini_api_key,
                primary_model=self.cfg.llm.gemini_model,
                fallback_model="gemini-2.5-flash",
                system_prompt=system,
                user_text=user,
                json_response=False,
                temperature=0.3,
                max_output_tokens=9999,
                component="VideoPromptImprover",
            )
            new = (raw or "").strip().strip('"')
            return new if new else current_desc
        except Exception as exc:
            logger.warning(
                "[Pipeline] _improve_video_prompt failed: %s — keeping original",
                exc,
            )
            return current_desc

    async def _improve_mmaudio_prompts(
        self,
        current_positive: str,
        current_negative: list[str],
        eval_reason: str,
        eval_score: float = 0.0,
        history: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Single chained LLM call: derives `missing` / `unwanted` from
        the evaluator's feedback, then uses that SAME analysis to emit
        BOTH a rewritten positive prompt AND a list of new tokens to
        append to the negative list — so the two updates are coordinated
        rather than independent.

        Returns
        -------
        (new_positive, neg_to_add)
            • `new_positive` — the rewritten positive prompt (already
              sanitised: ≤ 12 words, no comma, no 'and'/'or', no
              instructional verbs). Falls back to the sanitised current
              prompt if the LLM call fails or returns junk.
            • `neg_to_add` — list of short lowercase tokens the caller
              should de-dup-append to its `current_negative` list. Empty
              if the previous attempt didn't have any obvious leak.
        """
        import json as _json
        history = history or []

        # ---- positive-prompt sanitiser (mirrors SAM improver style) --
        def _sanitise_positive(raw: str) -> str:
            s = (raw or "").strip().strip('"').strip("'")
            if not s:
                return ""
            for verb in ("add ", "generate ", "make ", "create ", "produce "):
                if s.lower().startswith(verb):
                    s = s[len(verb):]
                    break
            s = s.replace(",", " ").replace(";", " ").replace("/", " ")
            s = " ".join(
                t for t in s.split()
                if t.lower() not in ("and", "or", "&")
            ).strip()
            words = s.split()
            if len(words) > 12:
                s = " ".join(words[:12])
            return s

        # ---- deterministic anchor stripper (used as last-resort fallback)
        # Cuts the first occurrence of a temporal/causal/visual-anchor
        # connector and everything after it, leaving just the leading
        # sound description. Used when the LLM rewriter gives up.
        def _strip_timing_clause(raw: str) -> str:
            s = (raw or "").strip()
            if not s:
                return ""
            connectors = (
                " when ", " while ", " as ", " after ", " before ",
                " under ", " at ", " on ", " against ", " over ",
                " synchron",  # synchronised / synchronized / synchronising
            )
            low = s.lower()
            cut = len(s)
            for c in connectors:
                idx = low.find(c)
                if idx >= 0 and idx < cut:
                    cut = idx
            return s[:cut].strip()

        # ---- negative-token sanitiser (lowercase, dedup, drop empties)
        def _sanitise_neg_tokens(raw_list) -> list[str]:
            if not isinstance(raw_list, list):
                return []
            out: list[str] = []
            for item in raw_list:
                if not isinstance(item, str):
                    continue
                t = item.strip().rstrip(".;,").lower()
                # Drop empty / overly long phrases.
                if not t or len(t) > 40:
                    continue
                # Negatives must not include 'and'/'or' connectors —
                # split if needed.
                for piece in t.replace(",", " ").split():
                    p = piece.strip()
                    if (
                        p
                        and p not in ("and", "or", "&")
                        and p not in current_negative
                        and p not in out
                    ):
                        out.append(p)
            return out

        current_word_count = max(2, len(current_positive.split()))
        word_cap = min(12, max(current_word_count, 6))

        history_text = (
            "\nPrompts already tried (DO NOT repeat verbatim):\n"
            + "\n".join(f"  - {p}" for p in history)
        ) if history else ""
        cur_neg_text = (
            ", ".join(current_negative) if current_negative else "(none)"
        )

        system = (
            "You are improving an MMAudio V2 audio-generation prompt "
            "after a failed attempt. You will perform TWO chained "
            "reasoning steps in ONE response:\n\n"
            "STEP 1 — IDENTIFY (read the evaluator feedback):\n"
            "  • `missing`  — REQUESTED elements that were NOT heard "
            "in the previous attempt (≤ 60 chars, comma-separated, "
            "empty if all present).\n"
            "  • `unwanted` — FORBIDDEN or HALLUCINATED sounds that WERE "
            "heard but should not have been (≤ 60 chars, comma-"
            "separated, empty if clean). Read the evaluator feedback "
            "carefully — anything explicitly called out as "
            "'forbidden', 'unwanted', 'hallucinated', 'should not "
            "appear', or that the evaluator listed in `what_you_hear` "
            "but is NOT what we asked for, MUST go into `unwanted`.\n\n"
            "STEP 2 — REWRITE (use the analysis from Step 1):\n"
            "  • `new_positive` — rewrite the positive prompt so it "
            "describes ONLY the `missing` element(s). Use a different "
            "wording / synonym so MMAudio doesn't latch onto the same "
            "tokens as last time. Do NOT describe the `unwanted` "
            "sound in the positive prompt — not even as a timing or "
            "causal qualifier (e.g. avoid 'when X happens', "
            "'synchronised with X', 'as X moves'). Mentioning the "
            "unwanted sound primes MMAudio to regenerate it; just "
            "describe the wanted sound on its own.\n"
            "      Example: if `unwanted=dog barking`, prefer "
            "'small metal bell jingling' over 'metal jingle when "
            "dog barks'.\n"
            "  • `neg_to_add` — every distinct unwanted item from "
            "STEP 1 should appear here as a short lowercase token "
            "(deduped against the existing negative list).\n\n"
            "POSITIVE-PROMPT OUTPUT SHAPE (HARD CONSTRAINT):\n"
            "  • A single DESCRIPTIVE phrase chain. NO instructional "
            "verbs (no 'Add' / 'Generate' / 'Make' / 'Create').\n"
            "  • NO comma. NO 'and' / 'or'. NO clause separators.\n"
            f"  • At most 12 simple English words (target ≈ "
            f"{current_word_count}, hard cap {word_cap}).\n"
            "  • Plain everyday vocabulary only.\n\n"
            "VOCABULARY (CRITICAL):\n"
            "  • MMAudio anchors on concrete sound nouns + 1-2 plain "
            "adjectives (e.g. 'a dog barking', 'rain on metal roof').\n"
            "  • Forbidden adjectives in the positive prompt:\n"
            "      cinematic / atmospheric / ethereal / dreamy /\n"
            "      ambient(adj) / haunting / melancholic / sonorous /\n"
            "      resonant / orchestral / discordant\n"
            "  • Preferred substitutes:\n"
            "      texture: dry / wet / metallic / wooden / glassy\n"
            "      impact:  thud / clack / crack / thump / smack\n"
            "      water:   splash / drip / gurgle / spray\n"
            "      crowd:   cheer / chatter / murmur / clap\n\n"
            "Return STRICT JSON with EXACTLY these four keys:\n"
            "{\n"
            '  "missing": "<short comma-separated list, or empty>",\n'
            '  "unwanted": "<short comma-separated list, or empty>",\n'
            '  "new_positive": "<rewritten positive prompt>",\n'
            '  "neg_to_add": ["<token>", ...]\n'
            "}\n"
            "No prose, no markdown, JSON only."
        )

        user = (
            f"Current positive prompt: \"{current_positive}\"\n"
            f"Current negative tokens: {cur_neg_text}\n"
            f"Evaluator score: {eval_score:.2f}\n"
            f"Evaluator feedback: {eval_reason}"
            f"{history_text}\n\n"
            "Run STEP 1 then STEP 2 and emit the JSON."
        )

        import re as _re

        def _strip_fence(s: str) -> str:
            """Drop markdown code fences (```json ... ``` or ``` ... ```)
            that Gemini sometimes wraps around its JSON. Idempotent on
            plain JSON."""
            if not s:
                return s
            t = s.strip()
            if t.startswith("```"):
                # Drop opening fence (```json or ```), then trailing ```.
                t = _re.sub(r"^```[a-zA-Z]*\s*", "", t)
                t = _re.sub(r"\s*```$", "", t.rstrip())
            return t.strip()

        def _regex_salvage(raw: str) -> dict | None:
            """When json.loads fails, hand-extract the 4 fields we care
            about. Tolerant to mismatched quotes / trailing commas / fence
            residue. Returns None if `new_positive` can't be salvaged."""
            if not raw:
                return None
            np_m = _re.search(
                r'"new_positive"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw,
            )
            if not np_m:
                return None
            mi_m = _re.search(
                r'"missing"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw,
            )
            uw_m = _re.search(
                r'"unwanted"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw,
            )
            neg_m = _re.search(r'"neg_to_add"\s*:\s*\[([^\]]*)\]', raw)
            neg_list: list[str] = []
            if neg_m:
                for item in _re.findall(r'"([^"\\]+)"', neg_m.group(1)):
                    neg_list.append(item)
            return {
                "missing": mi_m.group(1) if mi_m else "",
                "unwanted": uw_m.group(1) if uw_m else "",
                "new_positive": np_m.group(1),
                "neg_to_add": neg_list,
            }

        from av_editor.core._gemini_client import gemini_with_fallback

        async def _ask(temperature: float, extra_user: str = "") -> dict | None:
            """One MMAudio improver call through the official Gemini API."""
            try:
                raw = await asyncio.to_thread(
                    gemini_with_fallback,
                    gemini_api_key=self.cfg.llm.gemini_api_key,
                    primary_model=self.cfg.llm.gemini_model,
                    fallback_model="gemini-2.5-flash",
                    system_prompt=system,
                    user_text=user + extra_user,
                    json_response=True,
                    temperature=temperature,
                    max_output_tokens=9999,
                    component="MMAudioImprover",
                )
            except Exception as exc:
                logger.warning(
                    "[MMAudio-improver] both primary + fallback failed: %s",
                    exc,
                )
                return None
            raw = _strip_fence(raw or "")
            logger.info(
                "[MMAudio-improver] raw response (len=%d): %r",
                len(raw), raw[:600],
            )
            if not raw:
                return None
            try:
                parsed = _json.loads(raw)
                logger.info(
                    "[MMAudio-improver] parsed JSON: %s",
                    {k: (v[:80] if isinstance(v, str) else v) for k, v in parsed.items()}
                    if isinstance(parsed, dict) else parsed,
                )
                return parsed
            except Exception as exc:
                logger.warning(
                    "[MMAudio-improver] json.loads failed: %s — trying salvage",
                    exc,
                )
                salvaged = _regex_salvage(raw)
                if salvaged is not None:
                    logger.info(
                        "[MMAudio-improver] regex-salvaged JSON: %s",
                        salvaged,
                    )
                    return salvaged
                return None

        def _extract(parsed: dict | None) -> tuple[str, list[str], str, str]:
            if not parsed:
                return "", [], "", ""
            new_pos = _sanitise_positive(parsed.get("new_positive", ""))
            neg_add = _sanitise_neg_tokens(parsed.get("neg_to_add", []))
            missing = (parsed.get("missing") or "").strip()
            unwanted = (parsed.get("unwanted") or "").strip()
            return new_pos, neg_add, missing, unwanted

        # Pass 1: standard temperature.
        parsed = await _ask(0.3)
        new_pos, neg_add, missing, unwanted = _extract(parsed)
        accept = (
            new_pos and new_pos not in history and new_pos != current_positive
        )
        if accept:
            logger.info(
                "[MMAudio-improver] missing=%r unwanted=%r neg+=%s pos→%r",
                missing[:60], unwanted[:60], neg_add, new_pos,
            )
            return new_pos, neg_add

        # Pass 2: duplicate / empty → escalate temperature + blacklist.
        blacklist = sorted({w.lower() for p in history for w in p.split()})
        parsed2 = await _ask(
            0.9,
            extra_user=(
                "\n\nYour previous JSON had an empty or duplicate "
                "`new_positive`. You MUST emit a DIFFERENT positive "
                "prompt. Words already tried (avoid as adjectives): "
                f"{', '.join(blacklist) if blacklist else '(none)'}. "
                "Pick a synonym you have NOT used."
            ),
        )
        new_pos2, neg_add2, missing2, unwanted2 = _extract(parsed2)
        if new_pos2 and new_pos2 not in history and new_pos2 != current_positive:
            logger.info(
                "[MMAudio-improver] (pass2) missing=%r unwanted=%r "
                "neg+=%s pos→%r",
                missing2[:60], unwanted2[:60], neg_add2, new_pos2,
            )
            return new_pos2, neg_add2

        # Deterministic fallback: when both LLM passes return empty /
        # duplicate, strip the timing/anchor clause from the current
        # prompt (`when X moves`, `as X happens`, `under Y`, `at Z`) so
        # at least the next attempt isn't an exact replay. The visual
        # anchor is the most common reason MMAudio leaks the unwanted
        # sound (it primes on the trigger object), so dropping it is
        # often the right move when the LLM rewriter has nothing to say.
        det_pos = _sanitise_positive(_strip_timing_clause(current_positive))
        if (
            det_pos
            and det_pos not in history
            and det_pos != current_positive
        ):
            logger.info(
                "[MMAudio-improver] LLM gave no rewrite; deterministic "
                "anchor-strip → %r",
                det_pos,
            )
            return det_pos, (neg_add or neg_add2)

        # Final fallback: keep current positive (sanitised); still return
        # any negative tokens we managed to extract from either pass —
        # losing those would silently undo the negative-update path.
        fallback_neg = neg_add or neg_add2
        fallback_pos = _sanitise_positive(current_positive) or current_positive
        if fallback_neg:
            logger.info(
                "[MMAudio-improver] positive unchanged; still appending "
                "neg+=%s",
                fallback_neg,
            )
        return fallback_pos, fallback_neg

    # ═══════════════════════════════════════════════════════════════
    # Unified step-runner: dispatch + evaluator + retry per SubTask
    # ═══════════════════════════════════════════════════════════════

    # Per-action retry budgets. Video-side tasks (V2V edits and
    # lipsync) are expensive and the VLM evaluator is noisy, so we cap
    # them at 1 retry (2 attempts total). Audio / speech_tts keep a
    # looser budget so the evaluator-driven speaker_description
    # rewrite has room to converge.
    _MAX_RETRIES_VIDEO = 1
    # V2: each audio branch handles its own SAM/MMAudio retry via
    # `_retry_op`, and the replace branch has its own in-branch volume-
    # retry loop. The outer per-step retry would re-run the WHOLE branch
    # (including all 3 SAM tries), wasting time and money. Set to 0 so
    # the branch runs exactly once.
    _MAX_RETRIES_AUDIO = 0
    # speech_tts / speech_swap have NO internal SAM retry — the handler
    # is monolithic and returns False on a single SAM failure. They
    # rely entirely on the OUTER retry loop (with the dedicated
    # `_improve_speaker_description_for_step` / `_improve_voice_design_for_step`
    # improvers) to converge. Match V1's audio budget (3 attempts).
    _MAX_RETRIES_SPEECH = 2

    async def _run_subtasks_ordered(
        self,
        subtasks: list,
        context: _StepContext,
    ) -> None:
        """Execute all SubTasks in topo-sorted order, each with its own
        evaluator + retry loop. Mutates *context* in place."""
        ordered = _topo_order(subtasks)
        logger.info(
            "[Runner] executing %d subtask(s) in order: %s",
            len(ordered), [s.step for s in ordered],
        )
        from av_editor.schema import EditAction as _EA
        SPEECH_NEEDS_OUTER_RETRY = {_EA.SPEECH_TTS, _EA.SPEECH_SWAP}
        for st in ordered:
            if st.action in SPEECH_NEEDS_OUTER_RETRY:
                # speech_tts/swap: monolithic handler, no internal SAM
                # retry — give the outer loop room to call the
                # speaker_description / voice_description improver.
                retries = self._MAX_RETRIES_SPEECH
            elif st.action.is_audio:
                # audio_remove / replace / add / volume_adjust: each
                # branch has internal _retry_op so outer retry is 0.
                retries = self._MAX_RETRIES_AUDIO
            else:
                retries = self._MAX_RETRIES_VIDEO
            await self._run_one_subtask(st, context, retries)

    async def _run_one_subtask(
        self,
        subtask,
        context: _StepContext,
        max_retries: int,
        attempt_offset: int = 0,
    ) -> None:
        """Dispatch + evaluate + retry a single SubTask.

        `attempt_offset` shifts the per-handler attempt index so a re-entrant
        call (e.g. mix-eval regen) writes into a fresh `attempt_NN/` dir
        instead of clobbering the original run's artifacts.
        """
        from av_editor.schema import EditAction as _EA
        action = subtask.action
        logger.info(
            "[Runner] step %d: action=%s shot_index=%s deps=%s",
            subtask.step, action.value, subtask.shot_index, subtask.depends_on,
        )

        # Pick the handler
        if action == _EA.SPEECH_TTS:
            handler = self._run_step_speech_tts
            improver = self._improve_speaker_description_for_step
        elif action == _EA.SPEECH_SWAP:
            handler = self._run_step_speech_swap
            improver = self._improve_voice_design_for_step
        elif action == _EA.SPEECH_LIPSYNC:
            handler = self._run_step_speech_lipsync
            improver = self._improve_video_for_step
        elif action.is_audio:
            handler = self._run_step_audio
            improver = self._improve_audio_for_step
        else:
            handler = self._run_step_video
            improver = self._improve_video_for_step

        current = subtask
        best_applied = False
        for attempt in range(max_retries + 1):
            real_attempt = attempt + attempt_offset
            label = f"[step {subtask.step} attempt {real_attempt + 1}/{max_retries + 1 + attempt_offset}]"
            try:
                passed, eval_info = await handler(current, context, real_attempt)
            except Exception as exc:
                logger.warning("%s handler raised: %s", label, exc)
                passed, eval_info = False, {"reason": str(exc)}

            if passed:
                logger.info("%s PASS", label)
                best_applied = True
                break

            # Infra failure: if the evaluator itself blew up (API outage,
            # malformed LLM response, etc.), DO NOT treat this
            # as a content failure and burn another expensive retry on
            # the generator — accept the step's output as-is.
            info = eval_info or {}
            reason = str(info.get("reason", ""))
            if reason.startswith("Evaluation error:") or reason == "eval skipped" \
                    or reason == "audio eval skipped":
                logger.warning(
                    "%s accepted: evaluator infra failure (%s) — "
                    "not counting as content failure.",
                    label, reason[:120],
                )
                best_applied = True
                break

            # Soft-accept rule: audio_add_ambient is meant to be
            # SUBTLE. If the evaluator's only complaint is that the
            # generated ambient is quiet but some content was generated
            # (instruction_score > 0), accept it — retrying risks
            # producing TOO LOUD ambient that overpowers dialogue.
            # Drowned ambient is acceptable for this action type.
            if (
                action == _EA.AUDIO_ADD_AMBIENT
                and info.get("gen_quiet")
                and info.get("instruction_score", 0.0) > 0.0
                and info.get("fidelity_score", 1.0) >= 0.7
            ):
                logger.info(
                    "%s accepted despite gen_quiet: ambient sounds are "
                    "supposed to be subtle; original dialogue/audio "
                    "preserved (fidelity=%.2f).",
                    label, info.get("fidelity_score", 0.0),
                )
                best_applied = True
                break

            logger.warning("%s FAIL: %s", label, info.get("reason", ""))
            if attempt == max_retries:
                break
            current = await improver(current, eval_info)

        # Fallback to best attempt for audio steps: if no attempt
        # PASSED but at least one attempt produced audio + a score,
        # Always promote the highest-scoring attempt to
        # context.edited_audio. As long as the step produced any
        # non-zero scoring output, prefer it over reverting to the
        # prior state. Per the 2026-04-28 requirement, keep the best
        # non-zero MMAudio result without an additional per-step gate.
        if not best_applied and action.is_audio:
            best = context.best_audio_per_step.get(subtask.step)
            if best is not None:
                best_score, best_path, best_reason = best
                if (
                    action in (_EA.AUDIO_ADD_SFX, _EA.AUDIO_REPLACE_SFX)
                    and _reason_has_music_hallucination(best_reason)
                ):
                    logger.warning(
                        "[Runner] step %d: all attempts FAIL — refusing "
                        "best-attempt fallback for non-music SFX because "
                        "the best attempt mentions music/hallucinated "
                        "content (score=%.2f) | %s",
                        subtask.step, best_score, (best_reason or "")[:160],
                    )
                elif best_score > 0.0:
                    logger.warning(
                        "[Runner] step %d: all attempts FAIL — falling back "
                        "to best attempt (score=%.2f, audio=%s) | %s",
                        subtask.step, best_score, best_path.name,
                        (best_reason or "")[:120],
                    )
                    context.edited_audio = best_path
                    best_applied = True   # mark as applied for downstream
                else:
                    logger.warning(
                        "[Runner] step %d: all attempts FAIL — best score "
                        "is 0; leaving context.edited_audio untouched "
                        "(skipping this audio step entirely) | %s",
                        subtask.step, (best_reason or "")[:120],
                    )

        # Same best-attempt fallback for video edit steps. Per user
        # spec 2026-05-04: when no attempt PASSes, commit the
        # highest-scoring attempt instead of dropping the step.
        # Without this, the LAST attempt's output is committed
        # (because `_run_step_video` overwrites context.shot_videos
        # / current_global_video after every attempt) — but the last
        # attempt is often worse than an earlier one (the improver
        # may have regressed in trying to fix the previous failure).
        if action.is_video and not action.is_speech:
            best = context.best_video_per_step.get(subtask.step)
            if best is not None:
                best_score, best_path, best_shot_idx, best_info = best
                # Replace the committed output with the best-scoring
                # attempt's output if it isn't already there.
                if best_shot_idx is not None:
                    cur = context.shot_videos.get(best_shot_idx)
                    if cur != best_path:
                        logger.info(
                            "[Runner] step %d: promoting best-scoring video "
                            "attempt (score=%.2f, shot=%d, %s)",
                            subtask.step, best_score, best_shot_idx,
                            best_path.name,
                        )
                        context.shot_videos[best_shot_idx] = best_path
                else:
                    if context.current_global_video != best_path:
                        logger.info(
                            "[Runner] step %d: promoting best-scoring global "
                            "video attempt (score=%.2f, %s)",
                            subtask.step, best_score, best_path.name,
                        )
                        context.current_global_video = best_path
                if not best_applied:
                    # Mark applied even if no attempt PASSed — we still
                    # have a usable output (best of the failed bunch).
                    best_applied = True

        context.completed[subtask.step] = {
            "action": action.value,
            "shot_index": subtask.shot_index,
            "applied": best_applied,
        }
        # Remember the most recent successfully-applied audio subtask so
        # the mix evaluator can re-run it on needs_regenerate.
        if best_applied and action.is_audio:
            context.last_audio_subtask = subtask

    # ── per-step improvers (thin wrappers over existing helpers) ─────

    async def _improve_video_for_step(self, subtask, eval_info: dict):
        reason = (eval_info or {}).get("reason", "")
        new_desc = await self._improve_video_prompt(
            subtask.video_prompt, reason, subtask.action.value,
        )
        # Return a fresh SubTask with the improved video_prompt; all
        # other fields are copied verbatim. Audio / speech fields are
        # left at their defaults (empty), which is correct for video
        # actions.
        from av_editor.schema import SubTask as _ST
        return _ST(
            step=subtask.step, action=subtask.action,
            target=subtask.target, shot_index=subtask.shot_index,
            depends_on=subtask.depends_on,
            eval_criteria=list(subtask.eval_criteria),
            video_prompt=new_desc,
        )

    async def _improve_audio_for_step(self, subtask, eval_info: dict):
        """No-op (V2).

        Audio retries now happen INSIDE the branch handlers via
        `_retry_op` (SAM and MMAudio each have their own retry+improve
        loop with `_make_sam_improver` / `_make_mmaudio_improver`).
        The outer per-step retry budget for audio is `_MAX_RETRIES_AUDIO=0`,
        so this function is no longer reached. Kept as a stub so the
        dispatch wiring in `_run_one_subtask` (line ~1024) doesn't blow
        up if the budget ever changes.
        """
        return subtask

    async def _improve_sam_prompt_for_step(
        self, subtask, reason: str, param_key: str = "description",
    ):
        """No-op (V2). SAM retries are now handled inside `_retry_op`
        via `_make_sam_improver`. See `_improve_audio_for_step`."""
        return subtask

    async def _improve_voice_design_for_step(self, subtask, eval_info: dict):
        """speech_swap retry improver.

        Two distinct failure modes need DIFFERENT fixes:
          (a) SAM Audio failed to isolate the original speaker — the
              fix is to rewrite `original_speaker` (the separation
              prompt). The TARGET voice (voice_description) is
              completely unrelated and must be LEFT ALONE.
          (b) Voice Design produced an unsatisfactory voice (wrong
              gender / pitch / tone) — the fix is to rewrite
              `voice_description`.

        Route by reason-keyword; if ambiguous, default to voice
        description (most common path).
        """
        from av_editor.schema import SubTask as _ST

        reason = (eval_info or {}).get("reason", "") or ""
        reason_lc = reason.lower()
        sep_keywords = (
            "sam audio", "separate", "separation", "isolate", "isolation",
        )
        is_separation_failure = any(k in reason_lc for k in sep_keywords)

        from av_editor.core._gemini_client import gemini_with_fallback

        # Common: build a fresh SubTask copy with all speech_* fields
        # carried over; only one field is mutated below.
        def _clone(**overrides) -> _ST:
            base = dict(
                step=subtask.step, action=subtask.action,
                target=subtask.target, shot_index=subtask.shot_index,
                depends_on=list(subtask.depends_on),
                eval_criteria=list(subtask.eval_criteria),
                sam_eval_criteria=list(subtask.sam_eval_criteria),
                mmaudio_eval_criteria=list(subtask.mmaudio_eval_criteria),
                speech_text=subtask.speech_text,
                speech_speaker_description=subtask.speech_speaker_description,
                speech_voice_description=subtask.speech_voice_description,
                speech_reference_text=subtask.speech_reference_text,
                speech_language=subtask.speech_language,
            )
            base.update(overrides)
            return _ST(**base)

        if is_separation_failure:
            # Rewrite ONLY the SAM Audio prompt (speech_speaker_description).
            # Leave speech_voice_description untouched so we still
            # synthesise the user's target voice.
            current_sep = subtask.speech_speaker_description or ""
            system = (
                "You are writing a text prompt for a text-conditioned "
                "audio separation model (SAM Audio). The prompt should "
                "isolate ONE specific speaker's voice from a video. "
                "The current prompt failed to match anything — usually "
                "because it was too generic ('female voice') or carried "
                "language/accent fragments ('young adult female, "
                "American English') that SAM can't anchor on.\n\n"
                "FORMAT (strict): a SINGLE noun phrase of the shape "
                "`[adjective(s)] noun`, ≤ 8 words, with ONE noun head "
                "(use 'voice' or 'speech' as the head). NO comma, NO "
                "conjunction ('and' / 'or'), NO sub-clause, NO "
                "negation. Adjectives must be acoustic / demographic-"
                "via-acoustic: gender, age band, pitch, timbre.\n\n"
                "Examples (good): 'deep male voice', 'high female "
                "voice', 'elderly male voice', 'young female speech'.\n"
                "Examples (BAD): 'adult male voice, mid-pitch, "
                "American English' (3 fragments), 'the man speaking' "
                "(no acoustic adjective).\n\n"
                "Return ONLY the new prompt — no quotes, no "
                "explanation."
            )
            user = (
                f"Current separation prompt: \"{current_sep}\"\n"
                f"Evaluator feedback:\n{reason}\n\n"
                "Rewrite the separation prompt."
            )
            try:
                raw = await asyncio.to_thread(
                    gemini_with_fallback,
                    gemini_api_key=self.cfg.llm.gemini_api_key,
                    primary_model=self.cfg.llm.gemini_model,
                    fallback_model="gemini-2.5-flash",
                    system_prompt=system,
                    user_text=user,
                    json_response=False,
                    temperature=0.3,
                    max_output_tokens=9999,
                    component="SpeakerDescImprover(separation)",
                )
                new_sep = (raw or "").strip().strip('"')
                new_sep = _cap_words(new_sep, 12)
            except Exception as exc:
                logger.warning("[Runner] separation-prompt improve failed: %s", exc)
                new_sep = current_sep
            logger.info(
                "[Runner] speech_swap retry → rewrite speaker_description "
                "(leave voice_description=%r intact)",
                (subtask.speech_voice_description or "")[:60],
            )
            return _clone(speech_speaker_description=new_sep or current_sep)

        # Default path: the voice came out wrong → rewrite
        # speech_voice_description.
        current_voice = subtask.speech_voice_description or ""
        text = subtask.speech_text or ""
        system = (
            "You are a voice-casting prompt engineer for Qwen3 TTS "
            "Voice Design. Name gender, age, pitch, timbre, accent, "
            "speaking style — whatever disambiguates the TARGET voice "
            "the user wants. Keep under 25 words. Given the current "
            "description and evaluator feedback, rewrite it. "
            "Return ONLY the new description."
        )
        user = (
            f"Current voice_description: \"{current_voice}\"\n"
            f"Line to speak: \"{text}\"\n"
            f"Evaluator feedback:\n{reason}\n\n"
            "Rewrite the voice description."
        )
        try:
            raw = await asyncio.to_thread(
                gemini_with_fallback,
                gemini_api_key=self.cfg.llm.gemini_api_key,
                primary_model=self.cfg.llm.gemini_model,
                fallback_model="gemini-2.5-flash",
                system_prompt=system,
                user_text=user,
                json_response=False,
                temperature=0.3,
                max_output_tokens=9999,
                component="VoiceDesignImprover",
            )
            new_voice = (raw or "").strip().strip('"')
        except Exception as exc:
            logger.warning("[Runner] voice_design improve failed: %s", exc)
            new_voice = current_voice

        return _clone(speech_voice_description=new_voice or current_voice)

    async def _improve_speaker_description_for_step(self, subtask, eval_info: dict):
        """speech_tts retry improver — rewrite `speech_speaker_description`
        (the SAM Audio prompt for the original speaker)."""
        reason = (eval_info or {}).get("reason", "")
        new_desc = await self._improve_speaker_description(
            subtask.speech_speaker_description,
            subtask.speech_text,
            reason,
        )
        from av_editor.schema import SubTask as _ST
        return _ST(
            step=subtask.step, action=subtask.action,
            target=subtask.target, shot_index=subtask.shot_index,
            depends_on=list(subtask.depends_on),
            eval_criteria=list(subtask.eval_criteria),
            sam_eval_criteria=list(subtask.sam_eval_criteria),
            mmaudio_eval_criteria=list(subtask.mmaudio_eval_criteria),
            speech_text=subtask.speech_text,
            speech_speaker_description=new_desc or subtask.speech_speaker_description,
            speech_voice_description=subtask.speech_voice_description,
            speech_reference_text=subtask.speech_reference_text,
            speech_language=subtask.speech_language,
        )

    # ── handler: video edit (per-shot or global) ────────────────────

    async def _run_step_video(
        self, subtask, context: _StepContext, attempt: int,
    ) -> tuple[bool, dict]:
        """Execute a video edit step (style/scene/replace/etc). Updates
        context.shot_videos (per-shot) or context.current_global_video
        (global)."""
        from av_editor.core.shot_slicer import (
            slice_shot, slice_shot_padded, trim_to_duration,
        )
        from av_editor.schema import Shot as _Shot

        tools = self.registry.find_all(subtask.action.value)
        if not tools:
            return False, {"reason": f"No tool for {subtask.action.value}"}
        tool = tools[0]

        step_dir = context.session_dir / "execution" / f"step_{subtask.step:03d}" / f"attempt_{attempt + 1:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        shot_idx = subtask.shot_index

        if shot_idx is not None:
            # Per-shot: choose input (previously edited shot or fresh slice of base)
            shot = next((s for s in context.shots if s.index == shot_idx), None)
            if shot is None:
                return False, {"reason": f"shot {shot_idx} not in session"}
            src_input = context.shot_videos.get(shot_idx)
            if src_input is None:
                src_input_path = step_dir / f"src_shot_{shot_idx:03d}.mp4"
                slice_shot(
                    context.current_global_video or context.base_video,
                    shot, src_input_path,
                )
                src_input = src_input_path
            # Pad if shorter than V2V minimum
            padded = step_dir / f"padded_shot_{shot_idx:03d}.mp4"
            # Wrap src_input as a "whole-video shot" for the padded slicer
            shot_for_pad = _Shot(
                index=shot_idx, start=0.0, end=shot.duration,
                summary=shot.summary,
            )
            _, orig_dur = slice_shot_padded(
                src_input, shot_for_pad, padded,
                min_duration=V2V_MIN_DURATION_SEC,
            )
            tool_input = padded
        else:
            # Global: whole-video base
            src_input = context.current_global_video or context.base_video
            tool_input = src_input
            orig_dur = context.duration
            shot = None

        # New flat schema: video_prompt is the only field the V2V tool
        # needs. Strip any leaked shot terminology before forwarding.
        safe_prompt = _sanitize_tool_prompt(subtask.video_prompt)
        if safe_prompt != subtask.video_prompt:
            logger.info(
                "[Runner] step %d: sanitised shot terminology out of "
                "video_prompt before sending to tool", subtask.step,
            )
        # Pass `_description` to keep the V2V tool's prompt-builder
        # contract working — those tools read this key as the imperative
        # edit instruction.
        exec_params = {"_description": safe_prompt}
        try:
            result = await tool.execute(
                video_path=tool_input, action=subtask.action.value,
                params=exec_params, output_dir=step_dir,
            )
        except Exception as exc:
            return False, {"reason": f"tool error: {exc}"}
        if not result.success or not result.output_path or not result.output_path.exists():
            return False, {"reason": result.error_msg or "tool returned no output"}

        out_path = result.output_path
        if shot_idx is not None and orig_dur + 0.01 < V2V_MIN_DURATION_SEC:
            trimmed = step_dir / f"edited_shot_{shot_idx:03d}.mp4"
            try:
                trim_to_duration(out_path, orig_dur, trimmed)
                out_path = trimmed
            except Exception as exc:
                logger.warning("[Runner] trim failed: %s", exc)

        # Evaluate (VLM before/after)
        try:
            eval_result = await self.evaluator.evaluate(
                subtask=subtask,
                before_video=src_input,
                after_video=out_path,
                workspace=step_dir,
            )
            from av_editor.schema import EvalVerdict as _EV
            passed = eval_result.verdict == _EV.PASS
            info = {"reason": eval_result.reason,
                    "quality": eval_result.quality_score,
                    "consistency": eval_result.consistency_score}
        except Exception as exc:
            logger.warning("[Runner] video eval error: %s — accepting output", exc)
            passed, info = True, {"reason": "eval skipped"}

        # Track best-scoring attempt across retries. score = mean of
        # quality + consistency (both 0-1); a PASS attempt is always
        # at least as good as a FAIL attempt at the same numeric score
        # (so we slightly bump passing scores). Used by the runner to
        # commit the best output even when no attempt PASSes.
        quality = float(info.get("quality", 0.0) or 0.0)
        consistency = float(info.get("consistency", 0.0) or 0.0)
        score = (quality + consistency) / 2.0
        if passed:
            score += 0.001  # tiebreaker: prefer PASS over equal-scoring FAIL
        prior = context.best_video_per_step.get(subtask.step)
        if prior is None or score > prior[0]:
            context.best_video_per_step[subtask.step] = (
                score, out_path, shot_idx, info,
            )

        # Commit to context (even on fail we keep the attempt output so
        # downstream steps can see *something*; next attempt overwrites)
        if shot_idx is not None:
            context.shot_videos[shot_idx] = out_path
            context.any_per_shot_video_edit = True
        else:
            context.current_global_video = out_path
            context.any_global_video_edit = True
            # Invalidate per-shot edits since the global base changed.
            context.shot_videos.clear()
            context.any_per_shot_video_edit = False

        return passed, info

    # ── handler: generic audio (add/replace/remove SFX / BGM) ───────

    # ─────────────────────────────────────────────────────────────────
    # V2 audio pipeline — dispatch + per-action branch handlers
    #
    # Each branch (remove / replace / add) calls SAM and/or MMAudio via
    # `_retry_op`, which encapsulates the call → eval → improve →
    # passthrough-on-all-zero loop. Branch handlers don't reimplement
    # retry semantics; they only decide which tools to chain and how to
    # mix.
    #
    # Replace branch is the only one with an in-branch volume-retry
    # loop (per user spec — mix evaluation + adjustment lives in
    # replace). Every branch ends with one unified post-branch eval
    # via `AudioEvaluator.evaluate`.
    # ─────────────────────────────────────────────────────────────────

    # Per-tool retry budgets. Tools self-retry; the outer
    # `_run_one_subtask` loop is set to 0 for audio steps so the
    # branch isn't re-run end-to-end.
    _SAM_MAX_RETRIES = 2          # 3 attempts total
    _MMAUDIO_MAX_RETRIES = 2      # 3 attempts total
    _SAM_THRESHOLD = 0.6
    _GEN_THRESHOLD = 0.6
    _MIX_VOL_RETRIES = 2          # in-branch volume rebalance retries

    def _make_sam_evaluator(
        self, ae, target_description: str, expected_preserved: str,
        extra_eval_criteria: list[str], subtask, attempt: int,
        last_pair: dict[str, "Path | None"] | None = None,
    ):
        """Build a coroutine `(residual_path) -> (score, info)` for the
        unified 2-dim SAM evaluator.

        The unified evaluator listens to BOTH target and residual stems.
        `last_pair` is a closure dict {"target": Path, "residual": Path}
        the SAM call writes to before evaluate is invoked — this lets
        the evaluator see the target stem that `_run_audio_separation`
        otherwise stashes only on the side-channel
        `self._last_separation_target` attribute.
        """
        async def _eval(residual: Path) -> tuple[float, dict]:
            # Resolve target path: closure pair takes priority, otherwise
            # fall back to the side-channel attribute set by
            # `_run_audio_separation`.
            target = None
            if last_pair is not None:
                target = last_pair.get("target")
            if target is None:
                target = getattr(self, "_last_separation_target", None)

            if (
                target is None
                or not Path(target).exists()
                or residual is None
                or not Path(residual).exists()
            ):
                return 0.0, {
                    "reason": "SAM produced no usable target/residual stem",
                    "target_extraction": 0.0,
                    "residual_fidelity": 0.0,
                    "score": 0.0,
                }

            # Floor relaxes to 0.0 when the planner flagged the target
            # as not-prominent. Rationale: a faint / occasional source
            # (subtle BGM under speech, a quiet hum, a distant moo) can
            # legitimately come out of SAM as a near-empty target stem
            # — there's just not much there to extract. In that regime
            # the LLM evaluator's tex score is unreliable and shouldn't
            # gate the pipeline; we let combined ≥ threshold alone
            # decide pass/fail. When `expect_prominent_target=true` the
            # original 0.4 dual-floor still applies so a missed loud
            # source still triggers retry.
            sam_floor = 0.4 if bool(
                getattr(subtask, "expect_prominent_target", False)
            ) else 0.0
            passed, info = await ae.evaluate_separation(
                target_audio=Path(target),
                residual_audio=residual,
                target_description=target_description,
                expected_preserved=expected_preserved,
                extra_eval_criteria=extra_eval_criteria,
                threshold=self._SAM_THRESHOLD,
                floor=sam_floor,
                step=subtask.step,
                attempt=attempt,
                action=subtask.action.value,
            )
            # _retry_op uses score >= threshold for pass; we already
            # encode the dual-floor + combined threshold inside
            # `evaluate_separation`, so report combined score for
            # consistent telemetry. The dual-floor gate is reflected
            # in `passed` already, so even a high combined score that
            # violates a floor would NOT pass the underlying eval.
            score = info.get("score", 0.0)
            if not passed:
                # Force _retry_op to count this as failure regardless
                # of whether combined >= threshold (dual-floor block).
                score = min(score, self._SAM_THRESHOLD - 0.001)
            return score, info
        return _eval

    def _make_sam_improver(
        self,
        original_prompt: str = "",
        eval_criteria: list[str] | None = None,
    ):
        """Build `(prompt, info, history) -> new_prompt` that wraps the
        minimal-change SAM rewriter. The improver only rewrites the
        prompt string; the caller is responsible for plumbing it back
        into params (which `_retry_op` does internally — it just calls
        `call(new_prompt)` next round).

        `original_prompt` is the planner's first SAM input (sanitised)
        and `eval_criteria` is the SAM-stage `sam_eval_criteria` from
        Phase B. Both are surfaced verbatim to the LLM so each retry
        can compare its rewrite against the planner's intent and the
        success criteria, instead of drifting along with whatever the
        previous retry happened to emit.
        """
        # SAM Audio is positive-text-only — it has no negative_prompt
        # field on the fal API and the underlying model is trained
        # to MATCH the description, not to subtract sources from it.
        # All shape sanitisation goes through the module-level
        # `_sam_strip_negative` + `_sam_to_single_phrase` helpers,
        # which are also used by the branches at first-call time so
        # Phase B's prompt is normalised before SAM ever sees it.
        _strip_negative_clause = _sam_strip_negative
        _to_single_phrase = _sam_to_single_phrase
        anchor_prompt = (original_prompt or "").strip()
        criteria_list = list(eval_criteria or [])

        async def _improve(current: str, info: dict, history: list[str]) -> str:
            info = info or {}
            reason = info.get("reason", "")
            tex = info.get("target_extraction")
            fid = info.get("residual_fidelity")

            # Strategy: describe the SAME target more precisely under
            # a fixed shape. The improver compares the previous SAM
            # prompt against the planner's anchor prompt and the
            # SAM-stage eval criteria, then emits a new noun phrase
            # (≤ 8 words) that swaps a weak descriptor for a more
            # acoustically-anchored synonym. Output shape is HARD —
            # the underlying SAM model degrades quickly on multi-
            # fragment / clausal prompts.
            system = (
                "You are rewriting a SAM Audio separation prompt for "
                "the next retry. The previous attempt didn't isolate "
                "the target cleanly. Your job is to describe the SAME "
                "target more precisely so SAM can anchor on it.\n\n"
                "OUTPUT SHAPE (HARD CONSTRAINTS):\n"
                "  • A SINGLE noun phrase: `[adjective(s)] noun`. "
                "    Adjectives stack BEFORE one noun head.\n"
                "  • At most 8 words.\n"
                "  • NO comma. NO 'and' / 'or'. NO sub-clause "
                "    ('which is …'). NO negation tokens ('excluding "
                "    / without / no / not / except / avoid') — SAM "
                "    is positive-text-only.\n\n"
                "STRATEGY:\n"
                "  • Read the planner's ORIGINAL prompt and the SAM-"
                "    stage eval criteria. They tell you what target "
                "    you must isolate; do not pivot to a different "
                "    sound.\n"
                "  • Read the evaluator's reason and sub-scores. They "
                "    tell you what SAM heard vs what it should have "
                "    heard.\n"
                "  • Swap ONE weak adjective (or, if it's the wrong "
                "    word for what's in the audio, the noun head) for "
                "    a more acoustically-anchored synonym. Pick a "
                "    word that does NOT appear in the history.\n"
                "  • Keep complexity LOW — simple words, no piling on "
                "    new concepts (genre, mood, narrative).\n\n"
                "VOCABULARY (CRITICAL):\n"
                "  • Common everyday English. SAM was trained on "
                "    captions like 'a dog barking', 'people clapping', "
                "    'background music' — it does NOT anchor on "
                "    literary or genre adjectives.\n"
                "  • Avoid: melancholic / haunting / poignant / "
                "    cinematic / orchestral / atmospheric / ethereal / "
                "    dreamy / discordant / diegetic / sonorous, also "
                "    language ('English'), accent ('American'), "
                "    numerical pitch ('mid-pitch'), narrative "
                "    ('during the chorus').\n"
                "  • Prefer plain acoustic words:\n"
                "      voice:    deep / low / high / soft / loud /\n"
                "                male / female / child / talking\n"
                "      music:    background music / piano music /\n"
                "                string music / drum beat\n"
                "      texture:  dry / wet / metallic / wooden /\n"
                "                plastic / glass / hollow\n"
                "      impact:   tap / thud / clack / clunk /\n"
                "                knock / hit\n\n"
                "Return ONLY the new prompt — no quotes, no explanation."
            )
            history_text = (
                "\nPrompts already tried (DO NOT repeat):\n"
                + "\n".join(f"  - {p}" for p in history)
            ) if history else ""
            sub_str = (
                f"target_extraction={tex} residual_fidelity={fid}"
                if (tex is not None or fid is not None) else ""
            )
            anchor_block = (
                f"Planner's original target prompt: \"{anchor_prompt}\"\n"
                if anchor_prompt else ""
            )
            criteria_block = (
                "SAM-stage eval criteria (success conditions):\n"
                + "\n".join(f"  - {c}" for c in criteria_list)
                + "\n"
            ) if criteria_list else ""
            user = (
                f"{anchor_block}"
                f"{criteria_block}"
                f"Current SAM prompt: \"{current}\"\n"
                f"Evaluator sub-scores: {sub_str}\n"
                f"Evaluator feedback: {reason}\n"
                f"{history_text}\n"
                "Rewrite the SAM prompt as a single noun phrase "
                "(≤ 8 words) that describes the same target more "
                "precisely. No comma."
            )
            from av_editor.core._gemini_client import gemini_with_fallback

            async def _ask(temperature: float, extra_user: str = "") -> str:
                """SAM improver call through the official Gemini API."""
                try:
                    raw = await asyncio.to_thread(
                        gemini_with_fallback,
                        gemini_api_key=self.cfg.llm.gemini_api_key,
                        primary_model=self.cfg.llm.gemini_model,
                        fallback_model="gemini-2.5-flash",
                        system_prompt=system,
                        user_text=user + extra_user,
                        json_response=False,
                        temperature=temperature,
                        max_output_tokens=9999,
                        component="SAMImprover",
                    )
                except Exception as exc:
                    logger.warning("[SAM-improver] LLM call failed: %s", exc)
                    return ""
                raw = (raw or "").strip().strip('"').strip("'")
                return _cap_words(
                    _to_single_phrase(_strip_negative_clause(raw)), 8,
                )

            # Pass 1: standard temperature.
            cand = await _ask(0.3)
            if cand and cand not in history:
                return cand
            # Pass 2: LLM duplicated → escalate temperature + spell out
            # which words are blacklisted so it has to pick a different
            # adjective.
            blacklist = sorted({w for p in history for w in p.lower().split()})
            cand2 = await _ask(
                0.9,
                extra_user=(
                    f"\n\nYour previous answer matched a tried prompt "
                    f"(history). You MUST emit a DIFFERENT prompt. "
                    f"Words already tried (avoid as adjectives): "
                    f"{', '.join(blacklist)}. "
                    f"Pick a synonym you have NOT used."
                ),
            )
            if cand2 and cand2 not in history:
                return cand2
            # Last-resort: sanitise the current prompt. Caller's
            # `_retry_op` budget eventually exhausts.
            return _cap_words(
                _to_single_phrase(_strip_negative_clause(current)), 8,
            ) or current
        return _improve

    def _make_mmaudio_evaluator(self, ae, intent_text: str, subtask, attempt: int):
        async def _eval(gen_audio: Path) -> tuple[float, dict]:
            return await ae.evaluate_generation_intent(
                gen_audio, intent_text,
                step=subtask.step, attempt=attempt, action=subtask.action.value,
            )
        return _eval

    def _make_mmaudio_improver(self, current_negative: list[str]):
        """Build the chained MMAudio improver. A SINGLE LLM call inside
        `_improve_mmaudio_prompts` derives `missing`/`unwanted` from
        the evaluator feedback and uses the SAME analysis to emit both
        the rewritten positive prompt AND the new negative tokens —
        keeping the two updates coordinated rather than independent.
        This improver mutates the shared `current_negative` list so
        successive retries accumulate targeted negatives."""
        async def _improve(current_prompt: str, info: dict, history: list[str]) -> str:
            reason = (info or {}).get("reason", "")
            score = float((info or {}).get("score", 0.0) or 0.0)
            new_prompt, neg_to_add = await self._improve_mmaudio_prompts(
                current_positive=current_prompt,
                current_negative=current_negative,
                eval_reason=reason,
                eval_score=score,
                history=history,
            )
            for token in neg_to_add:
                if token and token not in current_negative:
                    current_negative.append(token)
            return new_prompt or current_prompt
        return _improve

    async def _post_branch_eval(
        self, subtask, context: _StepContext,
        edited_audio: Path, video_for_audio: Path, step_dir: Path,
    ) -> tuple[bool, dict]:
        """Mux edited_audio with video_for_audio and run the
        AudioEvaluator on the probe. Used as the unified post-branch
        verdict for every audio action."""
        from av_editor.core.audio_evaluator import AudioEvaluator
        from av_editor.core.postprocessor import merge_audio_video
        probe = step_dir / "probe_mux.mp4"
        try:
            merge_audio_video(video_for_audio, edited_audio, probe)
        except Exception as exc:
            logger.warning("[post-branch eval] mux failed: %s", exc)
            return True, {"reason": f"post-branch mux failed: {exc}"}

        try:
            ae = AudioEvaluator(llm_cfg=self.cfg.llm, session_dir=context.session_dir)
            r = await ae.evaluate(
                final_video=probe, audio_tasks=[subtask],
                original_audio_desc=_caption_audio_section(context.video_caption),
                has_original_audio=context.original_audio is not None,
            )
            logger.info(
                "[post-branch eval] step %d: overall=%.2f instr=%.2f "
                "sync=%.2f fid=%.2f gen_quiet=%s ori_quiet=%s contam=%s → %s",
                subtask.step, r.overall_score, r.instruction_score,
                r.sync_score, r.fidelity_score,
                r.generated_audio_too_quiet, r.original_audio_too_quiet,
                r.generated_audio_contamination,
                "PASS" if r.passed else "FAIL",
            )
            return r.passed, {
                "reason": r.reason,
                "instruction_score": r.instruction_score,
                "sync_score": r.sync_score,
                "fidelity_score": r.fidelity_score,
                "gen_quiet": r.generated_audio_too_quiet,
                "ori_quiet": r.original_audio_too_quiet,
                "contaminated": r.generated_audio_contamination,
                "score": r.overall_score,
            }
        except Exception as exc:
            logger.warning("[post-branch eval] error: %s — accepting", exc)
            return True, {"reason": f"audio eval skipped: {exc}"}

    # ── Branch: audio_remove ─────────────────────────────────────────
    async def _branch_audio_remove(
        self, subtask, context: _StepContext, attempt: int, step_dir: Path,
        video_for_audio: Path, current_audio: Path | None,
    ) -> tuple[bool, dict]:
        """Run SAM separation only. Output = residual (or passthrough).

        Reads new flat schema: `subtask.sam_prompt` (SAM input,
        typically `deleted_sound` plus 1-2 synonyms),
        `subtask.deleted_sound` (the inventory-level label, used in
        eval intent and post-branch logging), `expect_prominent_target`.
        """
        from av_editor.core.audio_evaluator import AudioEvaluator
        ae = AudioEvaluator(llm_cfg=self.cfg.llm, session_dir=context.session_dir)
        # Apply consecutive removals cumulatively. The original video still
        # provides visual guidance, while SAM receives the latest edited audio.
        sep_audio_path = current_audio or context.original_audio
        sam_video = context.original_video or video_for_audio
        expect_prominent = bool(subtask.expect_prominent_target)

        # Per-inner-try subdir so residual_*.wav from different SAM
        # retries don't pile up in the same flat directory. Each try
        # gets `step_dir/sam/try_NN/`.
        sam_root = step_dir / "sam"
        sam_try_count = {"n": 0}
        last_pair: dict[str, "Path | None"] = {"target": None, "residual": None}
        async def sam_call(p: str) -> Path | None:
            sam_try_count["n"] += 1
            try_dir = sam_root / f"try_{sam_try_count['n']:02d}"
            try_dir.mkdir(parents=True, exist_ok=True)
            residual = await self._run_audio_separation(
                sam_video, p, try_dir,
                audio_path=sep_audio_path,
                expect_prominent_target=expect_prominent,
            )
            last_pair["target"] = getattr(self, "_last_separation_target", None)
            last_pair["residual"] = residual
            return residual

        # Inputs to the unified evaluator: target description + what's
        # supposed to remain in residual + Phase B's task-specific
        # eval criteria.
        target_desc = subtask.deleted_sound or subtask.sam_prompt or ""
        preserved_str = ""
        inv = getattr(context, "audio_inventory", None)
        if inv and getattr(inv, "preserve", None):
            preserved_str = ", ".join(inv.preserve)
        extra_criteria = list(subtask.sam_eval_criteria or [])

        sam_initial = _sanitise_sam_prompt(
            subtask.sam_prompt or subtask.deleted_sound
        )
        op = await _retry_op(
            name=f"sam_remove[step{subtask.step}]",
            initial_prompt=sam_initial,
            call=sam_call,
            evaluate=self._make_sam_evaluator(
                ae, target_desc, preserved_str,
                extra_criteria, subtask, attempt, last_pair=last_pair,
            ),
            improve=self._make_sam_improver(
                original_prompt=sam_initial,
                eval_criteria=extra_criteria,
            ),
            fallback_input=current_audio,
            max_retries=self._SAM_MAX_RETRIES,
            threshold=self._SAM_THRESHOLD,
        )

        # Decide what `edited_audio` to commit. Three cases:
        #
        #   (a) SAM did NOT pass (`op.passed=False`) — either all
        #       attempts scored 0 and op fell back to passthrough, OR
        #       the best attempt was sub-threshold (e.g. 0.5/0.6). For
        #       a PURE audio_remove the user explicitly asked for the
        #       sound to be gone; returning a still-contaminated
        #       original / partial-residual is worse than returning
        #       silence. Generate a silent AAC.
        #
        #   (b) SAM passed with a WAV residual (PCM 16-bit, the fal
        #       SAM Audio output format) — re-encode to AAC so
        #       `context.edited_audio` is format-uniform with the rest
        #       of the pipeline (mix outputs, fallbacks, etc.).
        #
        #   (c) SAM passed and output is already AAC — use as-is.
        if op.output_path is None:
            edited = step_dir / "silent_fallback.aac"
            try:
                _generate_silent_aac(edited, duration=context.duration)
            except Exception as exc:
                return False, {
                    "reason": (
                        f"audio_remove: SAM produced no output and silent "
                        f"fallback failed: {exc}"
                    ),
                }
            logger.info(
                "[Runner] step %d: audio_remove SAM produced no usable "
                "output — silent fallback.",
                subtask.step,
            )
        else:
            if not op.passed:
                logger.info(
                    "[Runner] step %d: audio_remove SAM sub-threshold "
                    "(passthrough=%s, score=%.2f) — committing best-scoring "
                    "residual instead of silence.",
                    subtask.step, op.passthrough, op.score,
                )
            if Path(op.output_path).suffix.lower() == ".wav":
                edited = step_dir / "edited_audio.aac"
                try:
                    _wav_to_aac(op.output_path, edited)
                except Exception as exc:
                    logger.warning(
                        "[Runner] step %d: WAV→AAC conversion failed: %s — "
                        "using raw WAV (final mux re-encodes anyway)",
                        subtask.step, exc,
                    )
                    edited = op.output_path
            else:
                edited = op.output_path

        context.edited_audio = edited
        # Track for cross-step fallback as well.
        prior = context.best_audio_per_step.get(subtask.step)
        if prior is None or op.score > prior[0]:
            context.best_audio_per_step[subtask.step] = (
                op.score, edited, op.info.get("reason", "")
            )

        # Unified post-branch eval. For audio_remove the output is just
        # the residual; the evaluator will judge whether the target is
        # really gone and preserve content survives. probe_mux.mp4
        # lands in step_dir/eval/ so the flat root stays clean.
        eval_dir = step_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        passed, info = await self._post_branch_eval(
            subtask, context, edited, video_for_audio, eval_dir,
        )
        info.setdefault("score", op.score)
        info.setdefault("passthrough", op.passthrough)
        return passed, info

    # ── Branch: audio_replace_{sfx,bgm} ──────────────────────────────
    async def _branch_audio_replace(
        self, subtask, context: _StepContext, attempt: int, step_dir: Path,
        video_for_audio: Path, current_audio: Path | None,
    ) -> tuple[bool, dict]:
        """Three-stage chain: SAM (extract original_sound) → MMAudio
        (synthesise change_to) → mix. Mix evaluation + volume retry
        live in this branch only."""
        from av_editor.core.audio_evaluator import AudioEvaluator
        from av_editor.core.postprocessor import mix_audio_tracks, merge_audio_video
        from av_editor.schema import EditAction as _EA

        ae = AudioEvaluator(llm_cfg=self.cfg.llm, session_dir=context.session_dir)
        sep_audio_path = context.original_audio
        sam_video = context.original_video or video_for_audio
        expect_prominent = bool(subtask.expect_prominent_target)

        # ── Stage 1: SAM separate the deleted_sound ──────────────────
        # `sam_prompt` is the SAM tool input (deleted_sound + optional
        # synonyms by Phase B). `deleted_sound` is the inventory label
        # used in the eval intent.
        residual: Path | None = current_audio
        if subtask.sam_prompt:
            sam_root = step_dir / "sam"
            sam_try_count = {"n": 0}
            last_pair: dict[str, "Path | None"] = {"target": None, "residual": None}
            async def sam_call(p: str) -> Path | None:
                sam_try_count["n"] += 1
                try_dir = sam_root / f"try_{sam_try_count['n']:02d}"
                try_dir.mkdir(parents=True, exist_ok=True)
                r = await self._run_audio_separation(
                    sam_video, p, try_dir,
                    audio_path=sep_audio_path,
                    expect_prominent_target=expect_prominent,
                )
                last_pair["target"] = getattr(self, "_last_separation_target", None)
                last_pair["residual"] = r
                return r
            target_desc = subtask.deleted_sound or subtask.sam_prompt
            preserved_str = ""
            inv = getattr(context, "audio_inventory", None)
            if inv and getattr(inv, "preserve", None):
                preserved_str = ", ".join(inv.preserve)
            extra_criteria = list(subtask.sam_eval_criteria or [])

            sam_initial = _sanitise_sam_prompt(subtask.sam_prompt)
            sep_op = await _retry_op(
                name=f"sam_replace[step{subtask.step}]",
                initial_prompt=sam_initial,
                call=sam_call,
                evaluate=self._make_sam_evaluator(
                    ae, target_desc, preserved_str,
                    extra_criteria, subtask, attempt,
                    last_pair=last_pair,
                ),
                improve=self._make_sam_improver(
                    original_prompt=sam_initial,
                    eval_criteria=extra_criteria,
                ),
                fallback_input=current_audio,
                max_retries=self._SAM_MAX_RETRIES,
                threshold=self._SAM_THRESHOLD,
            )
            residual = sep_op.output_path
            # When the planner flagged the deleted_sound as not-prominent
            # (faint / occasional), and SAM didn't actually pass, prefer
            # the ORIGINAL audio over the best sub-threshold residual.
            # Rationale: a non-prominent source is likely barely there
            # to begin with; a half-baked separation often LOSES nearby
            # speech / ambient (residual_fidelity hit) without genuinely
            # removing the target. Just keeping the original and letting
            # MMAudio overlay the new sound on top usually produces a
            # better mix than a damaged residual.
            if (
                not sep_op.passed
                and current_audio is not None
                and not bool(getattr(subtask, "expect_prominent_target", False))
            ):
                logger.info(
                    "[Pipeline] step %d (audio_replace): SAM did not pass "
                    "and expect_prominent_target=false — falling back to "
                    "original audio instead of best sub-threshold residual.",
                    subtask.step,
                )
                residual = current_audio
            # NOTE: previously we tracked the SAM-stage residual into
            # `best_audio_per_step` at sep_score. That cross-stage
            # comparison was unsound — SAM's 1.0 means "separation
            # perfect", which is on a different axis from a post-branch
            # mix score (which measures full-task completion). The
            # outer fallback would then pick a residual-only output
            # over a sub-threshold mix that DID contain the new sound,
            # silently dropping the replace's MMAudio contribution.
            #
            # Replace branch instead writes only the FINAL post-branch
            # mix score into best_audio_per_step (below). Residual on
            # its own is reserved for audio_remove semantics, not
            # audio_replace.

        # ── Stage 2: MMAudio generate the new sound ──────────────────
        prompt = _sanitize_tool_prompt(subtask.mmaudio_prompt)
        # Negative prompt = preserve list, derived deterministically.
        # Improver may append eval-flagged unwanted items across retries.
        preserve = [
            s for s in subtask.existing_sounds
            if not _intent_overlaps_any(s, [subtask.deleted_sound])
        ] if subtask.deleted_sound else list(subtask.existing_sounds)
        cur_negs: list[str] = [s.lower().strip() for s in preserve if s.strip()]

        # Per-inner-try subdir for MMAudio (mirrors SAM layout above).
        # Each try gets `step_dir/mmaudio/try_NN/`. The underlying
        # tool writes the mmaudio_*.wav into try_dir/audio_gen/.
        # BGM has no visual sync requirement — the music plays
        # OVER the picture, not in response to it. Feeding video to
        # MMAudio for replace_bgm strongly biases the model toward
        # whatever the picture suggests (talking heads → whispering,
        # crowd shot → cheering, etc.) and overrides the user's BGM
        # description. SFX replace, by contrast, NEEDS the visual
        # sync (impacts must align with motion).
        from av_editor.schema import EditAction as _EA
        is_bgm_replace = (subtask.action == _EA.AUDIO_REPLACE_BGM)
        if subtask.action == _EA.AUDIO_REPLACE_SFX:
            for neg in _NON_MUSIC_SFX_NEGATIVES:
                if neg not in cur_negs:
                    cur_negs.append(neg)
        mm_root = step_dir / "mmaudio"
        mm_try_count = {"n": 0}
        async def mm_call(p: str) -> Path | None:
            mm_try_count["n"] += 1
            try_dir = mm_root / f"try_{mm_try_count['n']:02d}"
            try_dir.mkdir(parents=True, exist_ok=True)
            if subtask.action == _EA.AUDIO_REPLACE_SFX:
                neg = _with_non_music_sfx_negatives(cur_negs)
            else:
                neg = _cap_negative_list(", ".join(cur_negs), max_items=8)
            return await self._run_mmaudio_generation(
                video_for_audio, p, try_dir, context.duration,
                negative_prompt=neg,
                mask_away_clip=is_bgm_replace,
                guidance_scale=4.5,
            )
        gen_intent = (
            f"User intent: produce \"{prompt}\".\n"
            f"Forbidden sounds (must NOT appear): "
            f"\"{', '.join(cur_negs) or '(none)'}\"."
            + _inventory_annotation(context)
            + _mm_criteria_annotation(subtask)
        )
        mm_op = await _retry_op(
            name=f"mmaudio_replace[step{subtask.step}]",
            initial_prompt=prompt,
            call=mm_call,
            evaluate=self._make_mmaudio_evaluator(ae, gen_intent, subtask, attempt),
            improve=self._make_mmaudio_improver(cur_negs),
            fallback_input=None,   # raw gen has no meaningful passthrough
            max_retries=self._MMAUDIO_MAX_RETRIES,
            threshold=self._GEN_THRESHOLD,
        )

        # If MMAudio returned nothing usable, fall back to residual-only
        # (effectively a remove without replace) and let post-branch
        # eval flag it.
        if mm_op.output_path is None:
            edited_audio = residual or current_audio
            if edited_audio is None:
                return False, {"reason": "audio_replace: no usable audio after SAM+MMAudio"}
            context.edited_audio = edited_audio
            eval_dir = step_dir / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            passed, info = await self._post_branch_eval(
                subtask, context, edited_audio, video_for_audio, eval_dir,
            )
            info["mmaudio_failed"] = True
            return passed, info

        gen_audio = mm_op.output_path
        context.last_generated_audio = gen_audio

        # ── Stage 3: Mix + in-branch volume retry ────────────────────
        # Initial volumes: replace mixes the new SOURCE sound at full
        # level (1.0) so it actually replaces the original perceptually.
        # Earlier default 0.8 left the new sound under-pronounced when
        # the residual still contained trace clattering at full volume,
        # making the post-branch eval mistakenly score "audio
        # unchanged". When `expect_prominent_target=True` the planner
        # explicitly flagged the deleted sound as primary; the
        # replacement deserves the same prominence.
        # Each mix attempt (initial + each volume retry) writes into
        # its own subdir `step_dir/mix/try_NN/{mixed_audio.aac, probe_mux.mp4}`
        # so we can listen back to every volume permutation, not just
        # the last one.
        orig_vol = 1.0
        gen_vol = 1.1 if subtask.expect_prominent_target else 1.0
        mix_root = step_dir / "mix"

        def _mix_dir(idx: int) -> Path:
            d = mix_root / f"try_{idx:02d}"
            d.mkdir(parents=True, exist_ok=True)
            return d

        def _do_mix(o: float, g: float, mix_idx: int) -> Path:
            d = _mix_dir(mix_idx)
            out = d / "mixed_audio.aac"
            if residual is not None:
                mix_audio_tracks(
                    original_audio=residual, generated_audio=gen_audio,
                    output_path=out,
                    original_volume=o, generated_volume=g,
                    # Keep a quiet preserved bed (BGM/ambience) audible
                    # instead of letting a loud generated SFX mask it.
                    auto_balance_preserved=True,
                )
                return out
            return gen_audio   # no preserved track to mix in

        mix_idx = 1
        try:
            edited_audio = _do_mix(orig_vol, gen_vol, mix_idx)
        except Exception as exc:
            logger.warning("[Runner] audio mix failed: %s — using gen only", exc)
            edited_audio = gen_audio
        context.edited_audio = edited_audio
        context.last_gen_vol = gen_vol

        # Volume-retry loop: if post-branch eval flags volume imbalance
        # but content itself is OK, rebalance and re-mix (≤ _MIX_VOL_RETRIES).
        # Each post-branch eval writes its probe into the same mix/try_NN/
        # subdir as the audio it was evaluating, so files travel together.
        passed, info = await self._post_branch_eval(
            subtask, context, edited_audio, video_for_audio,
            _mix_dir(mix_idx),
        )
        for vol_attempt in range(self._MIX_VOL_RETRIES):
            if passed:
                break
            # Only loop on volume issues with non-trivial content score.
            instr = info.get("instruction_score", 0.0) or 0.0
            if instr < 0.5:
                break  # content broken — remix won't help
            gen_q = info.get("gen_quiet", False)
            ori_q = info.get("ori_quiet", False)
            if not (gen_q or ori_q):
                break
            # Volume policy: original is the reference baseline and
            # never amplifies past 1.0 — if the original feels too
            # quiet relative to the generated layer, attenuate the
            # generated layer instead. Generated can scale up to 1.5
            # when itself too quiet.
            if gen_q:
                gen_vol = min(1.5, gen_vol + 0.3)
            if ori_q:
                # Bring orig closer to 1.0; if already at 1.0, attenuate
                # gen instead so the speech is relatively louder.
                if orig_vol < 1.0:
                    orig_vol = min(1.0, orig_vol + 0.3)
                else:
                    gen_vol = max(0.05, gen_vol - 0.3)
            mix_idx += 1
            logger.info(
                "[Runner] step %d: in-branch volume retry %d/%d "
                "(orig=%.2f gen=%.2f) → mix/try_%02d",
                subtask.step, vol_attempt + 1, self._MIX_VOL_RETRIES,
                orig_vol, gen_vol, mix_idx,
            )
            try:
                edited_audio = _do_mix(orig_vol, gen_vol, mix_idx)
                context.edited_audio = edited_audio
                context.last_gen_vol = gen_vol
            except Exception as exc:
                logger.warning("[Runner] remix failed: %s", exc)
                break
            passed, info = await self._post_branch_eval(
                subtask, context, edited_audio, video_for_audio,
                _mix_dir(mix_idx),
            )

        # Cross-step fallback tracking with the final post-branch score.
        score = info.get("score", info.get("instruction_score", 0.0))
        prior = context.best_audio_per_step.get(subtask.step)
        if prior is None or score > prior[0]:
            context.best_audio_per_step[subtask.step] = (
                score, edited_audio, info.get("reason", "")
            )

        return passed, info

    # ── Branch: audio_add_{sfx,ambient} ──────────────────────────────
    async def _branch_audio_add(
        self, subtask, context: _StepContext, attempt: int, step_dir: Path,
        video_for_audio: Path, current_audio: Path | None,
    ) -> tuple[bool, dict]:
        """MMAudio-only chain: synthesise new sound and layer on top of
        the current audio track. No volume retry (per user spec — that's
        replace-only)."""
        from av_editor.core.audio_evaluator import AudioEvaluator
        from av_editor.core.postprocessor import mix_audio_tracks
        from av_editor.schema import EditAction as _EA

        ae = AudioEvaluator(llm_cfg=self.cfg.llm, session_dir=context.session_dir)
        prompt = _sanitize_tool_prompt(subtask.mmaudio_prompt)

        # Pure-add: nothing is being deleted, so the preserve list IS
        # the entire existing_sounds inventory. MMAudio must avoid
        # regenerating any of them.
        cur_negs: list[str] = [
            s.lower().strip() for s in subtask.existing_sounds if s.strip()
        ]
        if subtask.action == _EA.AUDIO_ADD_SFX:
            for neg in _NON_MUSIC_SFX_NEGATIVES:
                if neg not in cur_negs:
                    cur_negs.append(neg)

        mm_root = step_dir / "mmaudio"
        mm_try_count = {"n": 0}
        async def mm_call(p: str) -> Path | None:
            mm_try_count["n"] += 1
            try_dir = mm_root / f"try_{mm_try_count['n']:02d}"
            try_dir.mkdir(parents=True, exist_ok=True)
            if subtask.action == _EA.AUDIO_ADD_SFX:
                neg = _with_non_music_sfx_negatives(cur_negs)
            else:
                neg = _cap_negative_list(", ".join(cur_negs), max_items=8)
            return await self._run_mmaudio_generation(
                video_for_audio, p, try_dir, context.duration,
                negative_prompt=neg,
                mask_away_clip=(subtask.action == _EA.AUDIO_ADD_AMBIENT),
                guidance_scale=7.0,
            )
        gen_intent = (
            f"User intent: produce \"{prompt}\".\n"
            f"Forbidden sounds (must NOT appear): "
            f"\"{', '.join(cur_negs) or '(none)'}\"."
            + _inventory_annotation(context)
            + _mm_criteria_annotation(subtask)
        )
        mm_op = await _retry_op(
            name=f"mmaudio_add[step{subtask.step}]",
            initial_prompt=prompt,
            call=mm_call,
            evaluate=self._make_mmaudio_evaluator(ae, gen_intent, subtask, attempt),
            improve=self._make_mmaudio_improver(cur_negs),
            fallback_input=None,   # no passthrough for raw gen
            max_retries=self._MMAUDIO_MAX_RETRIES,
            threshold=self._GEN_THRESHOLD,
        )

        if mm_op.output_path is None:
            # Couldn't generate anything. Audio_add is purely additive,
            # so falling back to "no addition" = current_audio is safe.
            edited_audio = current_audio
            if edited_audio is None:
                return False, {"reason": "audio_add: MMAudio failed and no current audio"}
            context.edited_audio = edited_audio
            eval_dir = step_dir / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            passed, info = await self._post_branch_eval(
                subtask, context, edited_audio, video_for_audio, eval_dir,
            )
            info["mmaudio_failed"] = True
            return passed, info

        gen_audio = mm_op.output_path
        context.last_generated_audio = gen_audio

        # Compute mix volume. Ambient uses loudness-aware auto-match;
        # SFX uses a fixed 0.7 baseline.
        if subtask.action == _EA.AUDIO_ADD_AMBIENT and current_audio is not None:
            auto_vol, explain = _compute_ambient_gen_vol(
                original_audio=current_audio, generated_audio=gen_audio,
            )
            logger.info("[Runner] ambient loudness auto-match: %s", explain)
            gen_vol = auto_vol
        else:
            gen_vol = 0.7
        orig_vol = 1.0

        mix_dir = step_dir / "mix"
        mix_dir.mkdir(parents=True, exist_ok=True)
        edited_audio = mix_dir / "mixed_audio.aac"
        try:
            if current_audio is not None:
                mix_audio_tracks(
                    original_audio=current_audio, generated_audio=gen_audio,
                    output_path=edited_audio,
                    original_volume=orig_vol, generated_volume=gen_vol,
                )
            else:
                edited_audio = gen_audio
        except Exception as exc:
            logger.warning("[Runner] audio_add mix failed: %s", exc)
            edited_audio = gen_audio
        context.edited_audio = edited_audio
        context.last_gen_vol = gen_vol

        passed, info = await self._post_branch_eval(
            subtask, context, edited_audio, video_for_audio, mix_dir,
        )
        score = info.get("score", info.get("instruction_score", 0.0))
        prior = context.best_audio_per_step.get(subtask.step)
        if prior is None or score > prior[0]:
            context.best_audio_per_step[subtask.step] = (
                score, edited_audio, info.get("reason", "")
            )
        return passed, info

    # ── Branch: audio_volume_adjust ──────────────────────────────────
    async def _branch_audio_volume(
        self, subtask, context: _StepContext, attempt: int, step_dir: Path,
        video_for_audio: Path, current_audio: Path | None,
    ) -> tuple[bool, dict]:
        """Pure loudness adjust: SAM separates the target stem, then
        ffmpeg `volume=` (via `mix_audio_tracks`) re-mixes the boosted /
        attenuated stem back over the residual. NO generation.

        Reads new flat schema fields:
          - `volume_target` (Phase A): stem name (e.g. "human speech")
          - `volume_db`     (Phase A): signed gain delta in dB
          - `sam_prompt`    (Phase B): SAM input that isolates the stem

        On SAM passthrough (all attempts score 0): fall back to
        applying the volume change to the WHOLE current_audio. This
        preserves user intent direction even if the stem couldn't be
        isolated — the per-stem precision is sacrificed but the whole
        track shifts in the right direction. Post-branch eval still
        runs and can flag the contamination.
        """
        from av_editor.core.audio_evaluator import AudioEvaluator
        from av_editor.core.postprocessor import mix_audio_tracks

        ae = AudioEvaluator(llm_cfg=self.cfg.llm, session_dir=context.session_dir)
        sep_audio_path = context.original_audio
        sam_video = context.original_video or video_for_audio

        # Clamp dB to safe range and convert to linear gain.
        try:
            vdb = max(-12.0, min(12.0, float(subtask.volume_db or 0.0)))
        except (TypeError, ValueError):
            vdb = 0.0
        if vdb == 0.0:
            return False, {"reason": "audio_volume_adjust: volume_db is 0 (no-op)"}
        gain_factor = 10.0 ** (vdb / 20.0)
        direction = "boost" if vdb > 0 else "reduce"

        # ── Stage 1: SAM extract the stem to gain ────────────────────
        # Use the SAM `keep` mode-equivalent semantics: we need BOTH the
        # target (stem to gain) AND the residual (everything else),
        # then re-mix at adjusted ratio. `_run_audio_separation` runs
        # in remove mode but exposes both paths via
        # `_last_separation_target`, which is exactly what we need.
        sam_root = step_dir / "sam"
        sam_try_count = {"n": 0}
        last_pair: dict[str, "Path | None"] = {"target": None, "residual": None}
        last_target_holder: dict[str, Path | None] = {"path": None}

        async def sam_call(p: str) -> Path | None:
            sam_try_count["n"] += 1
            try_dir = sam_root / f"try_{sam_try_count['n']:02d}"
            try_dir.mkdir(parents=True, exist_ok=True)
            residual = await self._run_audio_separation(
                sam_video, p, try_dir,
                audio_path=sep_audio_path,
                expect_prominent_target=False,
            )
            tgt = getattr(self, "_last_separation_target", None)
            last_pair["target"] = tgt
            last_pair["residual"] = residual
            last_target_holder["path"] = tgt
            return residual

        target_desc = subtask.volume_target or subtask.sam_prompt
        # For volume_adjust, the entire `existing_sounds` minus the
        # volume_target is what should be cleanly preserved in residual.
        preserved_str = ""
        inv = getattr(context, "audio_inventory", None)
        if inv and getattr(inv, "preserve", None):
            preserved_str = ", ".join(
                p for p in inv.preserve
                if p.lower().strip() != (subtask.volume_target or "").lower().strip()
            )
        extra_criteria = list(subtask.sam_eval_criteria or [])

        sam_initial = _sanitise_sam_prompt(
            subtask.sam_prompt or subtask.volume_target
        )
        sam_op = await _retry_op(
            name=f"sam_volume[step{subtask.step}]",
            initial_prompt=sam_initial,
            call=sam_call,
            evaluate=self._make_sam_evaluator(
                ae, target_desc, preserved_str,
                extra_criteria, subtask, attempt,
                last_pair=last_pair,
            ),
            improve=self._make_sam_improver(
                original_prompt=sam_initial,
                eval_criteria=extra_criteria,
            ),
            fallback_input=current_audio,
            max_retries=self._SAM_MAX_RETRIES,
            threshold=self._SAM_THRESHOLD,
        )
        residual = sam_op.output_path
        target_stem = last_target_holder["path"]

        # ── Stage 2: re-mix with adjusted gain on the target stem ────
        mix_root = step_dir / "mix"
        mix_root.mkdir(parents=True, exist_ok=True)
        mix_dir = mix_root / "try_01"
        mix_dir.mkdir(parents=True, exist_ok=True)
        edited_audio = mix_dir / "mixed_audio.aac"

        if (
            not sam_op.passthrough
            and target_stem is not None
            and target_stem.exists()
            and residual is not None
            and residual.exists()
        ):
            # Stem-precise path: residual at unity + target at gain.
            mix_audio_tracks(
                original_audio=residual,
                generated_audio=target_stem,
                output_path=edited_audio,
                original_volume=1.0,
                generated_volume=gain_factor,
            )
            mode = "stem-precise"
        else:
            # Fallback: SAM couldn't isolate. Apply gain to whole
            # current_audio so the user's intent direction is at
            # least honoured. This is a known-degraded path; the
            # post-branch eval will likely flag side effects.
            if current_audio is None:
                return False, {
                    "reason": "audio_volume_adjust: no audio to adjust"
                }
            import subprocess as _sp
            cmd = [
                "ffmpeg", "-y", "-i", str(current_audio),
                "-filter:a", f"volume={gain_factor:.4f}",
                "-c:a", "aac", "-b:a", "192k",
                str(edited_audio),
            ]
            r = _sp.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                return False, {
                    "reason": f"audio_volume_adjust: ffmpeg volume failed: "
                              f"{r.stderr[:200]}"
                }
            mode = "whole-track-fallback"

        logger.info(
            "[volume_adjust] step %d: %s gain=%.3f (%+.1f dB) → %s [%s]",
            subtask.step, direction, gain_factor, vdb, edited_audio.name,
            mode,
        )

        context.edited_audio = edited_audio
        context.last_gen_vol = gain_factor

        # ── Stage 3: post-branch eval (probe_mux in mix_dir) ─────────
        passed, info = await self._post_branch_eval(
            subtask, context, edited_audio, video_for_audio, mix_dir,
        )
        info.setdefault("score", info.get("instruction_score", 0.0))
        info.setdefault("volume_adjust_mode", mode)
        info.setdefault("volume_adjust_db", vdb)
        score = info.get("score", 0.0)
        prior = context.best_audio_per_step.get(subtask.step)
        if prior is None or score > prior[0]:
            context.best_audio_per_step[subtask.step] = (
                score, edited_audio, info.get("reason", "")
            )
        return passed, info

    # ── Dispatcher ───────────────────────────────────────────────────
    async def _run_step_audio(
        self, subtask, context: _StepContext, attempt: int,
    ) -> tuple[bool, dict]:
        """Dispatch a global audio task to the right branch handler.
        Each branch handles its own SAM / MMAudio retry internally via
        `_retry_op`, so the outer `_run_one_subtask` retry budget for
        audio is set to 0."""
        from av_editor.schema import EditAction as _EA

        step_dir = (
            context.session_dir / "execution"
            / f"step_{subtask.step:03d}" / f"attempt_{attempt + 1:02d}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)

        action = subtask.action
        video_for_audio = self._current_video_snapshot(context)
        current_audio = context.edited_audio or context.original_audio

        if action == _EA.AUDIO_REMOVE:
            return await self._branch_audio_remove(
                subtask, context, attempt, step_dir,
                video_for_audio, current_audio,
            )
        if action in (_EA.AUDIO_REPLACE_SFX, _EA.AUDIO_REPLACE_BGM):
            return await self._branch_audio_replace(
                subtask, context, attempt, step_dir,
                video_for_audio, current_audio,
            )
        if action in (_EA.AUDIO_ADD_SFX, _EA.AUDIO_ADD_AMBIENT):
            return await self._branch_audio_add(
                subtask, context, attempt, step_dir,
                video_for_audio, current_audio,
            )
        if action == _EA.AUDIO_VOLUME_ADJUST:
            return await self._branch_audio_volume(
                subtask, context, attempt, step_dir,
                video_for_audio, current_audio,
            )
        return False, {"reason": f"unsupported audio action {action.value}"}

    # ── handler: speech_tts (global audio) ──────────────────────────

    async def _run_step_speech_tts(
        self, subtask, context: _StepContext, attempt: int,
    ) -> tuple[bool, dict]:
        from av_editor.core.audio_evaluator import AudioEvaluator
        from av_editor.core.postprocessor import (
            merge_audio_video,
            mix_audio_tracks,
            splice_replacement_audio,
        )
        import json as _json

        step_dir = context.session_dir / "execution" / f"step_{subtask.step:03d}" / f"attempt_{attempt + 1:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        new_text = (subtask.speech_text or "").strip()
        if not new_text:
            return False, {"reason": "speech_tts missing speech_text"}
        speaker_desc = (subtask.speech_speaker_description or "").strip() \
            or "human speech voice dialogue"
        # Sanitise to a single noun phrase before SAM sees it. Phase A
        # / improver may emit comma-separated descriptors like
        # "adult male voice, American English, mid-pitch" — those
        # dilute SAM's anchor (target_extraction → 0). Collapsing to
        # the longest single fragment recovers the noun head.
        speaker_desc = _sanitise_sam_prompt(speaker_desc, max_words=8) \
            or "human speech"
        reference_text = subtask.speech_reference_text
        language = subtask.speech_language or "auto"

        # Source for SAM visual separation: use the original video, because
        # the target voice/sound exists in the original audio-visual event.
        # Still pass original_audio explicitly so the tool can mux it with
        # that original visual track.
        video_for_sep = context.original_video or self._current_video_snapshot(context)

        # ── Stage 1: SAM separate the target speaker (with internal
        # retry via _retry_op so Qwen TTS only fires once we have a
        # clean reference). Mirrors the architecture used by the
        # audio_remove / audio_replace / audio_volume branches.
        ae_sep_tts = AudioEvaluator(
            llm_cfg=self.cfg.llm, session_dir=context.session_dir,
        )
        # Build expected_preserved list from inventory.
        preserved_str = ""
        inv = getattr(context, "audio_inventory", None)
        if inv and getattr(inv, "preserve", None):
            preserved_str = ", ".join(inv.preserve)

        sam_root = step_dir / "sam"
        sam_try_count = {"n": 0}
        last_pair: dict[str, Path | None] = {"target": None, "residual": None}

        async def sam_call_tracked(prompt: str) -> Path | None:
            sam_try_count["n"] += 1
            try_dir = sam_root / f"try_{sam_try_count['n']:02d}"
            try_dir.mkdir(parents=True, exist_ok=True)
            t, r = await self._separate_target_and_residual(
                video_for_sep, try_dir,
                speaker_description=prompt,
                audio_path=context.original_audio,
            )
            last_pair["target"] = t
            last_pair["residual"] = r
            return r

        # The evaluator's `target_description` is anchored to the
        # ORIGINAL planner-supplied speaker_desc. The improver may
        # rewrite the SAM prompt across retries, but the success
        # criterion ('did we isolate the speaker the planner asked
        # for?') stays constant — otherwise scores drift along with
        # the prompt and become non-comparable across attempts.
        sam_target_anchor = speaker_desc
        sam_extra_criteria = list(subtask.sam_eval_criteria or [])

        async def sam_evaluate(_residual_path: Path) -> tuple[float, dict]:
            target = last_pair.get("target")
            residual = last_pair.get("residual")
            if (
                target is None or not Path(target).exists()
                or residual is None or not Path(residual).exists()
            ):
                return 0.0, {
                    "reason": "SAM produced no usable target/residual stem",
                    "target_extraction": 0.0,
                    "residual_fidelity": 0.0,
                    "score": 0.0,
                }
            passed, info = await ae_sep_tts.evaluate_separation(
                target_audio=target,
                residual_audio=residual,
                target_description=sam_target_anchor,
                expected_preserved=preserved_str,
                extra_eval_criteria=sam_extra_criteria,
                threshold=self._SAM_THRESHOLD,
                floor=0.7,
                step=subtask.step,
                attempt=attempt,
                action=subtask.action.value,
            )
            score = info.get("score", 0.0)
            if not passed:
                score = min(score, self._SAM_THRESHOLD - 0.001)
            return score, info

        sam_op = await _retry_op(
            name=f"sam_speech_tts[step{subtask.step}]",
            initial_prompt=speaker_desc,
            call=sam_call_tracked,
            evaluate=sam_evaluate,
            improve=self._make_sam_improver(
                original_prompt=speaker_desc,
                eval_criteria=sam_extra_criteria,
            ),
            fallback_input=context.original_audio,
            max_retries=self._SAM_MAX_RETRIES,
            threshold=self._SAM_THRESHOLD,
        )

        speech_stem = last_pair.get("target")
        residual = sam_op.output_path if not sam_op.passthrough else None

        if not sam_op.passed or speech_stem is None:
            return False, {
                "reason": (
                    f"speech_tts SAM exhausted retries without capturing "
                    f"the target speaker {speaker_desc!r}. "
                    f"{sam_op.info.get('reason', '')}"
                ),
            }
        if residual is None:
            # SAM passthrough'd → no separation; skip the rest.
            return False, {
                "reason": (
                    f"speech_tts: SAM produced no usable residual for "
                    f"{speaker_desc!r}."
                ),
            }

        # Log soft-warnings if either dimension is mediocre but
        # passed dual-floor.
        sep_info = sam_op.info
        tex = sep_info.get("target_extraction", 1.0)
        fid = sep_info.get("residual_fidelity", 1.0)
        if tex < 0.7:
            logger.warning(
                "[Runner] step %d: speech_tts proceeding with mediocre "
                "target_extraction=%.2f (contaminants=%r).",
                subtask.step, tex,
                sep_info.get("target_contaminants", "")[:80],
            )
        if fid < 0.7:
            logger.warning(
                "[Runner] step %d: speech_tts proceeding with mediocre "
                "residual_fidelity=%.2f (audibility=%s, missing=%s).",
                subtask.step, fid,
                sep_info.get("target_audibility", "?"),
                sep_info.get("preserved_sounds_missing", "")[:80],
            )

        # ── Stage 2: Qwen TTS clone — runs ONCE with a verified clean
        # reference. Previously this fired on every outer retry even
        # when SAM had failed, wasting API calls.
        clone_ref = speech_stem
        cloned = await self._run_speech_clone(
            reference_audio=clone_ref, text=new_text, output_dir=step_dir,
            reference_text=reference_text, language=language,
        )
        if cloned is None:
            return False, {"reason": "Qwen TTS clone failed"}
        # Save cloned-only audio for downstream lipsync. Lipsync needs
        # a clean voice signal (no BGM) to drive lip animation.
        context.last_cloned_voice = cloned
        edited_audio = step_dir / "edited_audio.aac"
        preserved = residual
        splice_window = _resolve_speech_splice_window(subtask, context)
        splice_meta = {
            "mode": "localized_replace",
            "preserve_outside": True,
            "reference_text": reference_text,
            "resolved_window": None,
            "strategy": "fallback_global_mix",
        }
        if (
            preserved is not None
            and context.original_audio is not None
            and splice_window is not None
        ):
            start, end = splice_window
            try:
                splice_replacement_audio(
                    original_audio=context.original_audio,
                    replacement_bed_audio=preserved,
                    replacement_voice_audio=cloned,
                    output_path=edited_audio,
                    start=start,
                    end=end,
                    original_volume=1.0,
                    bed_volume=1.0,
                    voice_volume=1.0,
                    duration=context.duration,
                )
                splice_meta.update({
                    "resolved_window": {"start": start, "end": end},
                    "strategy": (
                        "original audio before/after window; SAM residual "
                        "+ cloned speech inside window"
                    ),
                })
                logger.info(
                    "[Runner] step %d: speech_tts local splice %.3f→%.3fs; "
                    "outside audio preserved from original.",
                    subtask.step, start, end,
                )
            except Exception as exc:
                logger.warning(
                    "[Runner] speech_tts local splice failed: %s — "
                    "falling back to global residual+clone mix", exc,
                )
                splice_meta["error"] = str(exc)
                try:
                    mix_audio_tracks(
                        original_audio=preserved, generated_audio=cloned,
                        output_path=edited_audio,
                        original_volume=1.0, generated_volume=1.0,
                    )
                except Exception as exc2:
                    logger.warning("[Runner] speech_tts mix failed: %s", exc2)
                    edited_audio = cloned
        elif preserved is not None:
            try:
                mix_audio_tracks(
                    original_audio=preserved, generated_audio=cloned,
                    output_path=edited_audio,
                    original_volume=1.0, generated_volume=1.0,
                )
            except Exception as exc:
                logger.warning("[Runner] speech_tts mix failed: %s", exc)
                edited_audio = cloned
        else:
            edited_audio = cloned
        try:
            (step_dir / "audio_splice.json").write_text(
                _json.dumps(splice_meta, ensure_ascii=False, indent=2)
            )
        except Exception:
            pass

        context.edited_audio = edited_audio
        context.speech_stem = speech_stem

        # Eval (mux onto current video just for probing)
        probe = step_dir / "probe_mux.mp4"
        merge_audio_video(video_for_sep, edited_audio, probe)
        try:
            orig_audio_desc = _caption_audio_section(context.video_caption)
            ae = AudioEvaluator(llm_cfg=self.cfg.llm, session_dir=context.session_dir)
            r = await ae.evaluate(
                final_video=probe, audio_tasks=[subtask],
                original_audio_desc=orig_audio_desc,
                has_original_audio=context.original_audio is not None,
            )
            return r.passed, {
                "reason": r.reason,
                "instruction_score": r.instruction_score,
                "sync_score": r.sync_score,
                "fidelity_score": r.fidelity_score,
                "gen_quiet": r.generated_audio_too_quiet,
                "ori_quiet": r.original_audio_too_quiet,
                "contaminated": r.generated_audio_contamination,
            }
        except Exception as exc:
            logger.warning("[Runner] speech_tts eval error: %s — accepting", exc)
            return True, {"reason": "eval skipped"}

    # ── handler: speech_swap (identity change via Qwen Voice Design) ─

    async def _run_step_speech_swap(
        self, subtask, context: _StepContext, attempt: int,
    ) -> tuple[bool, dict]:
        """Change the speaker IDENTITY (e.g. female → male). Uses
        Qwen3 Voice Design (natural-language voice description) rather
        than Voice Clone (which preserves the source timbre). Removes
        the original speaker via SAM Audio, synthesises the new line
        in the described voice, and mixes the two into edited_audio.
        """
        from av_editor.core.audio_evaluator import AudioEvaluator
        from av_editor.core.postprocessor import (
            merge_audio_video,
            mix_audio_tracks,
            splice_replacement_audio,
        )
        import json as _json

        step_dir = context.session_dir / "execution" / f"step_{subtask.step:03d}" / f"attempt_{attempt + 1:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        text = (subtask.speech_text or "").strip()
        voice_description = (subtask.speech_voice_description or "").strip()
        # speech_speaker_description is the SAM Audio prompt for the
        # ORIGINAL speaker being removed.
        original_speaker = (
            subtask.speech_speaker_description or ""
        ).strip() or "human speech voice dialogue"
        original_speaker = _sanitise_sam_prompt(
            original_speaker, max_words=8,
        ) or "human speech"
        language = subtask.speech_language or "auto"

        if not text:
            return False, {"reason": "speech_swap missing speech_text"}
        if not voice_description:
            return False, {
                "reason": "speech_swap missing speech_voice_description",
            }

        # 1. SAM separate the original speaker's voice with internal
        #    retry via _retry_op (mirrors speech_tts). Voice Design
        #    only fires once SAM passes the 3-dim check, saving
        #    expensive API calls on bad SAM outputs.
        video_for_sep = context.original_video or self._current_video_snapshot(context)

        ae_sep = AudioEvaluator(
            llm_cfg=self.cfg.llm, session_dir=context.session_dir,
        )
        preserved_str = ""
        inv = getattr(context, "audio_inventory", None)
        if inv and getattr(inv, "preserve", None):
            preserved_str = ", ".join(inv.preserve)

        sam_root = step_dir / "sam"
        sam_try_count = {"n": 0}
        last_pair: dict[str, Path | None] = {"target": None, "residual": None}

        async def sam_call_tracked(prompt: str) -> Path | None:
            sam_try_count["n"] += 1
            try_dir = sam_root / f"try_{sam_try_count['n']:02d}"
            try_dir.mkdir(parents=True, exist_ok=True)
            t, r = await self._separate_target_and_residual(
                video_for_sep, try_dir,
                speaker_description=prompt,
                audio_path=context.original_audio,
            )
            last_pair["target"] = t
            last_pair["residual"] = r
            return r

        # See speech_tts: anchor evaluator to the planner's original
        # speaker description so retry scores stay comparable.
        sam_target_anchor = original_speaker
        sam_extra_criteria = list(subtask.sam_eval_criteria or [])

        async def sam_evaluate(_residual_path: Path) -> tuple[float, dict]:
            target = last_pair.get("target")
            residual = last_pair.get("residual")
            if (
                target is None or not Path(target).exists()
                or residual is None or not Path(residual).exists()
            ):
                return 0.0, {
                    "reason": "SAM produced no usable target/residual stem",
                    "target_extraction": 0.0,
                    "residual_fidelity": 0.0,
                    "score": 0.0,
                }
            passed, info = await ae_sep.evaluate_separation(
                target_audio=target,
                residual_audio=residual,
                target_description=sam_target_anchor,
                expected_preserved=preserved_str,
                extra_eval_criteria=sam_extra_criteria,
                threshold=self._SAM_THRESHOLD,
                floor=0.7,
                step=subtask.step,
                attempt=attempt,
                action=subtask.action.value,
            )
            score = info.get("score", 0.0)
            if not passed:
                score = min(score, self._SAM_THRESHOLD - 0.001)
            return score, info

        sam_op = await _retry_op(
            name=f"sam_speech_swap[step{subtask.step}]",
            initial_prompt=original_speaker,
            call=sam_call_tracked,
            evaluate=sam_evaluate,
            improve=self._make_sam_improver(
                original_prompt=original_speaker,
                eval_criteria=sam_extra_criteria,
            ),
            fallback_input=context.original_audio,
            max_retries=self._SAM_MAX_RETRIES,
            threshold=self._SAM_THRESHOLD,
        )

        _target = last_pair.get("target")
        residual = sam_op.output_path if not sam_op.passthrough else None
        preserved = residual

        if not sam_op.passed or _target is None:
            return False, {
                "reason": (
                    f"speech_swap SAM exhausted retries without capturing "
                    f"the original speaker {original_speaker!r}. "
                    f"{sam_op.info.get('reason', '')}"
                ),
            }
        if preserved is None:
            return False, {
                "reason": (
                    f"speech_swap: SAM produced no usable residual for "
                    f"{original_speaker!r}."
                ),
            }

        sep_info = sam_op.info
        tex = sep_info.get("target_extraction", 1.0)
        fid = sep_info.get("residual_fidelity", 1.0)
        if tex < 0.7:
            logger.warning(
                "[Runner] step %d: speech_swap proceeding with mediocre "
                "target_extraction=%.2f.", subtask.step, tex,
            )
        if fid < 0.7:
            logger.warning(
                "[Runner] step %d: speech_swap proceeding with mediocre "
                "residual_fidelity=%.2f.", subtask.step, fid,
            )

        # 2. Generate the new voice via Voice Design API.
        design_tool = self.registry.find_speech_design_tool()
        if design_tool is None:
            return False, {"reason": "no Voice Design tool registered"}
        result = await design_tool.execute(
            video_path=Path(),
            action=subtask.action.value,
            params={
                "text": text,
                "voice_description": voice_description,
                "language": language,
            },
            output_dir=step_dir,
        )
        if not result.success or not result.output_path:
            return False, {
                "reason": f"Voice Design failed: {result.error_msg}",
            }
        new_voice = result.output_path
        # Save the clean designed voice for downstream lipsync (same
        # reasoning as speech_tts).
        context.last_cloned_voice = new_voice

        # 3. Build the complete edited audio by mixing residual + new voice.
        edited_audio = step_dir / "edited_audio.aac"
        splice_window = _resolve_speech_splice_window(subtask, context)
        splice_meta = {
            "mode": "localized_replace",
            "preserve_outside": True,
            "reference_text": subtask.speech_reference_text,
            "resolved_window": None,
            "strategy": "fallback_global_mix",
        }
        if (
            preserved is not None
            and context.original_audio is not None
            and splice_window is not None
        ):
            start, end = splice_window
            try:
                splice_replacement_audio(
                    original_audio=context.original_audio,
                    replacement_bed_audio=preserved,
                    replacement_voice_audio=new_voice,
                    output_path=edited_audio,
                    start=start,
                    end=end,
                    original_volume=1.0,
                    bed_volume=1.0,
                    voice_volume=1.0,
                    duration=context.duration,
                )
                splice_meta.update({
                    "resolved_window": {"start": start, "end": end},
                    "strategy": (
                        "original audio before/after window; SAM residual "
                        "+ designed speech inside window"
                    ),
                })
                logger.info(
                    "[Runner] step %d: speech_swap local splice %.3f→%.3fs; "
                    "outside audio preserved from original.",
                    subtask.step, start, end,
                )
            except Exception as exc:
                logger.warning(
                    "[Runner] speech_swap local splice failed: %s — "
                    "falling back to global residual+voice mix", exc,
                )
                splice_meta["error"] = str(exc)
                try:
                    mix_audio_tracks(
                        original_audio=preserved, generated_audio=new_voice,
                        output_path=edited_audio,
                        original_volume=1.0, generated_volume=1.0,
                    )
                except Exception as exc2:
                    logger.warning("[Runner] speech_swap mix failed: %s", exc2)
                    edited_audio = new_voice
        elif preserved is not None:
            try:
                mix_audio_tracks(
                    original_audio=preserved, generated_audio=new_voice,
                    output_path=edited_audio,
                    original_volume=1.0, generated_volume=1.0,
                )
            except Exception as exc:
                logger.warning("[Runner] speech_swap mix failed: %s", exc)
                edited_audio = new_voice
        else:
            edited_audio = new_voice
        try:
            (step_dir / "audio_splice.json").write_text(
                _json.dumps(splice_meta, ensure_ascii=False, indent=2)
            )
        except Exception:
            pass

        context.edited_audio = edited_audio

        # 4. Eval on a probe mux.
        probe = step_dir / "probe_mux.mp4"
        merge_audio_video(video_for_sep, edited_audio, probe)
        try:
            orig_audio_desc = _caption_audio_section(context.video_caption)
            ae = AudioEvaluator(llm_cfg=self.cfg.llm, session_dir=context.session_dir)
            r = await ae.evaluate(
                final_video=probe, audio_tasks=[subtask],
                original_audio_desc=orig_audio_desc,
                has_original_audio=context.original_audio is not None,
            )
            return r.passed, {
                "reason": r.reason,
                "instruction_score": r.instruction_score,
                "sync_score": r.sync_score,
                "fidelity_score": r.fidelity_score,
                "gen_quiet": r.generated_audio_too_quiet,
                "ori_quiet": r.original_audio_too_quiet,
                "contaminated": r.generated_audio_contamination,
            }
        except Exception as exc:
            logger.warning("[Runner] speech_swap eval error: %s — accepting", exc)
            return True, {"reason": "eval skipped"}

    # ── handler: speech_lipsync (per-shot video) ────────────────────

    async def _run_step_speech_lipsync(
        self, subtask, context: _StepContext, attempt: int,
    ) -> tuple[bool, dict]:
        from av_editor.core.shot_slicer import (
            slice_audio, slice_shot, slice_shot_padded, trim_to_duration,
        )
        from av_editor.schema import Shot as _Shot

        step_dir = context.session_dir / "execution" / f"step_{subtask.step:03d}" / f"attempt_{attempt + 1:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        shot_idx = subtask.shot_index
        if shot_idx is None:
            return False, {"reason": "speech_lipsync requires shot_index"}
        if context.edited_audio is None:
            return False, {"reason": "no edited_audio in context (speech_tts must run first)"}

        shot = next((s for s in context.shots if s.index == shot_idx), None)
        if shot is None:
            return False, {"reason": f"shot {shot_idx} not found"}

        # Audio fed to lipsync: PREFER the clean cloned voice (no BGM,
        # no residual mix). Sync Lipsync 2 drives lip animation off
        # the voice content of the input audio — a mix that has voice
        # only in the first 60% of the shot duration confuses the
        # model into producing weak / no animation. The final output
        # mux still uses `context.edited_audio` (the full mix), so
        # we're free to pass any audio here that helps the tool.
        #
        # We pad the cloned voice with trailing silence so its
        # duration matches the shot — model sees "voice → silence →
        # mouth closes", which is the natural physical interpretation.
        # Fall back to slicing `edited_audio` when the lipsync step
        # is downstream of an audio chain that didn't produce a
        # standalone cloned voice.
        audio_slice = step_dir / f"shot_{shot_idx:03d}_audio.wav"
        cloned = context.last_cloned_voice
        if cloned is not None and Path(cloned).exists():
            from av_editor.core.shot_slicer import _run as _ff_run
            shot_dur = max(0.05, shot.end - shot.start)
            _ff_run(
                [
                    "ffmpeg", "-y",
                    "-i", str(cloned),
                    "-af", f"apad,atrim=0:{shot_dur:.3f},asetpts=PTS-STARTPTS",
                    "-ar", "48000", "-ac", "1",
                    "-c:a", "pcm_s16le",
                    str(audio_slice),
                ],
                "lipsync_pad_cloned",
            )
            logger.info(
                "[Runner] step %d lipsync: padded cloned voice (%s) → %.2fs slice",
                subtask.step, Path(cloned).name, shot_dur,
            )
        else:
            slice_audio(
                context.edited_audio, shot.start, shot.end, audio_slice,
            )

        # Slice (possibly padded) the video for this shot from the current
        # per-shot or global state.
        src_input = context.shot_videos.get(shot_idx)
        if src_input is None:
            src_input = step_dir / f"src_shot_{shot_idx:03d}.mp4"
            slice_shot(
                context.current_global_video or context.base_video,
                shot, src_input,
            )
        padded = step_dir / f"padded_shot_{shot_idx:03d}.mp4"
        shot_for_pad = _Shot(index=shot_idx, start=0.0, end=shot.duration, summary=shot.summary)
        _, orig_dur = slice_shot_padded(
            src_input, shot_for_pad, padded,
            min_duration=V2V_MIN_DURATION_SEC,
        )

        ls_out = await self._run_lipsync(
            video_path=padded, audio_path=audio_slice,
            output_dir=step_dir / "lipsync", sync_mode="cut_off",
        )
        if ls_out is None:
            return False, {"reason": "lipsync failed"}

        # Trim back to original shot duration
        if orig_dur + 0.01 < V2V_MIN_DURATION_SEC:
            trimmed = step_dir / f"edited_shot_{shot_idx:03d}.mp4"
            try:
                trim_to_duration(ls_out, orig_dur, trimmed)
                ls_out = trimmed
            except Exception as exc:
                logger.warning("[Runner] lipsync trim failed: %s", exc)

        # Evaluate (VLM before/after on this shot)
        try:
            eval_result = await self.evaluator.evaluate(
                subtask=subtask, before_video=src_input,
                after_video=ls_out, workspace=step_dir,
            )
            from av_editor.schema import EvalVerdict as _EV
            passed = eval_result.verdict == _EV.PASS
            info = {"reason": eval_result.reason,
                    "quality": eval_result.quality_score,
                    "consistency": eval_result.consistency_score}
        except Exception as exc:
            logger.warning("[Runner] lipsync eval error: %s — accepting", exc)
            passed, info = True, {"reason": "eval skipped"}

        context.shot_videos[shot_idx] = ls_out
        context.any_per_shot_video_edit = True
        return passed, info

    # ── mid-pipeline snapshot: assemble current visual state ───────

    def _current_video_snapshot(self, context: _StepContext) -> Path:
        """Return a video reflecting the current visual edit state
        (per-shot edits concatenated with originals for untouched
        shots). Used by audio steps so MMAudio/AudioEval see the
        EDITED frames, not the untouched base video.

        Cheap no-op when no per-shot edits exist: returns
        current_global_video or base_video directly.
        """
        if not context.any_per_shot_video_edit or not context.shots:
            return context.current_global_video or context.base_video

        # Build a snapshot from the current shot_videos + fresh slices
        # of base_video for any untouched shot. Cache by a hash of the
        # current shot_videos mapping so we don't re-concat on every
        # audio step when nothing changed.
        mapping_key = tuple(
            (s.index, str(context.shot_videos.get(s.index, "")))
            for s in context.shots
        )
        cache = getattr(context, "_snapshot_cache", None) or {}
        if cache.get("key") == mapping_key and cache.get("path") and Path(cache["path"]).exists():
            return Path(cache["path"])

        from av_editor.core.shot_slicer import (
            concat_shots, slice_shot, _probe_fps, _probe_size,
        )
        snap_dir = context.session_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        pieces: list[Path] = []
        base = context.current_global_video or context.base_video
        for shot in context.shots:
            piece = context.shot_videos.get(shot.index)
            if piece is None:
                piece = snap_dir / f"src_shot_{shot.index:03d}.mp4"
                if not piece.exists():
                    slice_shot(base, shot, piece)
            pieces.append(piece)
        # Unique name per mapping so we can keep history
        stamp = f"snap_{hash(mapping_key) & 0xffffffff:08x}.mp4"
        out = snap_dir / stamp
        if not out.exists():
            src_fps = _probe_fps(context.base_video)
            src_size = _probe_size(context.base_video)
            concat_shots(
                pieces, out, reencode=True,
                target_fps=src_fps, target_size=src_size,
            )
        context._snapshot_cache = {"key": mapping_key, "path": str(out)}  # type: ignore
        logger.info("[Runner] current-video snapshot: %s", out.name)
        return out

    # ── final mixed-media evaluator + least-cost repair loop ────────

    _MAX_MIX_RETRIES = 2

    # At most one expensive regenerate cycle per mix-eval session — the
    # generation step costs ~60s+; volume remixes are essentially free.
    _MAX_REGEN_RETRIES = 1

    # Full replanning reruns every subtask from the original inputs, so keep
    # this budget deliberately small. The best result across all cycles wins.
    _MAX_FULL_REPLANS = 1

    async def _run_mix_eval_loop(
        self,
        context: _StepContext,
        video_part: Path,
        output_path: Path,
        inventory: "AudioInventory | None",
    ) -> Path:
        """Run MixEvaluator on the muxed output; remediate based on the
        result. Two remediation paths:

        - `volume_adjustment` set: re-mix the original + last generated
          stems with the suggested multipliers (cheap, ffmpeg-only). Up
          to `_MAX_MIX_RETRIES` rounds.
        - `needs_regenerate=True` AND we still have a regen budget AND
          the last audio subtask is known: re-run that subtask once,
          re-mux, re-evaluate. Up to `_MAX_REGEN_RETRIES` regen cycles.
        - `needs_replan=True`: stop local repair and expose the evaluator
          feedback to the outer pipeline, which may rebuild and rerun the
          complete plan from the original inputs.

        Returns the highest-scoring output across all attempts (initial
        output included).

        This evaluator also runs for video-only plans; an empty audio
        inventory simply disables the audio-specific local repair paths.
        """
        from av_editor.core.mix_evaluator import MixEvaluator
        from av_editor.core.postprocessor import (
            merge_audio_video, mix_audio_tracks,
        )

        evaluator = MixEvaluator(
            llm_cfg=self.cfg.llm, session_dir=context.session_dir,
        )
        best_output = output_path
        best_score = -1.0
        best_result = None
        candidate_path = output_path
        regen_budget = self._MAX_REGEN_RETRIES
        context.mix_eval_result = None
        context.replan_request = None

        for attempt in range(self._MAX_MIX_RETRIES + 1):
            result = await evaluator.evaluate(
                candidate_path,
                inventory,
                attempt=attempt,
                source_video=context.original_video,
                instruction=context.instruction,
                source_caption=context.video_caption,
                subtasks=context.subtasks,
            )
            # Track best across all attempts — including the initial
            # output and any later remix/regen candidate.
            if result.overall_score > best_score:
                best_score = result.overall_score
                best_output = candidate_path
                best_result = result
            if result.passed:
                context.mix_eval_result = best_result
                return best_output

            if result.needs_replan:
                if context.allow_full_replan:
                    context.replan_request = result
                    logger.warning(
                        "[MixEval] structural failure requires full replan: %s",
                        (result.replan_feedback or result.reason)[:300],
                    )
                else:
                    logger.warning(
                        "[MixEval] structural failure detected, but this run "
                        "mode does not allow full replanning: %s",
                        (result.replan_feedback or result.reason)[:300],
                    )
                break
            if attempt == self._MAX_MIX_RETRIES:
                break

            adj = result.volume_adjustment
            preserved = context.original_audio
            generated = context.last_generated_audio

            # Priority 1: REGEN when content is wrong / missing
            # (`needs_regenerate=True`). The evaluator says the audio
            # itself is incorrect (target sound absent, hallucinated
            # content, etc.) — only re-running the source step can fix
            # that. After the regen call, optionally apply MixEval's
            # `volume_adjustment` while remixing the new generated stem
            # against the preserved original — that lets one MixEval
            # round express both "wrong content" + "and the level was
            # off" without paying for a second iteration.
            if result.needs_regenerate:
                from av_editor.schema import EditAction as _EA
                # Actions that don't generate new audio content; re-running
                # them won't change the mix's content (audio_remove is
                # pure SAM separation, audio_volume_adjust is a level
                # shift).
                _NON_GENERATIVE = {
                    _EA.AUDIO_REMOVE, _EA.AUDIO_VOLUME_ADJUST,
                }
                target = context.last_audio_subtask
                regen_blocked_reason: str | None = None
                if regen_budget <= 0:
                    regen_blocked_reason = (
                        f"regen budget exhausted ({regen_budget})"
                    )
                elif target is None:
                    regen_blocked_reason = "no last audio subtask"
                elif target.action in _NON_GENERATIVE:
                    regen_blocked_reason = (
                        f"last audio step {target.step} is "
                        f"{target.action.value} (no generation to redo)"
                    )

                if regen_blocked_reason is None:
                    regen_budget -= 1
                    logger.warning(
                        "[MixEval] needs_regenerate=True — re-running audio "
                        "step %d (%s). Reason: %s",
                        target.step, target.action.value, result.reason[:160],
                    )
                    # Shift the inner attempt index past the original run's
                    # budget so the regen lands in a fresh attempt_NN/ dir
                    # instead of clobbering the original artifacts.
                    if target.action in (_EA.SPEECH_TTS, _EA.SPEECH_SWAP):
                        attempt_offset = self._MAX_RETRIES_SPEECH + 1
                    else:
                        attempt_offset = self._MAX_RETRIES_AUDIO + 1
                    try:
                        await self._run_one_subtask(
                            target, context, max_retries=0,
                            attempt_offset=attempt_offset,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[MixEval] regen handler raised: %s — keeping best",
                            exc,
                        )
                        break
                    new_audio = context.edited_audio
                    if new_audio is None:
                        logger.warning(
                            "[MixEval] regen produced no edited_audio — stopping"
                        )
                        break

                    # Optional: MixEval suggested both regen AND a volume
                    # tweak. Apply the tweak when remixing the new gen
                    # stem against the preserved original (only when both
                    # stems are available — otherwise just merge as-is).
                    new_preserved = context.original_audio
                    new_generated = context.last_generated_audio
                    if (
                        adj is not None
                        and new_preserved is not None
                        and new_generated is not None
                    ):
                        remix_dir = context.session_dir / "mix_retry"
                        remix_dir.mkdir(parents=True, exist_ok=True)
                        remixed_audio = remix_dir / f"regen{attempt + 1:02d}.aac"
                        try:
                            mix_audio_tracks(
                                original_audio=new_preserved,
                                generated_audio=new_generated,
                                output_path=remixed_audio,
                                original_volume=adj.original_volume,
                                generated_volume=adj.generated_volume,
                            )
                        except Exception as exc:
                            logger.warning(
                                "[MixEval] post-regen remix ffmpeg failed: "
                                "%s — using regen audio as-is", exc,
                            )
                        else:
                            new_audio = remixed_audio
                            context.edited_audio = remixed_audio
                            logger.info(
                                "[MixEval] post-regen remix orig=%.2f gen=%.2f",
                                adj.original_volume, adj.generated_volume,
                            )

                    new_output = context.session_dir / (
                        output_path.stem + f"_regen{attempt + 1:02d}.mp4"
                    )
                    merge_audio_video(video_part, new_audio, new_output)
                    logger.info("[MixEval] regen-mux → %s", new_output.name)
                    candidate_path = new_output
                    continue
                else:
                    # regen requested but blocked. If a volume tweak is
                    # also suggested, fall through to volume; otherwise
                    # stop.
                    logger.warning(
                        "[MixEval] needs_regenerate=True but %s — "
                        "trying volume fallback if available. Reason: %s",
                        regen_blocked_reason, result.reason[:160],
                    )
                    # falls through to volume branch below

            # Priority 2: VOLUME — cheap ffmpeg-only remix. Reached when
            # (a) needs_regenerate=False but adj is set (content is
            # right, just level off), OR (b) regen was requested but
            # blocked and a volume tweak was also suggested.
            if (
                adj is not None
                and preserved is not None
                and generated is not None
            ):
                remix_dir = context.session_dir / "mix_retry"
                remix_dir.mkdir(parents=True, exist_ok=True)
                remixed_audio = remix_dir / f"mix_{attempt + 1:02d}.aac"
                try:
                    mix_audio_tracks(
                        original_audio=preserved,
                        generated_audio=generated,
                        output_path=remixed_audio,
                        original_volume=adj.original_volume,
                        generated_volume=adj.generated_volume,
                    )
                except Exception as exc:
                    logger.warning("[MixEval] remix ffmpeg failed: %s", exc)
                    break
                context.edited_audio = remixed_audio
                new_output = context.session_dir / (
                    output_path.stem + f"_remix{attempt + 1:02d}.mp4"
                )
                merge_audio_video(video_part, remixed_audio, new_output)
                logger.info(
                    "[MixEval] re-mixed with orig=%.2f gen=%.2f → %s",
                    adj.original_volume, adj.generated_volume,
                    new_output.name,
                )
                candidate_path = new_output
                continue

            # Neither regen nor volume is applicable.
            logger.info(
                "[MixEval] fail with no actionable remediation "
                "(adj=%s, regen=%s) — stopping",
                adj is not None, result.needs_regenerate,
            )
            break

        context.mix_eval_result = best_result
        return best_output

    # ── assembly: concat shots + mux with edited audio ──────────────

    async def _assemble_final(
        self, context: _StepContext, output_path: Path,
    ) -> Path:
        """Assemble the final video from per-shot pieces (if any) and
        mux with the edited audio. Returns *output_path*."""
        from av_editor.core.postprocessor import merge_audio_video
        from av_editor.core.shot_slicer import concat_shots, slice_shot

        if context.any_per_shot_video_edit and context.shots:
            assembly_dir = context.session_dir / "assembly"
            assembly_dir.mkdir(parents=True, exist_ok=True)
            pieces: list[Path] = []
            base = context.current_global_video or context.base_video
            for shot in context.shots:
                piece = context.shot_videos.get(shot.index)
                if piece is None:
                    piece = assembly_dir / f"src_shot_{shot.index:03d}.mp4"
                    slice_shot(base, shot, piece)
                pieces.append(piece)
            # Normalise all pieces to the ORIGINAL source's fps/size
            # during concat. Edited outputs may land at exact 24 fps
            # while the source (and our slices) are 23.976 NTSC, and the
            # concat demuxer drops/corrupts frames in that case.
            from av_editor.core.shot_slicer import _probe_fps, _probe_size
            src_fps = _probe_fps(context.base_video)
            src_size = _probe_size(context.base_video)
            assembled = assembly_dir / "assembled.mp4"
            concat_shots(
                pieces, assembled, reencode=True,
                target_fps=src_fps, target_size=src_size,
            )
            video_part = assembled
        elif context.any_global_video_edit and context.current_global_video:
            video_part = context.current_global_video
        else:
            video_part = context.base_video

        audio_part = context.edited_audio or context.original_audio
        merge_audio_video(video_part, audio_part, output_path)
        logger.info("[Runner] final output → %s", output_path)

        # Final mixed-media evaluation runs for every plan. It first tries
        # the cheapest applicable local repair and exposes structural
        # failures to the outer full-replan loop.
        inventory = getattr(context, "audio_inventory", None)
        return await self._run_mix_eval_loop(
            context, video_part, output_path, inventory,
        )

    async def _improve_speaker_description(
        self,
        current_desc: str,
        new_text: str,
        eval_reason: str,
    ) -> str:
        """Rewrite ``speaker_description`` (the SAM Audio separation
        prompt for speech_replace_full) based on evaluator feedback
        indicating the wrong speaker was isolated last time. Returns a
        more disambiguating description."""
        from av_editor.core._gemini_client import gemini_with_fallback
        system = (
            "You are helping a speech-separation pipeline that uses a "
            "text-prompt audio separation model (SAM Audio). The current "
            "prompt failed to isolate the correct speaker. Rewrite it "
            "so it better disambiguates the TARGET speaker from any "
            "other voices in the audio.\n\n"
            "FORMAT (strict): a SINGLE noun phrase of the shape "
            "`[adjective(s)] noun`, ≤ 8 words, with ONE noun head "
            "(use 'voice' or 'speech' as the head). NO comma, NO "
            "conjunction ('and' / 'or'), NO sub-clause, NO negation. "
            "Adjectives must be acoustic / demographic-via-acoustic: "
            "gender, age band, pitch, timbre. Drop pure-language / "
            "accent / numerical-pitch / narrative descriptors unless "
            "they are the only disambiguator left.\n\n"
            "Examples (good): 'deep male voice', 'high female voice', "
            "'elderly male voice', 'young female speech'.\n"
            "Examples (BAD): 'adult male voice, mid-pitch, American "
            "English' (3 fragments), 'the man speaking' (no acoustic "
            "adjective).\n\n"
            "Return ONLY the new description — no quotes, no "
            "explanation."
        )
        user = (
            f"Current speaker description: \"{current_desc}\"\n"
            f"New line being synthesised: \"{new_text}\"\n"
            f"Evaluator feedback:\n{eval_reason}\n\n"
            "Rewrite the speaker description to fix the separation."
        )
        try:
            raw = await asyncio.to_thread(
                gemini_with_fallback,
                gemini_api_key=self.cfg.llm.gemini_api_key,
                primary_model=self.cfg.llm.gemini_model,
                fallback_model="gemini-2.5-flash",
                system_prompt=system,
                user_text=user,
                json_response=False,
                temperature=0.3,
                max_output_tokens=9999,
                component="SpeakerDescImprover",
            )
            new_desc = (raw or "").strip().strip('"')
            new_desc = _cap_words(new_desc, 15)
            return new_desc if new_desc else current_desc
        except Exception as exc:
            logger.warning(
                "[Pipeline] _improve_speaker_description failed: %s — "
                "keeping current", exc,
            )
            return current_desc

    # ── public API ─────────────────────────────────────────────────────

    async def run_audio_only(
        self,
        session_id: str,
        audio_instruction: str | None = None,
    ) -> Path:
        """
        Re-run only the audio stage for an existing session, keeping
        the previously-edited video-only track and just re-running the
        audio / speech SubTasks through the unified step runner.
        """
        import json as _json
        session_dir = self.workspace / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        logger.info("=" * 60)
        logger.info("Audio-only run for session %s", session_id)
        logger.info("=" * 60)

        # ── find edited video ─────────────────────────────────────────
        # Preference order:
        #   1) assembly/assembled.mp4      — unified runner's concat of
        #      per-shot edited clips (video-only, correct for muxing
        #      a new audio track on top).
        #   2) states/state_chain.json     — recorded state-chain fallback.
        #   3) preprocess/*_video_only.mp4 — unedited fallback (used
        #      only when no video edits ran in the original session).
        edited_video: Path | None = None
        assembled = session_dir / "assembly" / "assembled.mp4"
        if assembled.exists():
            edited_video = assembled
            logger.info("  using unified-runner assembled video: %s", assembled)
        if edited_video is None:
            state_chain_path = session_dir / "states" / "state_chain.json"
            if state_chain_path.exists():
                states = _json.loads(state_chain_path.read_text())
                cand = self.workspace.parent / states[-1]["video_path"]
                if not cand.exists():
                    cand = Path(states[-1]["video_path"])
                if cand.exists():
                    edited_video = cand
        if edited_video is None:
            vo_files = sorted((session_dir / "preprocess").glob("*_video_only.mp4"))
            if vo_files:
                edited_video = vo_files[0]
                logger.warning(
                    "  no edited video found — falling back to "
                    "preprocessed video-only (unedited): %s",
                    edited_video.name,
                )
        if edited_video is None:
            raise FileNotFoundError(
                f"No edited video or preprocessed video-only file in {session_dir}"
            )
        logger.info("  edited video: %s", edited_video)

        # ── original audio ────────────────────────────────────────────
        preprocess_dir = session_dir / "preprocess"
        audio_files = sorted(preprocess_dir.glob("*_audio.aac"))
        original_audio: Path | None = audio_files[0] if audio_files else None

        # ── video duration ────────────────────────────────────────────
        import subprocess as _sp
        probe = _sp.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(edited_video)],
            capture_output=True, text=True,
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 6.0

        # ── caption + shots ───────────────────────────────────────────
        caption_path = session_dir / "caption.txt"
        video_caption = (
            caption_path.read_text(encoding="utf-8").strip()
            if caption_path.exists() else ""
        )

        shots_json = session_dir / "shots.json"
        from av_editor.schema import Shot as _Shot
        shots_list: list = []
        if shots_json.exists():
            for s in _json.loads(shots_json.read_text()):
                shots_list.append(_Shot(
                    index=int(s["index"]),
                    start=float(s["start"]),
                    end=float(s["end"]),
                    summary=s.get("summary", ""),
                ))

        # ── subtasks: re-plan if new instruction, else reuse plan.json ─
        plan_path = session_dir / "plan.json"
        effective_instruction = audio_instruction or ""
        if audio_instruction:
            raw_subtasks = await self.planner.plan(
                instruction=audio_instruction,
                video_caption=video_caption or None,
                shots=shots_list or None,
            )
            audio_subs = [t for t in raw_subtasks if t.is_audio or t.action.is_speech]
        elif plan_path.exists():
            from av_editor.schema import EditAction as _EA, SubTask as _ST, TargetScope as _TS
            plan_data = _json.loads(plan_path.read_text())
            effective_instruction = str(plan_data.get("instruction", ""))
            audio_subs = []
            for t in plan_data.get("subtasks", []):
                try:
                    action = _EA(t["action"])
                except ValueError:
                    continue
                if not (action.is_audio or action.value.startswith("speech_")):
                    continue
                # Mirror the flat V3+ schema written by SubTask.to_dict():
                # action-specific fields are only present when relevant,
                # so default each one. params/description were dropped in
                # the V3 flattening — read intent + per-modality fields.
                audio_subs.append(_ST(
                    step=t["step"], action=action,
                    target=_TS(t.get("target", "global")),
                    shot_index=t.get("shot_index"),
                    depends_on=t.get("depends_on", []) or [],
                    intent=t.get("intent", ""),
                    eval_criteria=t.get("eval_criteria", []),
                    existing_sounds=t.get("existing_sounds", []),
                    deleted_sound=t.get("deleted_sound", ""),
                    new_sound=t.get("new_sound", ""),
                    sam_prompt=t.get("sam_prompt", ""),
                    mmaudio_prompt=t.get("mmaudio_prompt", ""),
                    expect_prominent_target=t.get("expect_prominent_target", False),
                    volume_target=t.get("volume_target", ""),
                    volume_db=t.get("volume_db", 0.0),
                    speech_text=t.get("speech_text", ""),
                    speech_speaker_description=t.get("speech_speaker_description", ""),
                    speech_voice_description=t.get("speech_voice_description", ""),
                    speech_reference_text=t.get("speech_reference_text", ""),
                    speech_language=t.get("speech_language", "auto"),
                    audio_splice=t.get("audio_splice", {}) if isinstance(
                        t.get("audio_splice", {}), dict
                    ) else {},
                ))
        else:
            raise FileNotFoundError("No plan.json and no audio_instruction provided")

        if not audio_subs:
            logger.warning("  no audio/speech subtasks — nothing to do")
            return edited_video

        # ── run ───────────────────────────────────────────────────────
        source_video: Path | None = None
        source_path_file = session_dir / "original_video_path.txt"
        if source_path_file.exists():
            source_candidate = Path(
                source_path_file.read_text(encoding="utf-8").strip()
            )
            if source_candidate.exists():
                source_video = source_candidate
        context = _StepContext(
            session_dir=session_dir,
            shots=shots_list,
            duration=duration,
            base_video=edited_video,
            original_audio=original_audio,
            original_video=source_video,
            video_caption=video_caption,
            instruction=effective_instruction,
            subtasks=audio_subs,
            audio_inventory=build_audio_inventory(audio_subs),
        )
        # Audio-only reruns: bump outer audio retries so a failed
        # post-branch eval (e.g. MMAudio leaked the visual subject's
        # native sound) gets another chance at a fresh SAM+MMAudio pass
        # rather than giving up after the branch's internal retries.
        saved_audio_retries = self._MAX_RETRIES_AUDIO
        self._MAX_RETRIES_AUDIO = max(saved_audio_retries, 3)
        try:
            await self._run_subtasks_ordered(audio_subs, context)
        finally:
            self._MAX_RETRIES_AUDIO = saved_audio_retries

        final_output = session_dir / f"final_audio_only_{session_id[:8]}.mp4"
        final_output = await self._assemble_final(context, final_output)
        logger.info("=" * 60)
        logger.info("Audio-only run completed → %s", final_output)
        logger.info("=" * 60)
        return final_output

    async def run(
        self,
        video_path: str | Path,
        instruction: str,
        plan_only: bool = False,
    ) -> Path:
        """
        Execute the full pipeline end-to-end.

        Parameters
        ----------
        video_path  : Path to the input video file.
        instruction : Natural-language editing instruction.
        plan_only   : If True, stop after planning (preprocess + caption + plan).

        Returns
        -------
        Path to the final output video (with original + generated audio).
        """
        video_path = Path(video_path).resolve()
        # Session dir = <video_stem>_<YYYYMMDD-HHMMSS> so concurrent or
        # repeated runs on the same video never clobber each other.
        from datetime import datetime
        base_name = video_path.stem
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_dir = self.workspace / f"{base_name}_{ts}"
        # Extremely unlikely, but handle collision within the same second.
        if session_dir.exists():
            session_dir = self.workspace / f"{base_name}_{ts}_{uuid.uuid4().hex[:4]}"
        session = EditSession(
            session_id=session_dir.name,
            original_video=video_path,
            instruction=instruction,
        )
        session_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Session %s started", session.session_id)
        logger.info("Input : %s", video_path)
        logger.info("Instruction: %s", instruction)
        logger.info("=" * 60)

        # Persist original video path so audio-only reruns can extract
        # lossless WAV directly from the source (avoiding AAC round-trip).
        (session_dir / "original_video_path.txt").write_text(
            str(video_path), encoding="utf-8",
        )

        # Set session_dir on planner / evaluator for JSON logging
        self.planner.session_dir = session_dir
        self.evaluator.session_dir = session_dir
        self.evaluator._eval_records = []  # reset for new session

        # ── 1. Preprocess: split audio / video ─────────────────────────
        logger.info("[1/5] Preprocessing...")
        session.preprocess = preprocess(video_path, session_dir)
        logger.info(
            "  video-only: %s | audio: %s | has_audio: %s",
            session.preprocess.video_path.name,
            session.preprocess.audio_path.name if session.preprocess.audio_path else "None",
            session.preprocess.has_audio,
        )

        # ── 1b. Extract keyframes for vision-aware planning ──────────
        keyframe_dir = session_dir / "preprocess" / "keyframes"
        keyframes = extract_keyframes(
            session.preprocess.video_path, keyframe_dir, session.preprocess.meta,
        )

        # ── 1c. Video captioning (Gemini 2.5 Flash) ──────────────────────
        video_caption = None
        from av_editor.core.video_captioner import CAPTION_MODEL
        logger.info("[1c] Captioning video with %s...", CAPTION_MODEL)
        # Use original video (with audio) so captioner can hear the soundtrack
        caption_video_path = (
            session.preprocess.original_video or session.preprocess.video_path
        )
        video_caption = await caption_video(
            caption_video_path,
            self.cfg.llm,
        )
        if video_caption:
            logger.info("  caption: %s...", video_caption[:120])
            (session_dir / "caption.txt").write_text(video_caption, encoding="utf-8")

        # ── 1d. Parse shots from caption ────────────────────────────────
        session.shots = parse_shots(
            video_caption or "",
            session.preprocess.meta.duration,
            video_path=session.preprocess.video_path,
        )
        for s in session.shots:
            logger.info(
                "  shot %d: [%.2f→%.2f]s — %s",
                s.index, s.start, s.end, s.summary[:60],
            )
        import json as _json
        (session_dir / "shots.json").write_text(
            _json.dumps([s.to_dict() for s in session.shots], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # ── 2. Plan: decompose instruction ─────────────────────────────
        logger.info("[2/5] Planning...")
        session.subtasks = await self.planner.plan(
            instruction=instruction,
            meta=session.preprocess.meta,
            keyframes=keyframes,
            video_caption=video_caption,
            shots=session.shots,
        )

        video_tasks = [t for t in session.subtasks if not t.is_audio]
        audio_tasks = [t for t in session.subtasks if t.is_audio]
        logger.info("  %d video subtask(s), %d audio subtask(s)",
                     len(video_tasks), len(audio_tasks))

        if not session.subtasks:
            logger.warning("Planner returned zero subtasks — nothing to edit.")

        # ── Plan-only mode: stop here ────────────────────────────────
        if plan_only:
            logger.info("=" * 60)
            logger.info("Plan-only mode — stopping after planning.")
            logger.info("Session dir: %s", session_dir)
            logger.info("=" * 60)
            return session_dir

        # ── 3. Execute, globally evaluate, and optionally full-replan ──
        current_subtasks = session.subtasks
        replan_feedback: str | None = None
        best_global_output: Path | None = None
        best_global_score = -1.0
        best_global_subtasks = current_subtasks
        replan_history: list[dict[str, Any]] = []

        for full_attempt in range(self._MAX_FULL_REPLANS + 1):
            cycle_dir = (
                session_dir
                if full_attempt == 0
                else session_dir / f"replan_{full_attempt:02d}"
            )
            cycle_dir.mkdir(parents=True, exist_ok=True)

            if full_attempt > 0:
                logger.warning(
                    "[FullReplan] cycle %d/%d — rebuilding plan from source",
                    full_attempt,
                    self._MAX_FULL_REPLANS,
                )
                self.planner.session_dir = cycle_dir
                self.evaluator.session_dir = cycle_dir
                self.evaluator._eval_records = []
                current_subtasks = await self.planner.plan(
                    instruction=instruction,
                    meta=session.preprocess.meta,
                    keyframes=keyframes,
                    video_caption=video_caption,
                    shots=session.shots,
                    replan_feedback=replan_feedback,
                )

            logger.info(
                "[3/3] Executing %d subtask(s) via unified runner%s...",
                len(current_subtasks),
                f" (full-replan cycle {full_attempt})" if full_attempt else "",
            )
            context = _StepContext(
                session_dir=cycle_dir,
                shots=session.shots,
                duration=session.preprocess.meta.duration,
                base_video=session.preprocess.video_path,
                original_audio=session.preprocess.audio_path,
                original_video=video_path,
                video_caption=video_caption or "",
                instruction=instruction,
                subtasks=current_subtasks,
                allow_full_replan=True,
                audio_inventory=getattr(self.planner, "last_inventory", None),
            )
            if current_subtasks:
                await self._run_subtasks_ordered(current_subtasks, context)
            else:
                logger.warning(
                    "[FullReplan] planner returned zero subtasks; evaluating "
                    "the unchanged source so the global evaluator can repair it"
                )

            cycle_output = cycle_dir / f"final_{video_path.stem}.mp4"
            candidate_output = await self._assemble_final(context, cycle_output)
            mix_result = context.mix_eval_result
            candidate_score = (
                mix_result.overall_score if mix_result is not None else 0.0
            )
            if best_global_output is None or candidate_score > best_global_score:
                best_global_output = candidate_output
                best_global_score = candidate_score
                best_global_subtasks = current_subtasks
                session.audio_inventory = context.audio_inventory

            replan_request = context.replan_request
            feedback = ""
            if replan_request is not None:
                observation = (
                    replan_request.replan_feedback
                    or replan_request.reason
                    or "The assembled result has a structural failure."
                )
                feedback = (
                    "Final mixed-evaluation evidence:\n"
                    f"- instruction_score: {replan_request.instruction_score:.3f}\n"
                    f"- fidelity_score: {replan_request.fidelity_score:.3f}\n"
                    f"- quality_score: {replan_request.quality_score:.3f}\n"
                    f"- structural_confidence: {replan_request.replan_confidence:.3f}\n"
                    f"- evaluator_reason: {replan_request.reason}\n"
                    f"- observed_structural_failure: {observation}"
                )
            try:
                candidate_ref = str(candidate_output.relative_to(session_dir))
            except ValueError:
                candidate_ref = candidate_output.name
            replan_history.append({
                "cycle": full_attempt,
                "candidate": candidate_ref,
                "score": candidate_score,
                "needs_replan": replan_request is not None,
                "feedback": feedback,
            })
            (session_dir / "full_replan.json").write_text(
                _json.dumps(replan_history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if replan_request is None:
                break
            if full_attempt >= self._MAX_FULL_REPLANS:
                logger.warning(
                    "[FullReplan] budget exhausted; returning global best "
                    "candidate (score %.3f)",
                    best_global_score,
                )
                break
            replan_feedback = feedback

        if best_global_output is None:
            raise RuntimeError("Pipeline produced no final candidate")

        # Publish the highest-scoring artifact at the session's canonical
        # output path even when it came from a nested replan cycle.
        final_output = session_dir / f"final_{video_path.stem}.mp4"
        if best_global_output.resolve() != final_output.resolve():
            shutil.copy2(best_global_output, final_output)
        session.subtasks = best_global_subtasks

        session.final_output = final_output
        logger.info("=" * 60)
        logger.info("Session %s completed", session.session_id)
        logger.info("Output: %s", final_output)
        logger.info("=" * 60)

        return final_output
