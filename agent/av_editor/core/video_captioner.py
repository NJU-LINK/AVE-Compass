"""
video_captioner.py - Video content analysis using Gemini 2.5 Flash
                     via the official Gemini API.

Sends the ENTIRE video to Gemini 2.5 Flash so the model
can analyse temporal dynamics (motion, actions, transitions), not
just static frames. Uses the official Gemini API channel.

The description is fed to the Planner as additional context so
editing plans are more precise.

Gracefully degrades: returns None if the API call fails.
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

from av_editor.core._api_log import log_prompt
from av_editor.schema import Shot

logger = logging.getLogger(__name__)

# Model to use for video captioning (official Gemini API)
CAPTION_MODEL = "gemini-2.5-flash"

CAPTION_PROMPT = """\
Describe this video for a video editing AI assistant.
Your description will be used to plan both VISUAL and AUDIO edits.

## Audio analysis (answer FIRST)
**CRITICAL: Describe ONLY what you actually HEAR in the audio track.
Do NOT guess or infer sounds based on what you SEE in the video.**
For example, if you see people walking but hear only background music,
report "background music" — do NOT report "footsteps".

1. What sounds do you ACTUALLY HEAR? Categorize each as:
   - music/BGM (describe genre, tempo, mood, instruments if identifiable).
     **Pay close attention**: many short clips have background music or
     cinematic scores. If you hear ANY melodic, rhythmic, or tonal
     pattern — even subtle — report it as "music/BGM".
   - speech/dialogue (who is talking, what language)
   - natural sounds (wind, rain, birdsong, water, etc.)
   - urban/mechanical sounds (traffic, machinery, etc.)
   - silence / near-silence
2. Audio layers: what is the PRIMARY sound (loudest/most prominent)?
   What is secondary? Is there human speech? (yes/no)
   Is there background music? (yes/no)
3. Audio mood: noisy, quiet, calm, chaotic, natural, urban?
4. Speech transcript (REQUIRED when there is any dialogue / narration):
   Transcribe EVERY spoken utterance VERBATIM, in chronological order.
   Format each line as:
     [start–end] (speaker description, language) "spoken text"
   Example:
     [0.00–2.80] (adult male voice, American English) "What, what it matters?"
     [2.80–5.84] (adult female voice, American English) "Why are you asking me that?"
   If speech is partial / low confidence, still transcribe what you hear
   and mark uncertain spans with […]. If there is no speech, write
   "No speech." The downstream planner uses this transcript as the
   LITERAL `text` for any speech edit — do NOT paraphrase or shorten.

## Visual analysis
4. Objects and people visible (appearance, material, color)
5. Spatial layout (foreground, midground, background)
6. Lighting, time of day, color palette
7. Motion and actions (camera movement, people movement)
8. Weather and environment (sky, ground, atmosphere)
9. Overall mood/atmosphere

## Shot list (answer LAST)
Split the video into distinct camera shots using cut/transition
boundaries. A "shot" is a continuous segment between two camera cuts
(hard cut, jump cut, fade, dissolve, or scene change). Mark every
real cut you see — DOWNSTREAM logic decides whether to merge shots
with identical camera setups into a single edit.

For each shot, the `summary` should describe the CAMERA SETUP
(framing, angle, position) plus the SUBJECT and any action — clear
enough that a planner can later decide whether two shots share the
same setup. Use phrases like "wide front shot", "close-up profile",
"low-angle behind", etc.

Do NOT split for: brief flashes, micro-stutters, encoding glitches,
or compression artifacts that are not real edits.

If the whole video is a single uninterrupted take, output exactly
one shot spanning the full duration.

The last shot's `end` MUST equal the total video duration. Do not
drop trailing content.

Output the shot list as a fenced ```json block using EXACTLY this schema:

```json
[
  {"index": 1, "start": 0.00, "end": 2.34, "summary": "close-up of the man speaking, warm lighting"},
  {"index": 2, "start": 2.34, "end": 5.82, "summary": "close-up of the woman listening"}
]
```

Rules:
- `index` is 1-based, contiguous, in presentation order.
- `start` and `end` are in seconds (two decimal places), relative to the
  video start. `start_k == end_{k-1}` (shots are back-to-back).
- The last shot's `end` must equal the total video duration.
- `summary` is ≤ 20 words, naming the subject, framing, and any action —
  enough for a downstream planner to match user intent to a specific shot.

Be specific and concise.
"""


_SHOT_BLOCK_RE = re.compile(
    r"##\s*Shot\s*list.*?```(?:json)?\s*(\[.*?\])\s*```",
    re.IGNORECASE | re.DOTALL,
)


def _detect_scene_cuts(
    video_path: Path, threshold: float = 0.3,
) -> list[float]:
    """Return ffmpeg-detected scene cut timestamps (seconds, sorted).

    Uses the standard `select='gt(scene,X)',metadata=print` filter
    chain. Returns an empty list if ffmpeg is unavailable or detects
    no cuts. The threshold controls sensitivity — 0.3 is a common
    value that catches hard cuts and most jump cuts without firing
    on intra-shot motion.
    """
    import subprocess as _sp
    try:
        proc = _sp.run(
            [
                "ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path),
                "-filter:v",
                f"select='gt(scene,{threshold})',metadata=print:file=-",
                "-an", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:
        logger.warning("[Captioner] scene-detect ffmpeg call failed: %s", exc)
        return []

    cuts: list[float] = []
    # metadata=print writes frame info to stdout; pts_time is the
    # display-time of the cut frame.
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        m = re.search(r"pts_time:([0-9]+\.?[0-9]*)", line)
        if m:
            try:
                cuts.append(float(m.group(1)))
            except ValueError:
                continue
    cuts = sorted(set(round(c, 3) for c in cuts))
    if cuts:
        logger.info(
            "[Captioner] ffmpeg scene-detect found %d cut(s) at %s",
            len(cuts),
            ", ".join(f"{c:.2f}s" for c in cuts[:8]),
        )
    return cuts


def parse_shots(
    caption: str,
    total_duration: float,
    video_path: Path | None = None,
) -> list[Shot]:
    """Extract the ``## Shot list`` JSON block from a caption and return
    a list of :class:`Shot`. Falls back to a single full-video shot if
    the block is missing or malformed.

    When ``video_path`` is provided, the timestamps from the caption are
    cross-checked against ffmpeg scene-detect output and replaced with
    the real cut times when Gemini's are clearly bogus.
    """
    fallback = [Shot(index=1, start=0.0, end=round(total_duration, 3),
                     summary="(entire video, no shot breakdown)")]
    if not caption:
        return fallback

    m = _SHOT_BLOCK_RE.search(caption)
    if not m:
        logger.info("[Captioner] no shot list found — treating as single shot")
        return fallback
    try:
        raw = json.loads(m.group(1))
    except Exception as exc:
        logger.warning("[Captioner] shot list JSON parse failed: %s", exc)
        return fallback

    shots: list[Shot] = []
    for i, item in enumerate(raw):
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", total_duration))
            if end <= start:
                continue
            shots.append(Shot(
                index=int(item.get("index", i + 1)),
                start=round(start, 3),
                end=round(end, 3),
                summary=str(item.get("summary", "")).strip(),
            ))
        except Exception as exc:
            logger.warning("[Captioner] skipping malformed shot %d: %s", i, exc)

    if not shots:
        return fallback

    # Sort and re-index contiguously.
    shots.sort(key=lambda s: s.start)
    for i, s in enumerate(shots):
        s.index = i + 1

    total = round(total_duration, 3)

    def _cuts_with_cascade(target_n: int) -> tuple[list[float], int]:
        """Run ffmpeg scene-detect at multiple thresholds; return the
        candidate (cuts, diff_to_target) whose segment count is closest
        to *target_n*. Empty cuts when ffmpeg detects nothing across
        all thresholds."""
        best: list[float] = []
        best_diff = float("inf")
        if video_path is None:
            return best, int(best_diff) if best_diff != float("inf") else -1
        for thresh in (0.3, 0.2, 0.15, 0.1, 0.05):
            cand = _detect_scene_cuts(video_path, threshold=thresh)
            cand = [c for c in cand if 0.05 < c < total - 0.05]
            diff = abs((len(cand) + 1) - target_n)
            if diff < best_diff:
                best_diff = diff
                best = cand
            if diff == 0:
                break
        return best, int(best_diff) if best_diff != float("inf") else -1

    def _shots_from_boundaries(
        boundaries: list[float], summaries: list[str],
    ) -> list[Shot]:
        out: list[Shot] = []
        n = len(boundaries) - 1
        for i in range(n):
            summary = (
                summaries[i] if i < len(summaries)
                else f"(detected shot {i + 1}, no caption summary)"
            )
            out.append(Shot(
                index=i + 1,
                start=round(boundaries[i], 3),
                end=round(boundaries[i + 1], 3),
                summary=summary,
            ))
        return out

    # PRIMARY path: when Gemini reports multiple shots, use ffmpeg
    # scene-detect for the ACTUAL cut frame timestamps. Gemini's
    # named timestamps are LLM hallucinations of seconds — even when
    # they don't trigger the bogus check, they're often off by 0.5-2s
    # or miss real cuts entirely (mv_03 case). ffmpeg's pixel-diff
    # scene filter is frame-accurate. We keep Gemini's shot summaries
    # and pair them with the detected segments by order.
    target_n = len(shots)
    if target_n > 1 and video_path is not None:
        cuts, diff = _cuts_with_cascade(target_n)
        if cuts:
            boundaries = [0.0] + cuts + [total]
            n_detected = len(boundaries) - 1
            if n_detected != target_n:
                logger.warning(
                    "[Captioner] Gemini reported %d shot(s) but ffmpeg "
                    "scene-detect found %d; using detected count.",
                    target_n, n_detected,
                )
            ffmpeg_shots = _shots_from_boundaries(
                boundaries, [s.summary for s in shots],
            )
            logger.info(
                "[Captioner] used ffmpeg cuts (target=%d, detected=%d): %s",
                target_n, n_detected,
                ", ".join(f"[{s.start:.2f}-{s.end:.2f}]" for s in ffmpeg_shots[:8]),
            )
            return ffmpeg_shots
        # ffmpeg found nothing usable (rare — possibly all soft cuts
        # below threshold 0.05). Fall through to Gemini's timestamps.
        logger.warning(
            "[Captioner] Gemini reported %d shot(s) but ffmpeg detected "
            "no cuts; keeping Gemini's timestamps as best-effort.",
            target_n,
        )

    # Sanity rescue for captioner timestamp hallucinations on the
    # single-shot path (or when ffmpeg above produced nothing).
    # Symptom: all shots packed into <1s of a multi-second video.
    annotated_max = max(s.end for s in shots)
    coverage_ratio = annotated_max / total if total > 0 else 1.0
    timestamps_bogus = (
        total >= 1.0
        and (annotated_max < 1.0 or coverage_ratio < 0.5)
    )
    if timestamps_bogus:
        logger.warning(
            "[Captioner] shot timestamps look bogus "
            "(max end %.2fs vs video %.2fs, %.1f%% coverage) — "
            "running ffmpeg scene-detect to recover real cut times.",
            annotated_max, total, 100 * coverage_ratio,
        )
        cuts, _ = _cuts_with_cascade(len(shots))
        if cuts:
            boundaries = [0.0] + cuts + [total]
            new_shots = _shots_from_boundaries(
                boundaries, [s.summary for s in shots],
            )
            logger.info(
                "[Captioner] rescued %d shot(s) from scene-detect: %s",
                len(new_shots),
                ", ".join(f"[{s.start:.2f}-{s.end:.2f}]" for s in new_shots[:8]),
            )
            return new_shots
        logger.warning(
            "[Captioner] scene-detect produced no usable cuts; "
            "collapsing to a single full-video shot.",
        )
        return [Shot(
            index=1, start=0.0, end=total,
            summary=shots[0].summary or "(entire video)",
        )]

    # Extend the last shot to cover the full video duration. The
    # captioner sometimes truncates the trailing seconds (treating a
    # late jump-cut as out-of-scope or just rounding short). Without
    # this extension the per-shot router silently drops that tail.
    if shots[-1].end < total:
        logger.info(
            "[Captioner] extending last shot end %.2f → %.2f to cover full video",
            shots[-1].end, total,
        )
        shots[-1].end = total
    elif shots[-1].end > total:
        shots[-1].end = total

    logger.info("[Captioner] parsed %d shot(s)", len(shots))
    return shots


def _video_to_base64_url(video_path: Path) -> str:
    """Read video file and return as data:video/...;base64,... URL."""
    mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


async def caption_video(
    video_path: Path,
    llm_config,
) -> str | None:
    """
    Send the entire video to Gemini 2.5 Flash via the official Gemini API
    for detailed scene description.

    Returns None on any failure.
    """
    def _call() -> str:
        from av_editor.core._gemini_client import generate_with_media

        # Retry with exponential backoff
        max_retries = 3
        last_err = None
        for attempt in range(max_retries):
            try:
                text = generate_with_media(
                    api_key=getattr(llm_config, "gemini_api_key", ""),
                    model=CAPTION_MODEL,
                    system_prompt="You are a professional video analyst.",
                    user_text=CAPTION_PROMPT,
                    media_paths=[video_path],
                    json_response=False,
                    temperature=0,
                    max_output_tokens=9999,
                    component="VideoCaptioner",
                )
                text = (text or "").strip()
                if not text:
                    raise ValueError("Empty response")
                return text
            except Exception as e:
                last_err = e
                wait_s = min(2 ** attempt, 10)
                logger.warning(
                    "[Captioner] attempt %d/%d failed: %s — retry in %ds",
                    attempt + 1, max_retries, e, wait_s,
                )
                time.sleep(wait_s)

        raise last_err if last_err else RuntimeError("Caption API failed")

    try:
        result = await asyncio.to_thread(_call)
        logger.info("[Captioner] generated caption (%d chars)", len(result or ""))
        return result
    except Exception as exc:
        logger.warning("[Captioner] failed: %s — continuing without caption", exc)
        return None
