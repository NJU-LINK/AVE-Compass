"""
shot_slicer.py - ffmpeg helpers for slicing a video into shots and
concatenating edited/unedited shots back into a single video.

Used by the per-shot executor: video tasks with ``shot_indices`` only
process the targeted shots, while the remaining shots are passed
through unchanged. After each task the edited and untouched shot
files are concatenated back into one ``current_video``.

All outputs are re-encoded once with consistent codec params so later
concat operations can be lossless (-c copy).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from av_editor.schema import Shot

logger = logging.getLogger(__name__)


def _run(cmd: list[str], description: str = "") -> subprocess.CompletedProcess:
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[{description}] failed (rc={result.returncode}):\n{result.stderr}"
        )
    return result


# ── shot slicing ───────────────────────────────────────────────────────

def slice_shot(
    source_video: Path,
    shot: Shot,
    output_path: Path,
    video_codec: str = "libx264",
    audio_codec: str | None = None,
    crf: int = 14,
) -> Path:
    """Cut ``[shot.start, shot.end)`` out of *source_video* and re-encode
    with stable codec params so later concat via the demuxer is lossless.

    If ``audio_codec`` is None, the source's audio (if any) is dropped —
    caller is expected to handle audio globally.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-ss", f"{shot.start:.3f}",
        "-to", f"{shot.end:.3f}",
        "-i", str(source_video),
        "-c:v", video_codec,
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
    ]
    if audio_codec is None:
        cmd += ["-an"]
    else:
        cmd += ["-c:a", audio_codec]
    cmd += [str(output_path)]

    _run(cmd, f"slice_shot[{shot.index}]")
    logger.info(
        "[ShotSlicer] shot %d [%.2f→%.2f] → %s",
        shot.index, shot.start, shot.end, output_path.name,
    )
    return output_path


def slice_all_shots(
    source_video: Path,
    shots: list[Shot],
    output_dir: Path,
    audio_codec: str | None = None,
) -> list[Path]:
    """Slice every shot into ``output_dir/shot_{idx:03d}.mp4`` and return
    the list of paths in index order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for shot in shots:
        dst = output_dir / f"shot_{shot.index:03d}.mp4"
        slice_shot(source_video, shot, dst, audio_codec=audio_codec)
        paths.append(dst)
    return paths


# ── shot slicing with freeze-frame padding ─────────────────────────────

def slice_shot_padded(
    source_video: Path,
    shot: Shot,
    output_path: Path,
    min_duration: float,
    video_codec: str = "libx264",
    crf: int = 14,
) -> tuple[Path, float]:
    """Cut out a shot and, if it is shorter than *min_duration*, extend
    it by cloning the last frame until it reaches *min_duration*.

    Some video APIs reject inputs below a fixed length.
    Freeze-frame padding is preferred over black/silence because it
    does not introduce a synthetic discontinuity the model might try
    to "explain" (e.g. by dimming lighting or shifting tone into the
    edited region).

    Returns ``(output_path, original_duration)`` so callers can trim
    the edited result back to the real shot length.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    orig_duration = shot.duration

    base_cmd: list[str] = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-ss", f"{shot.start:.3f}",
        "-to", f"{shot.end:.3f}",
        "-i", str(source_video),
    ]

    if orig_duration >= min_duration:
        cmd = base_cmd + [
            "-c:v", video_codec, "-crf", str(crf), "-preset", "medium",
            "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
            "-an", str(output_path),
        ]
    else:
        # Clone the final frame until we reach min_duration. ``tpad`` in
        # stop_mode=clone repeats the last frame; stop_duration is how
        # much to *add* beyond the source length.
        pad_secs = max(0.0, min_duration - orig_duration) + 0.05  # small safety margin
        vf = f"tpad=stop_mode=clone:stop_duration={pad_secs:.3f}"
        cmd = base_cmd + [
            "-vf", vf,
            "-c:v", video_codec, "-crf", str(crf), "-preset", "medium",
            "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
            "-an", str(output_path),
        ]
        logger.info(
            "[ShotSlicer] shot %d [%.2f→%.2f] is %.2fs (<%.1fs) — "
            "padding with frozen last frame +%.2fs",
            shot.index, shot.start, shot.end,
            orig_duration, min_duration, pad_secs,
        )

    _run(cmd, f"slice_shot_padded[{shot.index}]")
    logger.info(
        "[ShotSlicer] shot %d → %s (orig=%.2fs)",
        shot.index, output_path.name, orig_duration,
    )
    return output_path, orig_duration


def trim_to_duration(
    source_video: Path,
    duration: float,
    output_path: Path,
    video_codec: str = "libx264",
    crf: int = 14,
) -> Path:
    """Trim *source_video* to exactly *duration* seconds from the
    start. Used to drop the frozen padding tail introduced by
    :func:`slice_shot_padded` after a V2V edit.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-i", str(source_video),
        "-t", f"{duration:.3f}",
        "-c:v", video_codec, "-crf", str(crf), "-preset", "medium",
        "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
        "-an", str(output_path),
    ]
    _run(cmd, "trim_to_duration")
    logger.info(
        "[ShotSlicer] trimmed → %s (%.2fs)", output_path.name, duration,
    )
    return output_path


# ── audio slicing ──────────────────────────────────────────────────────

def slice_audio(
    source_audio: Path,
    start: float,
    end: float,
    output_path: Path,
    codec: str = "pcm_s16le",
) -> Path:
    """Cut ``[start, end)`` out of *source_audio* and re-encode so the
    result is directly usable as an AI model input. Default output is
    16-bit PCM WAV (widely accepted by lipsync/TTS APIs).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(source_audio),
        "-c:a", codec,
        str(output_path),
    ]
    _run(cmd, "slice_audio")
    logger.info(
        "[ShotSlicer] audio [%.2f→%.2f] → %s",
        start, end, output_path.name,
    )
    return output_path


# ── concat ─────────────────────────────────────────────────────────────

def concat_shots(
    shot_files: list[Path],
    output_path: Path,
    reencode: bool = False,
    target_fps: str | None = None,
    target_size: tuple[int, int] | None = None,
) -> Path:
    """Concatenate shot files in order.

    - When ``reencode=False`` (default) use the ffmpeg concat demuxer
      with ``-c copy`` — lossless and fast, but requires every input
      to share codec / fps / resolution / timebase.
    - When ``reencode=True`` use the ffmpeg concat FILTER instead —
      slower (re-encode) but handles heterogeneous inputs correctly
      (different fps, timebases, codecs). If ``target_fps`` is given
      (e.g. ``"24000/1001"``) or ``target_size`` is given, inputs are
      normalised to those values before concatenation. This is the
      right choice when mixing outputs from different models (generated
      clips may use exact 24fps while our slices inherit the source's
      23.976 fps — concat demuxer + -c copy silently corrupts timing).
    """
    if not shot_files:
        raise ValueError("concat_shots requires at least one input")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if reencode:
        return _concat_via_filter(
            shot_files, output_path, target_fps, target_size,
        )

    list_file = output_path.parent / f"{output_path.stem}_concat.txt"
    list_file.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in shot_files),
        encoding="utf-8",
    )

    cmd: list[str] = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]

    try:
        _run(cmd, "concat_shots")
    except RuntimeError:
        logger.warning(
            "[ShotSlicer] concat via -c copy failed, retrying with filter-based concat",
        )
        try:
            list_file.unlink()
        except OSError:
            pass
        return _concat_via_filter(shot_files, output_path, None, None)

    logger.info(
        "[ShotSlicer] concatenated %d shot(s) → %s",
        len(shot_files), output_path.name,
    )
    try:
        list_file.unlink()
    except OSError:
        pass
    return output_path


def _concat_via_filter(
    shot_files: list[Path],
    output_path: Path,
    target_fps: str | None,
    target_size: tuple[int, int] | None,
) -> Path:
    """Concatenate via ffmpeg's concat FILTER — normalises fps/size
    per input before joining, so mixed-fps inputs don't corrupt
    timestamps. Video-only (audio is handled separately in postprocess)."""
    if target_fps is None:
        target_fps = _probe_fps(shot_files[0])
    if target_size is None:
        target_size = _probe_size(shot_files[0])

    inputs: list[str] = []
    for p in shot_files:
        inputs += ["-i", str(p)]

    # Per-input normalising chain: force size, SAR, fps, pixel format.
    # Using setpts=PTS-STARTPTS keeps timestamps well-formed after fps
    # normalisation.
    w, h = target_size
    per_input = []
    for i in range(len(shot_files)):
        per_input.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,fps={target_fps},format=yuv420p,"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
    concat_inputs = "".join(f"[v{i}]" for i in range(len(shot_files)))
    filter_complex = (
        ";".join(per_input)
        + f";{concat_inputs}concat=n={len(shot_files)}:v=1:a=0[outv]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "libx264", "-crf", "14", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-r", target_fps, "-fps_mode", "cfr",
        "-an",
        str(output_path),
    ]
    _run(cmd, "concat_via_filter")
    logger.info(
        "[ShotSlicer] filter-concat %d shot(s) @ fps=%s %dx%d → %s",
        len(shot_files), target_fps, w, h, output_path.name,
    )
    return output_path


def _probe_fps(video_path: Path) -> str:
    """Return r_frame_rate (e.g. '24000/1001') as a string."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    s = (r.stdout or "").strip()
    return s if s and "/" in s else "24000/1001"


def _probe_size(video_path: Path) -> tuple[int, int]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=s=x:p=0", str(video_path)],
        capture_output=True, text=True,
    )
    s = (r.stdout or "").strip()
    try:
        w, h = s.split("x")
        return int(w), int(h)
    except Exception:
        return 1920, 1080
