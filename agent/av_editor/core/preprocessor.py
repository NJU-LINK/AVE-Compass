"""
preprocessor.py - Audio/Video separation and metadata extraction.

Responsibilities
----------------
1. Probe the input video with ffprobe to get metadata.
2. Extract the audio track and save it separately.
3. Strip the audio from the video, producing a video-only file.

The audio file is kept untouched throughout the entire pipeline and
re-muxed in the postprocessor.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from av_editor.schema import PreprocessResult, VideoMeta

logger = logging.getLogger(__name__)


def _run(cmd: list[str], description: str = "") -> subprocess.CompletedProcess:
    """Run a subprocess and raise on failure."""
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[{description or cmd[0]}] failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr}"
        )
    return result


# ── Probe ──────────────────────────────────────────────────────────────────

def probe_video(video_path: Path) -> VideoMeta:
    """Extract video metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path),
    ]
    result = _run(cmd, "ffprobe")
    info = json.loads(result.stdout)

    # Find the video stream
    video_stream = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            video_stream = s
            break
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    # Parse fps from avg_frame_rate "30/1" or "30000/1001"
    fps_parts = video_stream.get("avg_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 and float(fps_parts[1]) else 30.0

    duration = float(info.get("format", {}).get("duration", 0))
    total_frames = int(video_stream.get("nb_frames", 0))
    if total_frames == 0:
        total_frames = int(fps * duration)

    return VideoMeta(
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=round(fps, 3),
        duration=round(duration, 3),
        codec=video_stream.get("codec_name", "unknown"),
        total_frames=total_frames,
    )


def _has_audio_stream(video_path: Path) -> bool:
    """Check whether the file contains at least one audio stream."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())


# ── Split ──────────────────────────────────────────────────────────────────

def extract_audio(video_path: Path, output_dir: Path) -> Path | None:
    """
    Extract the audio track to a separate file.
    Returns the audio file path, or None if the video has no audio.
    """
    if not _has_audio_stream(video_path):
        logger.info("Video has no audio stream — skipping audio extraction.")
        return None

    audio_path = output_dir / f"{video_path.stem}_audio.aac"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",            # drop video
        "-c:a", "copy",   # copy audio codec (lossless passthrough)
        str(audio_path),
    ]
    _run(cmd, "extract_audio")
    logger.info("Audio extracted → %s", audio_path)
    return audio_path


def strip_audio(video_path: Path, output_dir: Path) -> Path:
    """
    Produce a video-only file (audio removed) by stream-copying.
    """
    video_only = output_dir / f"{video_path.stem}_video_only.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-an",            # drop audio
        "-c:v", "copy",   # copy video codec (lossless)
        str(video_only),
    ]
    _run(cmd, "strip_audio")
    logger.info("Video-only file → %s", video_only)
    return video_only


# ── HDR → SDR tone mapping ────────────────────────────────────────────────

def _is_hdr(video_path: Path) -> bool:
    """Detect HDR video by checking color transfer characteristics."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=color_transfer,color_primaries,color_space,pix_fmt",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if not streams:
        return False
    s = streams[0]
    hdr_indicators = {"smpte2084", "arib-std-b67", "bt2020nc", "bt2020c", "bt2020"}
    vals = {
        s.get("color_transfer", ""),
        s.get("color_primaries", ""),
        s.get("color_space", ""),
    }
    pix_fmt = s.get("pix_fmt", "")
    # 10-bit pixel formats are also a strong HDR indicator
    if "10le" in pix_fmt or "10be" in pix_fmt or "p010" in pix_fmt:
        return True
    return bool(vals & hdr_indicators)


def _has_zscale() -> bool:
    """Check if ffmpeg was built with zscale (libzimg) support."""
    import subprocess as _sp
    try:
        out = _sp.run(
            ["ffmpeg", "-filters"],
            capture_output=True, text=True, timeout=10,
        )
        return "zscale" in out.stdout
    except Exception:
        return False


def tonemap_hdr_to_sdr(video_path: Path, output_dir: Path) -> Path:
    """
    Convert HDR video to SDR using ffmpeg tone-mapping filters.

    Prefers zscale + tonemap (high quality, requires libzimg).
    Falls back to colorspace filter (built-in, no extra libs needed)
    when zscale is not available (e.g. macOS Homebrew ffmpeg).
    """
    sdr_path = output_dir / f"{video_path.stem}_sdr.mp4"

    # Source peak luminance. HDR10 masters are typically graded at 1000 nit
    # (and occasionally 4000). We default to 1000, which is a much better
    # tonemapper anchor than 100 — setting npl=100 tells the tonemapper
    # "the source is already ~SDR", so it under-expands the mid-tones and
    # the result looks washed out and dim. peak= feeds the same figure to
    # hable so it knows the real dynamic range it's compressing.
    src_peak_nit = 1000

    if _has_zscale():
        # zscale decodes PQ/HLG → linear light correctly; then hable
        # compresses to ~100 nit; finally we encode back to BT.709 gamma.
        # desat=2 is hable's default — it pulls out-of-gamut highlights
        # toward the neutral axis instead of hard-clipping to chalky white.
        vf = (
            f"zscale=t=linear:npl={src_peak_nit},"
            "format=gbrpf32le,"
            "zscale=p=bt709,"
            f"tonemap=tonemap=hable:desat=2:peak={src_peak_nit},"
            "zscale=t=bt709:m=bt709:r=tv,"
            "format=yuv420p,"
            "setsar=1:1"
        )
    else:
        # Fallback (no libzimg): there is no clean PQ→linear path in
        # vanilla ffmpeg without zscale (the `colorspace` filter cannot
        # do it; `libplacebo` would but is also rare). The chain below
        # feeds PQ values directly into `tonemap`, which assumes linear
        # input — the result is darker and lower-contrast than the
        # zscale path. We keep this so the pipeline can RUN on HDR
        # input even without zimg, but quality is degraded.
        #
        # To get correct HDR→SDR conversion, install ffmpeg with
        # libzimg, e.g.:
        #   brew tap homebrew-ffmpeg/ffmpeg
        #   brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-zimg
        # Verify with: `ffmpeg -filters | grep zscale`.
        logger.warning(
            "HDR→SDR fallback (no zscale): output WILL be darker and "
            "less saturated than a proper zimg-backed conversion. "
            "Install ffmpeg with libzimg for correct results."
        )
        vf = (
            "format=gbrpf32le,"
            f"tonemap=tonemap=hable:desat=2:peak={src_peak_nit},"
            "colorspace=all=bt709:iall=bt2020:fast=0,"
            "format=yuv420p,"
            "setsar=1:1"
        )

    cmd = [
        "ffmpeg", "-y", "-fflags", "+genpts", "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "14",
        "-fps_mode", "cfr",
        "-an",
        str(sdr_path),
    ]
    _run(cmd, "tonemap_hdr_to_sdr")
    logger.info("HDR → SDR tone-mapped → %s", sdr_path)
    return sdr_path


# ── Upscale ────────────────────────────────────────────────────────────────

MIN_DIM = 720    # Cloud video editors expect dimensions in [720, 2160]
MAX_DIM = 2160

def upscale_if_needed(video_path: Path, meta: VideoMeta, output_dir: Path) -> tuple[Path, VideoMeta]:
    """
    Ensure BOTH dimensions are within the supported [MIN_DIM, MAX_DIM] range.
    Earlier the function only checked width — portrait clips like
    1440x2560 then sailed through preprocessing only to be rejected later.
    We now scale to fit the long edge under MAX_DIM and
    the short edge above MIN_DIM, preserving aspect ratio.
    """
    w, h = meta.width, meta.height

    long_edge, short_edge = max(w, h), min(w, h)
    if MIN_DIM <= short_edge and long_edge <= MAX_DIM:
        return video_path, meta

    scale = 1.0
    if long_edge > MAX_DIM:
        scale = MAX_DIM / long_edge
    if short_edge * scale < MIN_DIM:
        # Bumping short edge up may push long edge over MAX_DIM; in
        # that rare case (extreme aspect ratio) we accept the long
        # edge stays above MAX_DIM and let the backend reject it — there is
        # no proportional fix.
        scale = max(scale, MIN_DIM / short_edge)

    new_w = int(round(w * scale / 2)) * 2  # even
    new_h = int(round(h * scale / 2)) * 2

    if new_w == w and new_h == h:
        return video_path, meta

    if scale > 1.0:
        tag = "upscaled"
        logger.info(
            "Video %dx%d below MIN_DIM=%d — upscaling to %dx%d",
            w, h, MIN_DIM, new_w, new_h,
        )
    else:
        tag = "downscaled"
        logger.info(
            "Video %dx%d exceeds MAX_DIM=%d — downscaling to %dx%d",
            w, h, MAX_DIM, new_w, new_h,
        )
    scale_filter = f"scale={new_w}:{new_h}"

    out = output_dir / f"{video_path.stem}_{tag}.mp4"
    cmd = [
        "ffmpeg", "-y", "-fflags", "+genpts", "-i", str(video_path),
        "-vf", f"{scale_filter},setsar=1:1",
        "-c:v", "libx264", "-preset", "medium", "-crf", "14",
        "-fps_mode", "cfr",
        "-an",  # already stripped
        str(out),
    ]
    _run(cmd, tag)

    new_meta = probe_video(out)
    logger.info("Scaled %dx%d → %dx%d", w, h, new_meta.width, new_meta.height)
    return out, new_meta


# ── Public entry point ─────────────────────────────────────────────────────

# ── Keyframe extraction ───────────────────────────────────────────────────

def extract_keyframe(
    video_path: Path,
    output_dir: Path,
    time_sec: float = 0.5,
    prefix: str = "",
) -> Path:
    """Extract a single JPEG keyframe from *video_path* at *time_sec*.

    *prefix* is prepended to the output filename — pass distinct
    prefixes (e.g. ``"before_"`` vs ``"after_"``) when extracting
    frames from multiple source videos into the same *output_dir*,
    otherwise the second call silently overwrites the first.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / f"{prefix}frame_{time_sec:.2f}s.jpg"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(time_sec),
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(frame_path),
    ]
    _run(cmd, "extract_keyframe")
    return frame_path


def extract_frames_sampled(
    video_path: Path,
    output_dir: Path,
    fps: float = 2.0,
    prefix: str = "",
    max_frames: int = 8,
) -> list[Path]:
    """Sample *fps* frames per second from *video_path*, up to
    *max_frames* total. Each frame goes into
    ``output_dir/{prefix}frame_{t:.2f}s.jpg``. Returns the list of
    frame paths in chronological order.

    Used by the Evaluator for multi-frame before/after comparison —
    a single keyframe can't distinguish lip movement or other
    motion-dependent edits.
    """
    # Probe duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        duration = float((probe.stdout or "0").strip())
    except ValueError:
        duration = 0.0
    if duration <= 0.0:
        return []

    # Distribute sample times evenly across the clip.
    # Leave a 50 ms tail safety margin — ffmpeg can fail when ``-ss``
    # lands in the last few ms of a clip if the container's declared
    # duration slightly exceeds decodable frames (common with generated
    # video and lipsync outputs).
    #
    # Adaptive step: if 2 fps would yield more than max_frames samples
    # (long clips), widen the step so the full duration is covered —
    # otherwise the trailing half of the clip is never evaluated,
    # which misses late-developing edits like an action ramp-up.
    safety_tail = 0.05
    usable = max(0.0, duration - safety_tail)
    step = max(1.0 / max(fps, 0.01), usable / max(max_frames, 1))
    times: list[float] = []
    t = step / 2.0
    while t < usable and len(times) < max_frames:
        times.append(round(t, 3))
        t += step
    if not times:
        times = [round(max(0.0, usable / 2.0), 3)]

    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for ts in times:
        try:
            frames.append(extract_keyframe(
                video_path, output_dir, time_sec=ts, prefix=prefix,
            ))
        except Exception as exc:
            logger.warning(
                "[Preprocess] frame @ %.2fs failed: %s — skipping",
                ts, exc,
            )
    return frames


def extract_keyframes(video_path: Path, output_dir: Path, meta: VideoMeta) -> list[Path]:
    """
    Smart multi-frame sampling based on video duration.
      <3s  → 1 frame (middle)
      3-10s → 2 frames (start, middle)
      >10s → 3 frames (start, middle, end)
    """
    d = meta.duration
    if d < 3:
        times = [min(0.5, d * 0.5)]
    elif d < 10:
        times = [0.5, d * 0.5]
    else:
        times = [0.5, d * 0.5, max(d - 0.5, d * 0.5)]

    frames = [extract_keyframe(video_path, output_dir, t) for t in times]
    logger.info("Extracted %d keyframe(s) from %s", len(frames), video_path.name)
    return frames


# ── Caption preview (small video WITH audio for VLM captioning) ───────────

CAPTION_PREVIEW_MAX_WIDTH = 720
CAPTION_PREVIEW_VIDEO_BITRATE = "1500k"
CAPTION_PREVIEW_AUDIO_BITRATE = "128k"


def _make_caption_preview(video_path: Path, output_dir: Path) -> Path | None:
    """
    Create a small preview video that keeps audio for VLM captioning.
    Without this, the captioner receives a video-only file and cannot
    analyse audio content (BGM, speech, ambient sounds).
    """
    preview_path = output_dir / f"{video_path.stem}_caption_preview.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"scale='min({CAPTION_PREVIEW_MAX_WIDTH},iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast",
        "-b:v", CAPTION_PREVIEW_VIDEO_BITRATE,
        "-c:a", "aac", "-b:a", CAPTION_PREVIEW_AUDIO_BITRATE, "-ac", "2",
        "-movflags", "+faststart",
        str(preview_path),
    ]
    try:
        _run(cmd, "caption_preview")
        logger.info("Caption preview (with audio) -> %s", preview_path)
        return preview_path
    except Exception as exc:
        logger.warning("Failed to create caption preview: %s - using original", exc)
        return None


# ── Public entry point ─────────────────────────────────────────────────────

def preprocess(video_path: Path, workspace: Path) -> PreprocessResult:
    """
    Full preprocessing pipeline:
      1. probe metadata
      2. extract audio (if present)
      3. create video-only file
      4. upscale / downscale if resolution outside [720, 2160] width range
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    session_dir = workspace / "preprocess"
    session_dir.mkdir(parents=True, exist_ok=True)

    meta = probe_video(video_path)
    logger.info(
        "Video info: %dx%d, %.2f fps, %.2f s, codec=%s",
        meta.width, meta.height, meta.fps, meta.duration, meta.codec,
    )

    audio_path = extract_audio(video_path, session_dir)
    video_only = strip_audio(video_path, session_dir)

    # HDR → SDR conversion for cloud video editor compatibility.
    if _is_hdr(video_only):
        logger.info("HDR video detected — converting to SDR")
        video_only = tonemap_hdr_to_sdr(video_only, session_dir)
        meta = probe_video(video_only)

    # Auto-scale to meet cloud video editor resolution requirements.
    video_only, meta = upscale_if_needed(video_only, meta, session_dir)

    # Create a low-res preview WITH audio for captioning (so VLM can hear)
    caption_preview = _make_caption_preview(video_path, session_dir)

    return PreprocessResult(
        video_path=video_only,
        audio_path=audio_path,
        meta=meta,
        has_audio=audio_path is not None,
        original_video=caption_preview or video_path,
    )
