"""
audiosep.py - Audio separation tool using AudioSep (Audio-AGI/AudioSep).

AudioSep uses text-queried audio separation: given a text description and
a mixture audio, it separates the target sound from the mixture.

For "remove" mode: residual = original - target (simple spectral subtraction).
For "keep" mode:   returns the target directly.

Requires:
    - AUDIOSEP_REPO pointing to a local AudioSep checkout.
    - AUDIOSEP_CKPT pointing to the AudioSep checkpoint downloaded from
      OpenAudioLab/AudioSep on HuggingFace.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from av_editor.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

AUDIOSEP_ACTIONS = {
    "audio_remove",    # remove the described sound, keep the rest
    "audio_keep",      # keep only the described sound
}

# Local AudioSep paths. Keep these configurable for public releases.
AUDIOSEP_REPO = Path(os.getenv("AUDIOSEP_REPO", "AudioSep")).expanduser()
AUDIOSEP_CKPT = Path(os.getenv("AUDIOSEP_CKPT", "audiosep_base.ckpt")).expanduser()
AUDIOSEP_CFG = Path(
    os.getenv("AUDIOSEP_CFG", str(AUDIOSEP_REPO / "config" / "audiosep_base.yaml"))
).expanduser()

# Lazy model singleton (loaded once per process)
_model = None
_model_device = None


def _load_model(device: str = "cuda"):
    global _model, _model_device
    if _model is not None and _model_device == device:
        return _model

    import sys
    if str(AUDIOSEP_REPO) not in sys.path:
        sys.path.insert(0, str(AUDIOSEP_REPO))

    from pipeline import build_audiosep

    logger.info("[AudioSep] Loading model from %s on %s...", AUDIOSEP_CKPT, device)
    _model = build_audiosep(
        config_yaml=str(AUDIOSEP_CFG),
        checkpoint_path=str(AUDIOSEP_CKPT),
        device=device,
    )
    _model_device = device
    logger.info("[AudioSep] Model loaded.")
    return _model


def _extract_audio_wav(video_path: Path, output_dir: Path, sample_rate: int = 32000) -> Path:
    """Extract audio from video as mono WAV at the AudioSep native sample rate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{video_path.stem}_audio.wav"
    if wav_path.exists():
        return wav_path
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    logger.info("[AudioSep] extracted audio → %s", wav_path)
    return wav_path


def _separate(
    model,
    audio_path: Path,
    text: str,
    output_dir: Path,
    device: str = "cuda",
) -> dict[str, Path]:
    """
    Run AudioSep separation.

    Returns dict with keys: target, residual.
    - target  : the audio matching `text`
    - residual: original minus target (what remains)
    """
    import sys
    import torch
    import torchaudio
    import numpy as np
    if str(AUDIOSEP_REPO) not in sys.path:
        sys.path.insert(0, str(AUDIOSEP_REPO))
    from pipeline import separate_audio

    target_path = output_dir / "target.wav"
    separate_audio(
        model=model,
        audio_file=str(audio_path),
        text=text,
        output_file=str(target_path),
        device=device,
        use_chunk=False,
    )

    # Compute residual = original - target (clipped to [-1, 1])
    orig, sr_o = torchaudio.load(str(audio_path))
    tgt, sr_t  = torchaudio.load(str(target_path))

    # Resample target to match original if needed
    if sr_t != sr_o:
        tgt = torchaudio.functional.resample(tgt, sr_t, sr_o)

    # Align lengths
    min_len = min(orig.shape[-1], tgt.shape[-1])
    orig = orig[..., :min_len]
    tgt  = tgt[..., :min_len]

    residual_wav = torch.clamp(orig - tgt, -1.0, 1.0)

    residual_path = output_dir / "residual.wav"
    torchaudio.save(str(residual_path), residual_wav, sr_o)
    logger.info("[AudioSep] target → %s | residual → %s", target_path, residual_path)

    return {"target": target_path, "residual": residual_path}


class AudioSepTool(BaseTool):
    """
    Text-queried audio separation using AudioSep.

    Params expected from Planner:
        description : str  – text describing the sound to separate/remove
                             e.g. "dog barking", "rain", "background music"
        mode        : str  – "remove" (keep residual) or "keep" (keep target)
                             Default: "remove"
    """

    name = "audiosep"
    actions = AUDIOSEP_ACTIONS
    is_audio_separation_tool = True

    def __init__(self, device: str = "cuda"):
        self.device = device

    async def execute(
        self,
        video_path: Path,
        action: str,
        params: dict[str, Any],
        output_dir: Path,
    ) -> ToolResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        description = params.get("description", params.get("prompt", ""))
        mode = "keep" if action == "audio_keep" else params.get("mode", "remove")

        if not description:
            return ToolResult(success=False, error_msg="No 'description' provided for audio separation")

        logger.info("[AudioSep] action=%s description='%s' mode=%s", action, description, mode)

        explicit_audio = params.get("audio_path")

        try:
            result_paths = await asyncio.to_thread(
                self._run_separation,
                video_path=video_path,
                description=description,
                mode=mode,
                output_dir=output_dir,
                audio_path=Path(explicit_audio) if explicit_audio else None,
            )
            return ToolResult(
                success=True,
                output_path=result_paths["kept"],
                raw_response={
                    "target_path": str(result_paths["target"]),
                    "residual_path": str(result_paths["residual"]),
                    "kept_path": str(result_paths["kept"]),
                    "description": description,
                    "mode": mode,
                },
            )
        except Exception as exc:
            logger.error("[AudioSep] failed: %s", exc)
            return ToolResult(success=False, error_msg=str(exc))

    def _run_separation(
        self,
        video_path: Path,
        description: str,
        mode: str,
        output_dir: Path,
        audio_path: Path | None = None,
    ) -> dict[str, Path]:
        model = _load_model(self.device)
        wav_path = audio_path if audio_path else _extract_audio_wav(video_path, output_dir)
        paths = _separate(model, wav_path, description, output_dir, self.device)
        kept = paths["target"] if mode == "keep" else paths["residual"]
        paths["kept"] = kept
        return paths
