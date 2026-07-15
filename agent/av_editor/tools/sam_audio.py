"""
sam_audio.py - Audio separation tool using Meta's SAM Audio model.

SAM Audio (Segment Anything in Audio) supports text-prompted audio
separation: describe what you want to isolate (e.g. "vocals", "rain
sounds", "dog barking") and the model separates target vs residual.

This tool runs locally on GPU.  The model is loaded lazily on first
use and cached for subsequent calls.

Usage in the pipeline:
    Preprocessor extracts original audio → SAM Audio separates it into
    target (to discard/replace) and residual (to keep) → Postprocessor
    mixes the kept audio with MMAudio-generated audio.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from av_editor.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# ── Supported actions ──────────────────────────────────────────────────────

SAM_AUDIO_ACTIONS = {
    "audio_remove",          # remove a specific sound (keep residual)
    "audio_isolate_vocals",  # shortcut: isolate vocals
    "audio_remove_vocals",   # shortcut: remove vocals, keep rest
}

# ── Lazy model singleton ──────────────────────────────────────────────────

_model = None
_processor = None


def _load_model(model_name: str = "facebook/sam-audio-large", device: str = "cuda"):
    """Load SAM Audio model (lazy, cached globally)."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from sam_audio import SAMAudio, SAMAudioProcessor

    logger.info("[SAMAudio] Loading model '%s' on %s...", model_name, device)
    _processor = SAMAudioProcessor.from_pretrained(model_name)
    _model = SAMAudio.from_pretrained(model_name).to(device).eval()
    logger.info("[SAMAudio] Model loaded successfully.")
    return _model, _processor


# ── Helper: extract audio from video ──────────────────────────────────────

def _extract_audio_wav(video_path: Path, output_dir: Path) -> Path:
    """Extract audio from video as WAV (required by SAM Audio)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{video_path.stem}_audio.wav"
    if wav_path.exists():
        return wav_path
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    logger.info("[SAMAudio] extracted audio → %s", wav_path)
    return wav_path


# ── Tool class ────────────────────────────────────────────────────────────

class SAMAudioTool(BaseTool):
    """
    Audio separation tool using Meta SAM Audio.

    Params expected from Planner:
        description : str   – text prompt describing the sound to separate
                              e.g. "background music", "rain sounds", "vocals"
        mode        : str   – "keep" (keep the described sound, discard rest)
                              or "remove" (remove the described sound, keep rest)
                              Default: "remove"
        model       : str   – model variant (default: "facebook/sam-audio-large")
        reranking   : int   – number of reranking candidates (default: 0, set 8 for best quality)
    """

    name = "sam_audio"
    actions = SAM_AUDIO_ACTIONS
    is_audio_separation_tool = True

    def __init__(
        self,
        model_name: str = "facebook/sam-audio-large",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device

    async def execute(
        self,
        video_path: Path,
        action: str,
        params: dict[str, Any],
        output_dir: Path,
    ) -> ToolResult:
        """
        Separate audio from the video using a text prompt.

        Returns ToolResult where output_path points to a dict-like structure:
            - target.wav   : the separated target sound
            - residual.wav : everything else
        The output_path points to the *kept* audio (based on mode).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve description and mode from action shortcuts
        if action == "audio_isolate_vocals":
            description = "vocals"
            mode = "keep"
        elif action == "audio_remove_vocals":
            description = "vocals"
            mode = "remove"
        else:
            description = params.get("description", params.get("prompt", ""))
            mode = params.get("mode", "remove")

        if not description:
            return ToolResult(
                success=False,
                error_msg="No 'description' provided for audio separation",
            )

        reranking = params.get("reranking", 0)

        logger.info(
            "[SAMAudio] action=%s, description='%s', mode=%s, reranking=%d",
            action, description, mode, reranking,
        )

        try:
            result_paths = await asyncio.to_thread(
                self._separate,
                video_path=video_path,
                description=description,
                mode=mode,
                reranking=reranking,
                output_dir=output_dir,
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
            logger.error("[SAMAudio] separation failed: %s", exc)
            return ToolResult(success=False, error_msg=str(exc))

    def _separate(
        self,
        video_path: Path,
        description: str,
        mode: str,
        reranking: int,
        output_dir: Path,
    ) -> dict[str, Path]:
        """Run SAM Audio separation (synchronous, runs in thread)."""
        import torch
        import torchaudio

        model, processor = _load_model(self.model_name, self.device)

        # Extract audio from video as WAV
        wav_path = _extract_audio_wav(video_path, output_dir)

        # Prepare inputs
        inputs = processor(
            audios=[str(wav_path)],
            descriptions=[description],
        ).to(self.device)

        # Run separation
        with torch.inference_mode():
            kwargs = {"predict_spans": True}
            if reranking > 0:
                kwargs["reranking_candidates"] = reranking
            result = model.separate(inputs, **kwargs)

        # Save outputs
        target_path = output_dir / "target.wav"
        residual_path = output_dir / "residual.wav"

        torchaudio.save(
            str(target_path),
            result.target[0].unsqueeze(0).cpu(),
            processor.audio_sampling_rate,
        )
        torchaudio.save(
            str(residual_path),
            result.residual[0].unsqueeze(0).cpu(),
            processor.audio_sampling_rate,
        )

        logger.info("[SAMAudio] target → %s", target_path)
        logger.info("[SAMAudio] residual → %s", residual_path)

        # Determine which audio to keep
        if mode == "keep":
            kept_path = target_path    # keep the described sound
        else:
            kept_path = residual_path  # remove the described sound, keep rest

        return {
            "target": target_path,
            "residual": residual_path,
            "kept": kept_path,
        }
