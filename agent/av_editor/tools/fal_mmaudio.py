"""
fal_mmaudio.py - MMAudio V2 via fal.ai.

fal's MMAudio endpoint returns an MP4 with generated audio. The AVE
pipeline expects an audio file from the audio tool, so this wrapper
downloads the MP4 and extracts its audio track to WAV.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import httpx

from av_editor.tools.base import BaseTool, ToolResult
from av_editor.tools.fal_common import FalQueueClient, download_url, extract_audio_wav

logger = logging.getLogger(__name__)

MMAUDIO_SUPPORTED_ACTIONS: set[str] = {
    "audio_add_sfx",
    "audio_add_ambient",
    "audio_replace_sfx",
    "audio_replace_bgm",
    "audio_generate",
}


class FalMMAudioTool(BaseTool):
    """MMAudio V2 audio generation through fal.ai."""

    name = "fal_mmaudio"
    actions = MMAUDIO_SUPPORTED_ACTIONS
    is_audio_tool = True

    MODEL_ID = "fal-ai/mmaudio-v2"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.fal = FalQueueClient(api_key, timeout=600)

    async def execute(
        self,
        video_path: Path,
        action: str,
        params: dict[str, Any],
        output_dir: Path,
    ) -> ToolResult:
        prompt = params.get("prompt", params.get("description", "")).strip()
        if not prompt:
            return ToolResult(success=False, error_msg="No prompt/description provided for audio generation")
        if not video_path.exists():
            return ToolResult(success=False, error_msg=f"video_path not found: {video_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex[:8]
        generated_video = output_dir / f"fal_mmaudio_{tag}.mp4"
        output_audio = output_dir / f"fal_mmaudio_{tag}.wav"

        duration = min(max(float(params.get("duration", 8.0)), 1.0), 30.0)
        payload: dict[str, Any] = {
            "video_url": await self.fal.upload_file(video_path),
            "prompt": prompt,
            "negative_prompt": params.get("negative_prompt", ""),
            "num_steps": int(params.get("num_steps", params.get("num_inference_steps", 25))),
            "duration": duration,
            "cfg_strength": float(params.get("cfg_strength", params.get("guidance_scale", 4.5))),
            "mask_away_clip": bool(params.get("mask_away_clip", False)),
        }
        if "seed" in params:
            payload["seed"] = int(params["seed"])

        logger.info("[FalMMAudio] action=%s duration=%.2fs prompt=%r", action, duration, prompt)

        async with httpx.AsyncClient(timeout=600) as client:
            try:
                result_data = await self.fal.run(self.MODEL_ID, payload, client, poll_interval=3)
                video_info = result_data.get("video", {})
                out_url = video_info.get("url", "")
                if not out_url:
                    raise RuntimeError(f"No video.url in fal result: {result_data}")
                await download_url(out_url, generated_video, client)
                extract_audio_wav(generated_video, output_audio)
                return ToolResult(success=True, output_path=output_audio, raw_response=result_data)
            except Exception as exc:
                logger.error("[FalMMAudio] failed: %s", exc)
                return ToolResult(success=False, error_msg=str(exc))
