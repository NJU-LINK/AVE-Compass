"""
fal_lipsync.py - Sync Lipsync 2 via fal.ai.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import httpx

from av_editor.tools.base import BaseTool, ToolResult
from av_editor.tools.fal_common import FalQueueClient, download_url

logger = logging.getLogger(__name__)

LIPSYNC_SUPPORTED_ACTIONS: set[str] = {
    "speech_lipsync",
}


class FalLipsyncTool(BaseTool):
    """Audio-driven lipsync via fal Sync Lipsync 2."""

    name = "fal_lipsync"
    actions = LIPSYNC_SUPPORTED_ACTIONS
    is_audio_tool = False
    is_lipsync_tool = True

    MODEL_ID = "fal-ai/sync-lipsync/v2"

    def __init__(self, api_key: str, model: str = "lipsync-2"):
        self.api_key = api_key
        self.model = model
        self.fal = FalQueueClient(api_key, timeout=900)

    async def execute(
        self,
        video_path: Path,
        action: str,
        params: dict[str, Any],
        output_dir: Path,
    ) -> ToolResult:
        audio_path = params.get("audio_path")
        if not audio_path:
            return ToolResult(success=False, error_msg="speech_lipsync requires params['audio_path']")
        audio_path = Path(audio_path)
        if not audio_path.exists():
            return ToolResult(success=False, error_msg=f"audio_path not found: {audio_path}")
        if not video_path.exists():
            return ToolResult(success=False, error_msg=f"video_path not found: {video_path}")

        sync_mode = params.get("sync_mode", "cut_off")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"fal_lipsync_{uuid.uuid4().hex[:8]}.mp4"

        async with httpx.AsyncClient(timeout=900) as client:
            try:
                video_url = await self.fal.upload_file(video_path)
                audio_url = await self.fal.upload_file(audio_path)
                payload: dict[str, Any] = {
                    "model": params.get("model", self.model),
                    "video_url": video_url,
                    "audio_url": audio_url,
                    "sync_mode": sync_mode,
                }
                result_data = await self.fal.run(self.MODEL_ID, payload, client, poll_interval=4)
                out_url = (result_data.get("video") or {}).get("url", "")
                if not out_url:
                    raise RuntimeError(f"No video.url in fal lipsync result: {result_data}")
                await download_url(out_url, output_file, client)
                return ToolResult(success=True, output_path=output_file, raw_response=result_data)
            except Exception as exc:
                logger.error("[FalLipsync] failed: %s", exc)
                return ToolResult(success=False, error_msg=str(exc))
