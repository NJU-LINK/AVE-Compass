"""
fal_wan.py - Wan 2.7 video edit via fal.ai.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any

import httpx

from av_editor.tools.base import BaseTool, ToolResult
from av_editor.tools.fal_common import FalQueueClient, download_url

logger = logging.getLogger(__name__)

WAN_SUPPORTED_ACTIONS = {
    "style_transfer",
    "scene_edit",
    "add_object",
    "remove_object",
    "replace_object",
    "recolor",
    "repainting",
    "depth_modify",
    "motion_edit",
}


class FalWanTool(BaseTool):
    """Wan 2.7 video-to-video edit through fal.ai queue API."""

    name = "fal_wan"
    actions = WAN_SUPPORTED_ACTIONS

    MODEL_ID = "fal-ai/wan/v2.7/edit-video"

    def __init__(self, api_key: str, model_variant: str = "2.7"):
        self.api_key = api_key
        self.model_variant = model_variant
        self.name = f"fal_wan_{model_variant}"
        self.fal = FalQueueClient(api_key, timeout=1800)

    @staticmethod
    def _pick_resolution(video_path: Path) -> str:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=s=x:p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            w_str, h_str = (result.stdout or "").strip().split("x")
            short_edge = min(int(w_str), int(h_str))
        except Exception as exc:
            logger.warning("[FalWan] resolution autodetect failed (%s) - using 720p", exc)
            return "720p"
        return "1080p" if short_edge >= 1080 else "720p"

    def _build_prompt(
        self, action: str, params: dict[str, Any], description: str = "",
    ) -> str:
        if description and description.strip():
            return description.strip()

        p = params
        fallbacks: dict[str, str] = {
            "style_transfer": f"Change the video to {p.get('style', 'stylised')} style.",
            "scene_edit": p.get("style", "Edit the scene."),
            "add_object": f"Add {p.get('object', 'an element')} to the scene.",
            "remove_object": f"Remove {p.get('object', 'the target')}.",
            "replace_object": f"Replace {p.get('object', 'A')} with {p.get('replacement', 'B')}.",
            "recolor": f"Recolor {p.get('target', p.get('region', 'the area'))} to {p.get('color', 'a new colour')}.",
            "repainting": f"Repaint {p.get('region', 'the area')}.",
            "depth_modify": f"Modify the {p.get('layer', 'foreground')} layer.",
            "motion_edit": f"Change {p.get('subject', 'the subject')}'s action to {p.get('motion', 'moving naturally')}.",
        }
        return fallbacks.get(action, "Edit the video.")

    async def execute(
        self,
        video_path: Path,
        action: str,
        params: dict[str, Any],
        output_dir: Path,
    ) -> ToolResult:
        description = params.pop("_description", "")
        prompt = self._build_prompt(action, params, description=description)
        logger.info("[FalWan] action=%s prompt=%r", action, prompt)

        if not video_path.exists():
            return ToolResult(success=False, error_msg=f"video_path not found: {video_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"fal_wan_{action}_{uuid.uuid4().hex[:8]}.mp4"

        async with httpx.AsyncClient(timeout=1800) as client:
            try:
                video_url = await self.fal.upload_file(video_path)
                payload: dict[str, Any] = {
                    "prompt": prompt,
                    "video_url": video_url,
                    "resolution": str(
                        params.get("resolution") or self._pick_resolution(video_path)
                    ),
                    # The pipeline owns final audio. Preserve input audio if
                    # present, but downstream muxing will replace/strip it.
                    "audio_setting": str(params.get("audio_setting", "origin")),
                    "duration": int(params.get("duration", 0)),
                    "enable_safety_checker": bool(
                        params.get("enable_safety_checker", True)
                    ),
                }
                if params.get("reference_image_url"):
                    payload["reference_image_url"] = params["reference_image_url"]
                if params.get("aspect_ratio"):
                    payload["aspect_ratio"] = params["aspect_ratio"]
                if params.get("seed") is not None:
                    payload["seed"] = int(params["seed"])

                result_data = await self.fal.run(
                    self.MODEL_ID, payload, client, poll_interval=5,
                )
                video_info = result_data.get("video", {})
                out_url = video_info.get("url", "")
                if not out_url:
                    raise RuntimeError(f"No video.url in fal result: {result_data}")
                await download_url(out_url, output_file, client)
                return ToolResult(
                    success=True,
                    output_path=output_file,
                    raw_response=result_data,
                )
            except Exception as exc:
                logger.error("[FalWan] failed: %s", exc)
                return ToolResult(success=False, error_msg=str(exc))
