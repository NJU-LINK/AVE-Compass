"""
fal_seedance.py - Seedance 2.0 "edit" via fal.ai reference-to-video.

Seedance has NO video-to-video edit endpoint. We approximate an edit by
feeding the SOURCE clip as a reference video (`video_urls`) plus the edit
instruction as the prompt — the model regenerates a clip conditioned on the
source. This preserves rough composition but is generation, not faithful
V2V editing (expect strong realism, weaker edit-fidelity / source-motion).
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any

import httpx

from av_editor.tools.base import BaseTool, ToolResult
from av_editor.tools.fal_common import FalQueueClient, download_url

logger = logging.getLogger(__name__)

SEEDANCE_SUPPORTED_ACTIONS = {
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

_ALLOWED_DURATIONS = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}


class FalSeedanceTool(BaseTool):
    """Seedance 2.0 reference-to-video, used as an approximate editor."""

    name = "fal_seedance"
    actions = SEEDANCE_SUPPORTED_ACTIONS

    # fal_client.subscribe uses the model route without the "fal-ai/" prefix.
    MODEL_ID = "bytedance/seedance-2.0/enterprise/v2/reference-to-video"

    def __init__(self, api_key: str, model_variant: str = "2.0"):
        self.api_key = api_key
        self.model_variant = model_variant
        self.name = f"fal_seedance_{model_variant}"
        self.fal = FalQueueClient(api_key, timeout=1800)

    @staticmethod
    def _probe(video_path: Path) -> tuple[int | None, float | None]:
        """Return (short_edge_px, duration_s) or (None, None) on failure."""
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height:format=duration",
                 "-of", "csv=s=x:p=0", str(video_path)],
                capture_output=True, text=True, timeout=30,
            )
            lines = [x for x in (r.stdout or "").strip().splitlines() if x]
            w, h = lines[0].split("x")[:2]
            dur = float(lines[-1]) if len(lines) > 1 else None
            return min(int(w), int(h)), dur
        except Exception as exc:
            logger.warning("[FalSeedance] ffprobe failed (%s)", exc)
            return None, None

    def _pick_resolution(self, short_edge: int | None) -> str:
        if short_edge is None:
            return "720p"
        return "1080p" if short_edge >= 1080 else "720p"

    def _pick_duration(self, dur: float | None, params: dict[str, Any]) -> Any:
        """Match output length to the source (rounded to an allowed enum);
        fall back to 'auto'. An explicit params['duration'] wins."""
        if params.get("duration"):
            return params["duration"]
        if dur is None:
            return "auto"
        d = max(4, min(15, round(dur)))
        return d if d in _ALLOWED_DURATIONS else "auto"

    def _build_prompt(self, action: str, params: dict[str, Any], description: str = "") -> str:
        base = description.strip() if description and description.strip() else None
        if not base:
            p = params
            base = {
                "style_transfer": f"Render the clip in {p.get('style', 'a new')} style.",
                "scene_edit": p.get("style", "Edit the scene."),
                "add_object": f"Add {p.get('object', 'an element')} to the scene.",
                "remove_object": f"Remove {p.get('object', 'the target')}.",
                "replace_object": f"Replace {p.get('object', 'A')} with {p.get('replacement', 'B')}.",
                "recolor": f"Recolor {p.get('target', p.get('region', 'the area'))} to {p.get('color', 'a new colour')}.",
                "repainting": f"Repaint {p.get('region', 'the area')}.",
                "depth_modify": f"Modify the {p.get('layer', 'foreground')} layer.",
                "motion_edit": f"Change {p.get('subject', 'the subject')}'s action to {p.get('motion', 'moving naturally')}.",
            }.get(action, "Edit the video.")
        # Reference the source clip so Seedance conditions on it.
        return f"Based on @Video1, {base[0].lower()}{base[1:]}"

    async def execute(
        self,
        video_path: Path,
        action: str,
        params: dict[str, Any],
        output_dir: Path,
    ) -> ToolResult:
        description = params.pop("_description", "")
        prompt = self._build_prompt(action, params, description=description)
        logger.info("[FalSeedance] action=%s prompt=%r", action, prompt)

        if not video_path.exists():
            return ToolResult(success=False, error_msg=f"video_path not found: {video_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"fal_seedance_{action}_{uuid.uuid4().hex[:8]}.mp4"
        short_edge, dur = self._probe(video_path)

        try:
            video_url = await self.fal.upload_file(video_path)
            args: dict[str, Any] = {
                "prompt": prompt,
                "video_urls": [video_url],
                "resolution": str(params.get("resolution") or self._pick_resolution(short_edge)),
                "aspect_ratio": params.get("aspect_ratio") or "auto",
                "duration": str(self._pick_duration(dur, params)),
                # The pipeline owns final audio — don't let Seedance add its own.
                "generate_audio": bool(params.get("generate_audio", False)),
            }
            if params.get("bitrate_mode"):
                args["bitrate_mode"] = params["bitrate_mode"]

            result_data = await asyncio.to_thread(self._subscribe, args)
            out_url = (result_data.get("video") or {}).get("url", "")
            if not out_url:
                raise RuntimeError(f"No video.url in fal result: {result_data}")
            async with httpx.AsyncClient(timeout=1800) as client:
                await download_url(out_url, output_file, client)
            return ToolResult(success=True, output_path=output_file, raw_response=result_data)
        except Exception as exc:
            logger.error("[FalSeedance] failed: %s", exc)
            return ToolResult(success=False, error_msg=str(exc))

    def _subscribe(self, args: dict[str, Any]) -> dict[str, Any]:
        """Blocking fal_client.subscribe call (run in a thread)."""
        import os
        import fal_client

        os.environ.setdefault("FAL_KEY", self.api_key)
        return fal_client.subscribe(self.MODEL_ID, arguments=args, with_logs=False)
