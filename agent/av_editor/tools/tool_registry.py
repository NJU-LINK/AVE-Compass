"""
tool_registry.py - Central registry that maps EditAction → available tools.

The Executor queries this registry to find the best tool for a given action.
Tools are ranked by priority; the first tool that supports the action is used.
"""

from __future__ import annotations

import logging
from typing import Any

from av_editor.config import ToolAPIConfig
from av_editor.tools.base import BaseTool
from av_editor.tools.fal_wan import FalWanTool
from av_editor.tools.fal_seedance import FalSeedanceTool
from av_editor.tools.fal_mmaudio import FalMMAudioTool
from av_editor.tools.fal_qwen_tts import (
    FalQwenTTSTool,
    FalQwenTTSDesignTool,
)
from av_editor.tools.fal_lipsync import FalLipsyncTool
from av_editor.tools.audiosep import AudioSepTool
from av_editor.tools.fal_samaudio import FalSAMAudioTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Manages all available video editing tools.
    Tools are registered with a priority (lower = preferred).
    """

    def __init__(self) -> None:
        # list of (priority, tool_instance)
        self._tools: list[tuple[int, BaseTool]] = []

    # ── registration ───────────────────────────────────────────────────

    def register(self, tool: BaseTool, priority: int = 100) -> None:
        self._tools.append((priority, tool))
        self._tools.sort(key=lambda t: t[0])
        logger.info("Registered tool '%s' (priority=%d)", tool.name, priority)

    # ── lookup ─────────────────────────────────────────────────────────

    def find(self, action: str) -> BaseTool | None:
        """Return the highest-priority tool that supports *action*."""
        for _, tool in self._tools:
            if tool.supports(action):
                return tool
        return None

    def find_all(self, action: str) -> list[BaseTool]:
        """Return all tools that support *action*, sorted by priority."""
        return [tool for _, tool in self._tools if tool.supports(action)]

    def find_audio_tool(self) -> BaseTool | None:
        """Return the highest-priority audio tool."""
        for _, tool in self._tools:
            if getattr(tool, "is_audio_tool", False):
                return tool
        return None

    def find_audio_separation_tool(self) -> BaseTool | None:
        """Return the highest-priority audio separation tool (SAM Audio)."""
        for _, tool in self._tools:
            if getattr(tool, "is_audio_separation_tool", False):
                return tool
        return None

    def find_speech_tool(self) -> BaseTool | None:
        """Return the highest-priority speech (voice-clone TTS) tool."""
        for _, tool in self._tools:
            if getattr(tool, "is_speech_tool", False):
                return tool
        return None

    def find_speech_design_tool(self) -> BaseTool | None:
        """Return the highest-priority voice-design (identity-swap) tool."""
        for _, tool in self._tools:
            if getattr(tool, "is_speech_design_tool", False):
                return tool
        return None

    def find_lipsync_tool(self) -> BaseTool | None:
        """Return the highest-priority lipsync (V2V mouth re-animation) tool."""
        for _, tool in self._tools:
            if getattr(tool, "is_lipsync_tool", False):
                return tool
        return None

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "priority": p, "actions": sorted(t.actions)}
            for p, t in self._tools
        ]


def build_default_registry(cfg: ToolAPIConfig) -> ToolRegistry:
    """
    Construct the default tool registry from config.
    Agent uses fal-backed tools. Video defaults to fal Wan and can be
    explicitly switched to fal Seedance with video_backend='seedance'.
    """
    registry = ToolRegistry()

    video_backend = getattr(cfg, "video_backend", "wan").lower()
    if video_backend == "wan":
        if cfg.fal_api_key:
            registry.register(
                FalWanTool(
                    api_key=cfg.fal_api_key,
                    model_variant=getattr(cfg, "wan_model_variant", "2.7"),
                ),
                priority=10,
            )
        else:
            logger.warning("FAL_KEY not set — fal Wan video tool disabled")
    elif video_backend == "seedance":
        if cfg.fal_api_key:
            registry.register(
                FalSeedanceTool(
                    api_key=cfg.fal_api_key,
                    model_variant=getattr(cfg, "seedance_model_variant", "2.0"),
                ),
                priority=10,
            )
        else:
            logger.warning("FAL_KEY not set — fal Seedance video tool disabled")
    else:
        raise ValueError(
            "Unsupported video_backend=%r (expected 'wan' or 'seedance')"
            % video_backend
        )

    if cfg.fal_api_key:
        registry.register(FalMMAudioTool(api_key=cfg.fal_api_key), priority=10)
        registry.register(FalQwenTTSTool(api_key=cfg.fal_api_key), priority=10)
        registry.register(FalQwenTTSDesignTool(api_key=cfg.fal_api_key), priority=10)
        registry.register(FalLipsyncTool(api_key=cfg.fal_api_key), priority=10)
    else:
        logger.warning(
            "FAL_KEY not set — fal MMAudio/Qwen/Lipsync tools disabled",
        )

    # Audio separation: fal.ai SAM Audio API (preferred, priority=3)
    if cfg.fal_api_key:
        fal_samaudio = FalSAMAudioTool(api_key=cfg.fal_api_key)
        registry.register(fal_samaudio, priority=3)
    else:
        logger.warning("FAL_KEY not set — fal.ai SAM Audio tool disabled")

    # Audio separation: AudioSep local GPU (fallback, priority=5)
    audiosep = AudioSepTool(device=cfg.audiosep_device)
    registry.register(audiosep, priority=5)

    return registry
