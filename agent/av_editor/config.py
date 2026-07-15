"""
config.py - Centralised configuration for AVE Agent.

API keys are read from environment variables.  Workspace paths, retry
limits, model names, etc. live here so every module reads from one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """Settings for the LLM used by Planner / Evaluator."""
    model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_PLANNER_MODEL", "gemini-2.5-flash"
        )
    )
    temperature: float = 0.2
    max_tokens: int = 9999

    # Gemini calls use Google's official Gemini API via google-genai.
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_EVALUATOR_MODEL", "gemini-3.1-pro-preview"
        )
    )


@dataclass
class ToolAPIConfig:
    """API endpoints / keys for external video editing services."""
    # --- fal.ai (Wan / Seedance / MMAudio / Qwen TTS / Lipsync / SAM Audio) ─
    fal_api_key: str = field(
        default_factory=lambda: os.getenv("FAL_KEY", "")
    )

    # Video editor backend. Both options use fal.ai.
    video_backend: str = field(
        default_factory=lambda: os.getenv("VIDEO_BACKEND", "wan")
    )

    # Wan model variant (fal.ai "fal-ai/wan/v2.7/edit-video").
    wan_model_variant: str = field(
        default_factory=lambda: os.getenv("WAN_MODEL_VARIANT", "2.7")
    )

    # Seedance model variant (fal.ai reference-to-video backend).
    seedance_model_variant: str = field(
        default_factory=lambda: os.getenv("SEEDANCE_MODEL_VARIANT", "2.0")
    )

    # --- AudioSep (local audio separation, no API key needed) ───────────────
    audiosep_device: str = field(
        default_factory=lambda: os.getenv("AUDIOSEP_DEVICE", "cuda")
    )


@dataclass
class PipelineConfig:
    """Pipeline-level knobs."""
    workspace_dir: Path = Path("workspace")         # intermediate files go here
    eval_quality_threshold: float = 0.6
    eval_consistency_threshold: float = 0.7


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tools: ToolAPIConfig = field(default_factory=ToolAPIConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def ensure_workspace(self) -> Path:
        self.pipeline.workspace_dir.mkdir(parents=True, exist_ok=True)
        return self.pipeline.workspace_dir
