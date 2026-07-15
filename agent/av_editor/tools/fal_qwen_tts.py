"""
fal_qwen_tts.py - Qwen3 TTS speech tools via fal.ai.
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

QWEN_TTS_SUPPORTED_ACTIONS: set[str] = {
    "speech_replace_full",
    "speech_clone",
}

QWEN_DESIGN_SUPPORTED_ACTIONS: set[str] = {
    "speech_swap",
}

SUPPORTED_LANGUAGES = {
    "Auto", "Chinese", "English", "German", "Italian", "Portuguese",
    "Spanish", "Japanese", "Korean", "French", "Russian",
}


def _normalise_language(language: str) -> str:
    if not language or language.lower() == "auto":
        return "Auto"
    for item in SUPPORTED_LANGUAGES:
        if item.lower() == language.lower():
            return item
    logger.warning("[FalQwenTTS] unsupported language %r → Auto", language)
    return "Auto"


class FalQwenTTSTool(BaseTool):
    """Voice-clone speech synthesis via fal Qwen3 TTS.

    fal's clone endpoint first creates a speaker embedding; the wrapper then
    calls text-to-speech with that embedding and returns the generated audio.
    """

    name = "fal_qwen_tts"
    actions = QWEN_TTS_SUPPORTED_ACTIONS
    is_audio_tool = False
    is_speech_tool = True

    CLONE_MODEL_ID = "fal-ai/qwen-3-tts/clone-voice/1.7b"
    TTS_MODEL_ID = "fal-ai/qwen-3-tts/text-to-speech/1.7b"

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
        ref_audio = params.get("reference_audio")
        text = (params.get("text") or "").strip()
        if not ref_audio:
            return ToolResult(success=False, error_msg="speech_replace_full requires params['reference_audio']")
        if not text:
            return ToolResult(success=False, error_msg="speech_replace_full requires params['text']")

        ref_audio = Path(ref_audio)
        if not ref_audio.exists():
            return ToolResult(success=False, error_msg=f"reference_audio not found: {ref_audio}")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"fal_qwen_tts_{uuid.uuid4().hex[:8]}.mp3"
        language = _normalise_language(params.get("language", "Auto"))
        reference_text = (params.get("reference_text") or "").strip()

        async with httpx.AsyncClient(timeout=600) as client:
            try:
                audio_url = await self.fal.upload_file(ref_audio)
                clone_payload: dict[str, Any] = {"audio_url": audio_url}
                if reference_text:
                    clone_payload["reference_text"] = reference_text
                clone_result = await self.fal.run(self.CLONE_MODEL_ID, clone_payload, client)
                embedding_url = (clone_result.get("speaker_embedding") or {}).get("url", "")
                if not embedding_url:
                    raise RuntimeError(f"No speaker_embedding.url in fal clone result: {clone_result}")

                tts_payload: dict[str, Any] = {
                    "text": text,
                    "language": language,
                    "speaker_voice_embedding_file_url": embedding_url,
                }
                if reference_text:
                    tts_payload["reference_text"] = reference_text
                if params.get("prompt"):
                    tts_payload["prompt"] = params["prompt"]

                tts_result = await self.fal.run(self.TTS_MODEL_ID, tts_payload, client)
                out_url = (tts_result.get("audio") or {}).get("url", "")
                if not out_url:
                    raise RuntimeError(f"No audio.url in fal TTS result: {tts_result}")
                await download_url(out_url, output_file, client)
                raw = {"clone": clone_result, "tts": tts_result}
                return ToolResult(success=True, output_path=output_file, raw_response=raw)
            except Exception as exc:
                logger.error("[FalQwenTTS] failed: %s", exc)
                return ToolResult(success=False, error_msg=str(exc))


class FalQwenTTSDesignTool(BaseTool):
    """Voice-design speech synthesis via fal Qwen3 TTS."""

    name = "fal_qwen_tts_design"
    actions = QWEN_DESIGN_SUPPORTED_ACTIONS
    is_audio_tool = False
    is_speech_tool = False
    is_speech_design_tool = True

    MODEL_ID = "fal-ai/qwen-3-tts/voice-design/1.7b"

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
        text = (params.get("text") or "").strip()
        voice_description = (params.get("voice_description") or "").strip()
        if not text:
            return ToolResult(success=False, error_msg="speech_swap requires params['text']")
        if not voice_description:
            return ToolResult(success=False, error_msg="speech_swap requires params['voice_description']")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"fal_qwen_design_{uuid.uuid4().hex[:8]}.mp3"
        payload: dict[str, Any] = {
            "text": text,
            "language": _normalise_language(params.get("language", "Auto")),
            "prompt": voice_description,
        }

        async with httpx.AsyncClient(timeout=600) as client:
            try:
                result_data = await self.fal.run(self.MODEL_ID, payload, client)
                out_url = (result_data.get("audio") or {}).get("url", "")
                if not out_url:
                    raise RuntimeError(f"No audio.url in fal voice-design result: {result_data}")
                await download_url(out_url, output_file, client)
                return ToolResult(success=True, output_path=output_file, raw_response=result_data)
            except Exception as exc:
                logger.error("[FalQwenDesign] failed: %s", exc)
                return ToolResult(success=False, error_msg=str(exc))
