#!/usr/bin/env python3
"""Native Gemini client used by the MLLM evaluation pipeline."""

import base64
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai


logger = logging.getLogger(__name__)
DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _safe_error(exc: Exception) -> str:
    return re.sub(r"AIza[A-Za-z0-9_-]+", "AIza***", str(exc))


class LLMClient:
    """Gemini client for checklist and realism evaluation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY for Gemini evaluation.")
        self.default_model = os.getenv("DEFAULT_MODEL", DEFAULT_MODEL)
        self.client = genai.Client(api_key=self.api_key)
        logger.info("Gemini evaluation client initialized: model=%s", self.default_model)

    def _parse_data_url(self, url: str) -> Tuple[bytes, str]:
        match = re.match(r"^data:([^;]+);base64,(.*)$", url, flags=re.DOTALL)
        if not match:
            raise ValueError("Unsupported media URL format; expected data:<mime>;base64,<payload>")
        return base64.b64decode(match.group(2).strip()), match.group(1).strip()

    def _messages_to_gemini(self, messages: List[Dict[str, Any]]) -> Tuple[str, List[Any]]:
        system_lines: List[str] = []
        parts: List[Any] = []
        role_prefix = {"user": "USER", "assistant": "ASSISTANT", "system": "SYSTEM"}

        for message in messages:
            role = str(message.get("role", "user")).lower()
            content = message.get("content", "")

            if role == "system":
                items = content if isinstance(content, list) else [{"type": "text", "text": content}]
                for item in items:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = str(item.get("text", "")).strip()
                        if text:
                            system_lines.append(text)
                continue

            if isinstance(content, str):
                text = content.strip()
                if text:
                    parts.append(genai.types.Part.from_text(text=f"[{role_prefix.get(role, role.upper())}] {text}"))
                continue
            if not isinstance(content, list):
                continue

            parts.append(genai.types.Part.from_text(text=f"[{role_prefix.get(role, role.upper())}]"))
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", "")).strip()
                if item_type == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        parts.append(genai.types.Part.from_text(text=text))
                elif item_type in {"image_url", "video_url"}:
                    media = item.get(item_type, {}) or {}
                    url = media.get("url")
                    if isinstance(url, str) and url.startswith("data:"):
                        raw, mime_type = self._parse_data_url(url)
                        parts.append(genai.types.Part.from_bytes(data=raw, mime_type=mime_type))
                elif item_type == "input_video":
                    video = item.get("input_video", {}) or {}
                    data = video.get("data", "")
                    fmt = str(video.get("format", "mp4")).lower().strip(".")
                    if data:
                        part = genai.types.Part.from_bytes(
                            data=base64.b64decode(data),
                            mime_type="video/mp4" if fmt == "mp4" else f"video/{fmt}",
                        )
                        fps = os.getenv("GEMINI_VIDEO_FPS", "").strip()
                        if fps:
                            part.video_metadata = genai.types.VideoMetadata(fps=float(fps))
                        parts.append(part)
                elif item_type == "input_audio":
                    audio = item.get("input_audio", {}) or {}
                    data = audio.get("data", "")
                    fmt = str(audio.get("format", "mp3")).lower().strip(".")
                    if data:
                        parts.append(
                            genai.types.Part.from_bytes(
                                data=base64.b64decode(data),
                                mime_type="audio/mpeg" if fmt in {"mp3", "mpeg"} else f"audio/{fmt}",
                            )
                        )

        if not parts:
            parts.append(genai.types.Part.from_text(text="Return valid JSON only."))
        return "\n\n".join(system_lines).strip(), parts

    def _call_gemini(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        system_instruction, parts = self._messages_to_gemini(messages)
        config: Dict[str, Any] = {
            "max_output_tokens": int(os.getenv("LLM_MAX_OUTPUT_TOKENS", str(max_tokens))),
            "temperature": temperature,
            "response_mime_type": "application/json",
        }
        thinking = os.getenv("GEMINI_THINKING", "").strip()
        if thinking:
            config["thinking_config"] = (
                {"thinking_budget": int(thinking)}
                if thinking.lstrip("-").isdigit()
                else {"thinking_level": thinking}
            )
        if system_instruction:
            config["system_instruction"] = system_instruction

        response = self.client.models.generate_content(
            model=model,
            contents=parts,
            config=config,
        )
        return (response.text or "").strip()

    def call(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 32000,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        current_model = model or self.default_model
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                result = self._call_gemini(messages, current_model, max_tokens, temperature)
                logger.info("Gemini call succeeded (model=%s, attempt=%s)", current_model, attempt + 1)
                return result
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Gemini call failed (model=%s, attempt=%s/%s): %s",
                    current_model,
                    attempt + 1,
                    max_retries,
                    _safe_error(exc),
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
        raise RuntimeError(f"Gemini evaluation failed for model {current_model}: {_safe_error(last_error or Exception())}")

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"raw_response": response}
        try:
            json_str = response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.rfind("```")
                json_str = response[start:end if end != -1 else len(response)].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.rfind("```")
                json_str = response[start:end if end != -1 else len(response)].strip()
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                result.update(parsed)
            return result
        except Exception as exc:
            start = response.find("{")
            if start != -1:
                depth = 0
                for index, char in enumerate(response[start:], start=start):
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                parsed = json.loads(response[start:index + 1])
                                if isinstance(parsed, dict):
                                    result.update(parsed)
                            except json.JSONDecodeError:
                                pass
                            break
            logger.warning("Failed to parse complete JSON response: %s", exc)
            return result
