"""Gemini media/text helpers through the official Google API."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import threading
import time
from pathlib import Path
from typing import Any

from av_editor.core._api_log import log_prompt

logger = logging.getLogger(__name__)


_INLINE_LIMIT_BYTES = 20 * 1024 * 1024     # 20 MB — Gemini inline cap

# After uploading a file via the File API, Gemini transitions it
# PROCESSING → ACTIVE asynchronously. Calling generate_content with a
# file still in PROCESSING raises 400 FAILED_PRECONDITION ("File <id>
# is not in an ACTIVE state and usage is not allowed."). Poll until
# ACTIVE before using the file. Empirically video files of 20-30 MB
# take 5-30s to become ACTIVE.
_FILE_ACTIVE_TIMEOUT_SEC = 180.0
_FILE_ACTIVE_POLL_INTERVAL_SEC = 2.0

_client_lock = threading.Lock()
_clients: dict[str, Any] = {}


def get_client(api_key: str = ""):
    """Return a cached official Gemini client."""
    resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not resolved_key:
        raise RuntimeError("Set GEMINI_API_KEY for Gemini calls")
    cache_key = f"api-key:{resolved_key}"
    with _client_lock:
        c = _clients.get(cache_key)
        if c is None:
            from google import genai
            c = genai.Client(api_key=resolved_key)
            _clients[cache_key] = c
        return c


def _parse_data_url(url: str) -> tuple[bytes, str]:
    if not url.startswith("data:") or ";base64," not in url:
        raise ValueError("expected data:<mime>;base64,<payload> URL")
    header, payload = url.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    return base64.b64decode(payload), mime_type


def _mime_for(path: Path) -> str:
    mt = mimetypes.guess_type(str(path))[0]
    if mt:
        return mt
    suffix = path.suffix.lower()
    if suffix in {".aac", ".m4a"}:
        return "audio/aac"
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp4":
        return "video/mp4"
    return "application/octet-stream"


def _wait_until_active(client, file_obj, file_path: Path):
    """Poll the File API until *file_obj* reaches ACTIVE state.

    Returns the (possibly refreshed) File handle. Raises RuntimeError
    if the file enters FAILED state, or if it doesn't become ACTIVE
    within ``_FILE_ACTIVE_TIMEOUT_SEC``.

    Why: ``client.files.upload`` returns immediately after the bytes
    upload, but Gemini still has to ingest+probe the media — it's
    PROCESSING for several seconds (longer for big videos). Calling
    generate_content too early raises:
        400 FAILED_PRECONDITION. The File <id> is not in an ACTIVE
        state and usage is not allowed.
    """
    name = getattr(file_obj, "name", None)
    if not name:
        # Defensive: some SDK builds expose .uri instead. Fall back.
        return file_obj

    # Sample initial state — if already ACTIVE, no poll needed.
    initial_state = getattr(file_obj, "state", None)
    if _state_is_active(initial_state):
        return file_obj
    if _state_is_failed(initial_state):
        raise RuntimeError(
            f"Gemini File upload {name} entered FAILED state immediately"
        )

    start = time.time()
    last_state: Any = initial_state
    while time.time() - start < _FILE_ACTIVE_TIMEOUT_SEC:
        time.sleep(_FILE_ACTIVE_POLL_INTERVAL_SEC)
        try:
            refreshed = client.files.get(name=name)
        except Exception as exc:
            logger.debug(
                "[Gemini] files.get(%s) raised %s — retrying", name, exc,
            )
            continue
        last_state = getattr(refreshed, "state", None)
        if _state_is_active(last_state):
            elapsed = time.time() - start
            logger.info(
                "[Gemini] file %s ACTIVE after %.1fs",
                file_path.name, elapsed,
            )
            return refreshed
        if _state_is_failed(last_state):
            raise RuntimeError(
                f"Gemini File upload {name} entered FAILED state "
                f"({last_state!r})"
            )
    raise RuntimeError(
        f"Gemini File upload {name} did not reach ACTIVE within "
        f"{_FILE_ACTIVE_TIMEOUT_SEC:.0f}s (last state={last_state!r})"
    )


def _state_is_active(state: Any) -> bool:
    """Match either FileState.ACTIVE enum or string 'ACTIVE'."""
    if state is None:
        return False
    name = getattr(state, "name", None)
    if name and name.upper() == "ACTIVE":
        return True
    return str(state).upper().endswith("ACTIVE")


def _state_is_failed(state: Any) -> bool:
    if state is None:
        return False
    name = getattr(state, "name", None)
    if name and name.upper() == "FAILED":
        return True
    return str(state).upper().endswith("FAILED")


def _build_file_part(client, file_path: Path) -> dict[str, Any] | Any:
    """Return either an inline-bytes dict or a remote File handle,
    chosen by file size. The genai SDK accepts both shapes in
    `contents=[...]`."""
    size = file_path.stat().st_size
    if size <= _INLINE_LIMIT_BYTES:
        return {
            "inline_data": {
                "mime_type": _mime_for(file_path),
                "data": file_path.read_bytes(),
            }
        }
    # Large files → upload via the File API. Then BLOCK until the
    # uploaded file reaches ACTIVE state (see _wait_until_active for
    # why) — otherwise the immediately-following generate_content call
    # 400s with FAILED_PRECONDITION and the evaluator hangs in fallback.
    logger.info(
        "[Gemini] file %s is %d MB — uploading via File API",
        file_path.name, size // (1024 * 1024),
    )
    uploaded = client.files.upload(file=str(file_path))
    return _wait_until_active(client, uploaded, file_path)


def _messages_to_gemini_parts(
    messages: list[dict[str, Any]],
    *,
    json_response: bool = False,
) -> tuple[str, list[Any]]:
    """Convert chat.completions-style messages to Gemini content parts.

    Supports the content shapes used in this repo: plain strings, text items,
    image/video data URLs, and input_audio/input_video base64 blobs.
    """
    from google import genai

    system_lines: list[str] = []
    parts: list[Any] = []
    role_prefix = {"user": "USER", "assistant": "ASSISTANT", "system": "SYSTEM"}

    for msg in messages:
        role = str(msg.get("role", "user")).lower()
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                text = content.strip()
                if text:
                    system_lines.append(text)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = str(item.get("text", "")).strip()
                        if text:
                            system_lines.append(text)
            continue

        if isinstance(content, str):
            text = content.strip()
            if text:
                parts.append(genai.types.Part.from_text(
                    text=f"[{role_prefix.get(role, role.upper())}] {text}"
                ))
            continue

        if not isinstance(content, list):
            continue

        parts.append(genai.types.Part.from_text(
            text=f"[{role_prefix.get(role, role.upper())}]"
        ))
        for item in content:
            if not isinstance(item, dict):
                continue
            typ = str(item.get("type", "")).strip()
            if typ == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(genai.types.Part.from_text(text=text))
                continue
            if typ == "image_url":
                url = (item.get("image_url", {}) or {}).get("url")
                if isinstance(url, str) and url.startswith("data:"):
                    raw, mime_type = _parse_data_url(url)
                    parts.append(genai.types.Part.from_bytes(
                        data=raw, mime_type=mime_type
                    ))
                continue
            if typ == "video_url":
                url = (item.get("video_url", {}) or {}).get("url")
                if isinstance(url, str) and url.startswith("data:"):
                    raw, mime_type = _parse_data_url(url)
                    parts.append(genai.types.Part.from_bytes(
                        data=raw, mime_type=mime_type
                    ))
                continue
            if typ == "input_video":
                iv = item.get("input_video", {}) or {}
                b64_data = iv.get("data", "")
                fmt = str(iv.get("format", "mp4")).lower().strip(".")
                if b64_data:
                    raw = base64.b64decode(b64_data)
                    mime_type = "video/mp4" if fmt == "mp4" else f"video/{fmt}"
                    parts.append(genai.types.Part.from_bytes(
                        data=raw, mime_type=mime_type
                    ))
                continue
            if typ == "input_audio":
                ia = item.get("input_audio", {}) or {}
                b64_data = ia.get("data", "")
                fmt = str(ia.get("format", "mp3")).lower().strip(".")
                if b64_data:
                    raw = base64.b64decode(b64_data)
                    mime_type = (
                        "audio/mpeg" if fmt in {"mp3", "mpeg"} else f"audio/{fmt}"
                    )
                    parts.append(genai.types.Part.from_bytes(
                        data=raw, mime_type=mime_type
                    ))
                continue

    if not parts:
        hint = "Return valid JSON only." if json_response else "Respond directly."
        parts.append(genai.types.Part.from_text(text=hint))
    return "\n\n".join(system_lines).strip(), parts


def generate_from_messages(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key: str = "",
    json_response: bool = False,
    temperature: float = 0.0,
    max_output_tokens: int = 9999,
    component: str = "Gemini",
) -> str:
    """Run an official Gemini API call from chat.completions-style messages."""
    text_for_log = " | ".join(
        str(p.get("text", ""))
        for m in messages
        for p in ([{"type": "text", "text": m.get("content", "")}]
                  if isinstance(m.get("content", ""), str)
                  else (m.get("content", []) or []))
        if isinstance(p, dict) and p.get("type") == "text"
    )
    log_prompt(
        component, text_for_log, model=model,
        extra={
            "json": json_response,
        },
    )

    client = get_client(api_key)
    system_instruction, parts = _messages_to_gemini_parts(
        messages, json_response=json_response,
    )

    from google.genai import types as gt

    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if json_response:
        config_kwargs["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=gt.GenerateContentConfig(**config_kwargs),
    )
    text = response.text or ""
    if not text and json_response:
        retry_cfg = dict(config_kwargs)
        retry_cfg.pop("response_mime_type", None)
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=gt.GenerateContentConfig(**retry_cfg),
        )
        text = response.text or ""
    return text


def generate_with_media(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    media_path: Path | None = None,
    media_paths: list[Path] | None = None,
    json_response: bool = False,
    temperature: float = 0.0,
    max_output_tokens: int = 9999,
    component: str = "Gemini",
) -> str:
    """Run one Gemini call with optional media (one or more
    video/audio files) and return the raw text response. Pass either
    `media_path` (single file) or `media_paths` (list, sent in
    order). Logs the prompt via the standard [API] logger so it
    lands in the run log alongside other tool calls.
    """
    paths: list[Path] = []
    if media_path is not None:
        paths.append(media_path)
    if media_paths:
        paths.extend(media_paths)

    log_prompt(
        component, user_text, model=model,
        extra={
            "media": [p.name for p in paths] or None,
            "json": json_response,
        },
    )

    client = get_client(api_key)

    contents: list[Any] = []
    for p in paths:
        contents.append(_build_file_part(client, p))
    contents.append(user_text)

    from google.genai import types as gt

    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "system_instruction": system_prompt,
    }
    if json_response:
        config_kwargs["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=gt.GenerateContentConfig(**config_kwargs),
    )
    text = response.text or ""

    # Diagnostic: log finish_reason + token usage + part count when the
    # text response is suspiciously short. This catches cases where the
    # model's reasoning consumed the output budget (finish_reason=
    # MAX_TOKENS), or content arrived as multiple parts that response.text
    # may not be concatenating, or JSON-mode emitted a truncated stub.
    if len(text) < 50:
        try:
            cands = getattr(response, "candidates", None) or []
            finish = [getattr(c, "finish_reason", None) for c in cands]
            parts_per_cand = [
                len((getattr(getattr(c, "content", None), "parts", None) or []))
                for c in cands
            ]
            usage = getattr(response, "usage_metadata", None)
            usage_str = ""
            if usage is not None:
                usage_str = (
                    f" usage(prompt={getattr(usage, 'prompt_token_count', '?')},"
                    f" thoughts={getattr(usage, 'thoughts_token_count', '?')},"
                    f" output={getattr(usage, 'candidates_token_count', '?')},"
                    f" total={getattr(usage, 'total_token_count', '?')})"
                )
            logger.info(
                "[Gemini] short response from %s len=%d finish=%s parts=%s%s",
                model, len(text), finish, parts_per_cand, usage_str,
            )
        except Exception as exc:
            logger.warning("[Gemini] diagnostic log failed: %s", exc)
    if not text:
        # Empty text happens when (a) finish_reason=SAFETY, (b) max
        # output reached, or (c) json-mode rejection. Log diagnostics
        # then retry once without JSON forcing — Gemini sometimes
        # returns empty when its JSON schema check rejects the hint
        # but will answer fine in text mode.
        try:
            cands = getattr(response, "candidates", None) or []
            finish = [getattr(c, "finish_reason", None) for c in cands]
            safety = [getattr(c, "safety_ratings", None) for c in cands]
            logger.warning(
                "[Gemini] empty text from %s (finish=%s safety=%s). "
                "Retrying without json_response.",
                model, finish, safety,
            )
        except Exception:
            logger.warning("[Gemini] empty text from %s; retrying without json_response.", model)
        if json_response:
            retry_cfg = dict(config_kwargs)
            retry_cfg.pop("response_mime_type", None)
            response2 = client.models.generate_content(
                model=model,
                contents=contents,
                config=gt.GenerateContentConfig(**retry_cfg),
            )
            text = response2.text or ""
    return text


def gemini_with_fallback(
    *,
    gemini_api_key: str,
    primary_model: str = "gemini-3.1-pro-preview",
    fallback_model: str = "gemini-2.5-flash",
    system_prompt: str,
    user_text: str,
    media_paths: list[Path] | None = None,
    json_response: bool = False,
    temperature: float = 0.0,
    max_output_tokens: int = 9999,
    component: str = "Gemini",
) -> str:
    """Call Gemini models through Google's official API, primary then fallback."""
    try:
        return generate_with_media(
            api_key=gemini_api_key,
            model=primary_model,
            system_prompt=system_prompt,
            user_text=user_text,
            media_paths=media_paths,
            json_response=json_response,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            component=f"{component}[gemini-primary={primary_model}]",
        )
    except Exception as exc:
        logger.warning(
            "[Gemini-fallback] primary %s failed (%s) — falling back "
            "to %s", primary_model, exc, fallback_model,
        )
        return generate_with_media(
            api_key=gemini_api_key,
            model=fallback_model,
            system_prompt=system_prompt,
            user_text=user_text,
            media_paths=media_paths,
            json_response=json_response,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            component=f"{component}[gemini-fallback={fallback_model}]",
        )
