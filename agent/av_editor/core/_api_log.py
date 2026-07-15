"""Tiny helper: log every API/LLM call's prompt at INFO level.

All components funnel their prompts through `log_prompt()` so a single
`grep "[API]"` over the run log shows every request the agent made.

Format keeps it greppable without flooding context: component name,
optional model, truncated user/text payload, optional extras.
"""

from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger("av_editor.api")


def log_prompt(
    component: str,
    prompt: str,
    *,
    model: str | None = None,
    extra: dict[str, Any] | None = None,
    max_chars: int = 600,
) -> None:
    """Emit a one-line INFO log capturing an API/LLM call's prompt."""
    if not prompt:
        prompt = ""
    p = prompt.strip().replace("\n", " \\n ")
    if len(p) > max_chars:
        p = p[:max_chars] + f" ... [+{len(prompt) - max_chars} chars]"
    parts = [f"[API] {component}"]
    if model:
        parts.append(f"model={model}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v!r}")
    parts.append(f"prompt={p!r}")
    _logger.info(" ".join(parts))
