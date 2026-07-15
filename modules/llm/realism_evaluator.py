#!/usr/bin/env python3
"""Standalone MLLM evaluator for target-clip realism."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .llm_client import LLMClient
    from .utils import extract_and_encode_audio, extract_and_encode_video
except ImportError:
    from llm_client import LLMClient
    from utils import extract_and_encode_audio, extract_and_encode_video


VISUAL_DIMENSIONS = ("object_integrity", "interaction_physics", "naturalness")
AUDIO_DIMENSIONS = ("AAS", "MTC")
ALL_DIMENSIONS = VISUAL_DIMENSIONS + AUDIO_DIMENSIONS


def _parse_score(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, str) and value.strip().upper() in {"", "NA", "N/A"}):
        return None
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(1, min(5, score))


def _mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def _normalize(raw_score: Optional[float]) -> Optional[float]:
    if raw_score is None:
        return None
    return round((raw_score - 1.0) / 4.0 * 100.0, 2)


class RealismEvaluator:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_path: Optional[str] = None,
    ):
        self.llm_client = llm_client or LLMClient()
        path = Path(prompt_path) if prompt_path else Path(__file__).parent / "prompts" / "prompt_realism.md"
        self.prompt_template = path.read_text(encoding="utf-8")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        parsed = self.llm_client.parse_json_response(response)
        if isinstance(parsed, dict) and any(key in parsed for key in ALL_DIMENSIONS):
            return parsed

        raw = parsed.get("raw_response", response) if isinstance(parsed, dict) else response
        try:
            reparsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return parsed if isinstance(parsed, dict) else {}
        if isinstance(reparsed, list) and reparsed:
            reparsed = reparsed[0]
        return reparsed if isinstance(reparsed, dict) else {}

    def evaluate(
        self,
        edited_video_path: str,
        edit_prompt: str = "",
        caption: str = "",
    ) -> Dict[str, Any]:
        video_b64, video_mime = extract_and_encode_video(edited_video_path)
        prompt = self.prompt_template.replace("{edit_prompt}", edit_prompt).replace("{caption}", caption)
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "\n--- Target video ---"},
            {
                "type": "input_video",
                "input_video": {"data": video_b64, "format": video_mime.split("/")[-1]},
            },
        ]

        audio_included = False
        audio_warning = None
        try:
            audio_b64, audio_mime = extract_and_encode_audio(edited_video_path)
            content.extend(
                [
                    {"type": "text", "text": "\n--- Target audio ---"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": audio_mime.split("/")[-1]},
                    },
                ]
            )
            audio_included = True
        except Exception as exc:
            audio_warning = str(exc)

        response = self.llm_client.call(
            [{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0,
        )
        parsed = self._parse_response(response)
        subscores = {key: _parse_score(parsed.get(key)) for key in ALL_DIMENSIONS}
        if not audio_included:
            for key in AUDIO_DIMENSIONS:
                subscores[key] = None
        missing_visual = [key for key in VISUAL_DIMENSIONS if subscores[key] is None]
        if missing_visual:
            raise ValueError(f"Realism response is missing visual scores: {', '.join(missing_visual)}")

        video_raw = _mean([subscores[key] for key in VISUAL_DIMENSIONS])
        audio_raw = _mean([subscores[key] for key in AUDIO_DIMENSIONS])
        overall_raw = _mean([video_raw, audio_raw])
        result = {
            "success": True,
            "subscores": subscores,
            "raw_scores": {
                "overall": overall_raw,
                "video": video_raw,
                "audio": audio_raw,
            },
            "scores": {
                "overall": _normalize(overall_raw),
                "video": _normalize(video_raw),
                "audio": _normalize(audio_raw),
            },
            "reason": str(parsed.get("reason", "")).strip(),
            "audio_included": audio_included,
        }
        if audio_warning:
            result["warning"] = f"audio unavailable; audio realism omitted: {audio_warning}"
        return result
