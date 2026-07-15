#!/usr/bin/env python3
"""Checklist-based MLLM evaluator for audio-video edits."""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .llm_client import LLMClient
    from .utils import extract_and_encode_audio, extract_and_encode_video
except ImportError:
    from llm_client import LLMClient
    from utils import extract_and_encode_audio, extract_and_encode_video


logger = logging.getLogger(__name__)
PROMPT_DIR = Path(__file__).parent / "prompts"
MODALITIES = ("audio", "video", "general")
DIMENSIONS = ("Edit Response", "Instruction Following", "Fidelity")
SUBDIMENSION_TO_DIMENSION = {
    "edit_response": "Edit Response",
    "instruction_correctness": "Instruction Following",
    "non_edit_preservation": "Fidelity",
    "hallucination_control": "Fidelity",
}


def _score(total: int, yes: int) -> Optional[float]:
    return round(100.0 * yes / total, 2) if total else None


class VideoEditEvaluator:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        output_dir: str = "output",
        num_frames: int = 8,
        use_full_video: bool = True,
    ):
        self.llm_client = llm_client or LLMClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.use_full_video = use_full_video
        self.system_prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        files = {
            "default": "prompt_eval_restructured.md",
            "video_only": "prompt_eval_video_only.md",
            "audio_only": "prompt_eval_audio_only.md",
        }
        prompts = {}
        for category, filename in files.items():
            text = (PROMPT_DIR / filename).read_text(encoding="utf-8")
            match = re.search(
                r'CHECKLIST_EVALUATOR_SYSTEM_PROMPT\s*=\s*"""(.*?)"""',
                text,
                flags=re.DOTALL,
            )
            if not match:
                raise ValueError(f"Missing CHECKLIST_EVALUATOR_SYSTEM_PROMPT in {filename}")
            prompts[category] = match.group(1).strip()
        return prompts

    def _canonical_category(self, value: str) -> str:
        category = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "audio_video_joint_edit": "joint",
            "video_only_edit": "video_only",
            "audio_only_edit": "audio_only",
            "speech_edit": "speech",
        }
        allowed = {"joint", "video_only", "audio_only", "speech"}
        return aliases.get(category, category if category in allowed else "joint")

    def _infer_category(self, edit_type: str, edit_prompt: str) -> str:
        category = self._canonical_category(edit_type)
        if category != "joint" or "joint" in edit_type.lower():
            return category
        prompt = edit_prompt.lower()
        if any(token in prompt for token in ("speech", "speaker", "voice", "say ", "says ", "spoken")):
            return "speech"
        return "joint"

    def _canonical_dimension(self, value: str) -> str:
        key = value.strip().lower().replace("&", "and")
        if key in {"edit response", "response"}:
            return "Edit Response"
        if key in {"instruction following", "instruction correctness"}:
            return "Instruction Following"
        if key in {"fidelity", "fidelity preserving", "fidelity and preservation", "fidelity preservation"}:
            return "Fidelity"
        return value.strip()

    def _canonical_subdimension(self, value: str) -> str:
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "response": "edit_response",
            "instruction_following": "instruction_correctness",
            "fidelity": "non_edit_preservation",
            "fidelity_preserving": "non_edit_preservation",
            "hallucination": "hallucination_control",
        }
        return aliases.get(key, key)

    def _canonical_modality(self, value: str) -> str:
        key = value.strip().lower().replace("-", "_")
        aliases = {
            "visual": "video",
            "vision": "video",
            "av": "general",
            "audio_visual": "general",
            "audiovisual": "general",
        }
        return aliases.get(key, key)

    def _normalize_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        normalized = []
        seen_ids = set()
        for index, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Checklist item {index} must be an object")
            question = str(item.get("question", "")).strip()
            if not question:
                raise ValueError(f"Checklist item {index} has an empty question")
            question_id = str(item.get("question_id") or f"Q{index:02d}").strip()
            if question_id in seen_ids:
                raise ValueError(f"Duplicate checklist question_id: {question_id}")
            seen_ids.add(question_id)

            subdimension = self._canonical_subdimension(str(item.get("subdimension", "")))
            dimension = self._canonical_dimension(str(item.get("dimension", "")))
            expected_dimension = SUBDIMENSION_TO_DIMENSION.get(subdimension)
            if expected_dimension is None:
                raise ValueError(f"Unsupported checklist subdimension: {subdimension}")
            if dimension and dimension != expected_dimension:
                raise ValueError(
                    f"Question {question_id} has dimension {dimension!r}, expected {expected_dimension!r}"
                )
            modality = self._canonical_modality(str(item.get("modality_tag", "")))
            if modality not in MODALITIES:
                raise ValueError(f"Question {question_id} has invalid modality_tag: {modality!r}")
            normalized.append(
                {
                    "question_id": question_id,
                    "dimension": expected_dimension,
                    "subdimension": subdimension,
                    "modality_tag": modality,
                    "question": question,
                }
            )
        if not normalized:
            raise ValueError("Checklist questions must be a non-empty list")
        return normalized

    def _render_prompt(
        self,
        source_caption: str,
        edit_prompt: str,
        edit_type: str,
        questions: List[Dict[str, str]],
        modality: str,
    ) -> str:
        evidence = {
            "audio": "SOURCE audio and TARGET audio only",
            "video": "SOURCE video and TARGET video only",
            "general": "SOURCE and TARGET video plus their audio",
        }[modality]
        return f"""# Evaluation Input

You are provided with {evidence}.

Source caption:
{source_caption}

Edit instruction:
{edit_prompt}

Edit type:
{edit_type}

Checklist subset:
{json.dumps(questions, ensure_ascii=False, indent=2)}

Answer every item from the supplied media evidence. Keep question_id, dimension,
subdimension, modality_tag, and question unchanged. Return JSON only:
{{
  "discrepancy_analysis": "short evidence summary",
  "evaluations": [
    {{
      "question_id": "Q01",
      "dimension": "Instruction Following",
      "subdimension": "instruction_correctness",
      "modality_tag": "{modality}",
      "question": "...",
      "observation": "short direct evidence",
      "answer": "Yes",
      "justification": "short reason"
    }}
  ]
}}"""

    def _valid_response(self, parsed: Dict[str, Any], questions: List[Dict[str, str]]) -> bool:
        evaluations = parsed.get("evaluations")
        if not isinstance(evaluations, list) or len(evaluations) != len(questions):
            return False
        expected = {item["question_id"] for item in questions}
        returned = {
            str(item.get("question_id", "")).strip()
            for item in evaluations
            if isinstance(item, dict)
        }
        return returned == expected

    def _call_split(
        self,
        system_prompt: str,
        prompt: str,
        content: List[Dict[str, Any]],
        questions: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": prompt}, *content]},
        ]
        last = {}
        for attempt in range(5):
            response = self.llm_client.call(messages, temperature=0)
            parsed = self.llm_client.parse_json_response(response)
            if isinstance(parsed, dict) and self._valid_response(parsed, questions):
                return parsed
            last = parsed if isinstance(parsed, dict) else {}
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
        raise ValueError(f"MLLM response did not cover the checklist subset: {last}")

    def _summarize(self, evaluations: List[Dict[str, str]]) -> Dict[str, Any]:
        dimension_modality = {
            dimension: {
                modality: {"total": 0, "yes": 0, "score": None}
                for modality in MODALITIES
            }
            for dimension in DIMENSIONS
        }
        dimension_stats = {
            "Instruction Following": {"total": 0, "yes": 0, "score": None},
            "Fidelity": {"total": 0, "yes": 0, "score": None},
        }
        response_stats = {
            "total": 0,
            "yes": 0,
            "score": None,
            **{modality: {"total": 0, "yes": 0, "score": None} for modality in MODALITIES},
        }

        for item in evaluations:
            dimension = item["dimension"]
            modality = item["modality_tag"]
            yes = item["answer"] == "Yes"
            cell = dimension_modality[dimension][modality]
            cell["total"] += 1
            cell["yes"] += int(yes)
            if dimension == "Edit Response":
                response_stats["total"] += 1
                response_stats["yes"] += int(yes)
                response_stats[modality]["total"] += 1
                response_stats[modality]["yes"] += int(yes)
            elif dimension in dimension_stats:
                dimension_stats[dimension]["total"] += 1
                dimension_stats[dimension]["yes"] += int(yes)

        for dimension in DIMENSIONS:
            for modality in MODALITIES:
                cell = dimension_modality[dimension][modality]
                cell["score"] = _score(cell["total"], cell["yes"])
        for stats in dimension_stats.values():
            stats["score"] = _score(stats["total"], stats["yes"])
        response_stats["score"] = _score(response_stats["total"], response_stats["yes"])
        for modality in MODALITIES:
            cell = response_stats[modality]
            cell["score"] = _score(cell["total"], cell["yes"])

        instruction_score = dimension_stats["Instruction Following"]["score"]
        fidelity_score = dimension_stats["Fidelity"]["score"]
        editing_intent = (
            round(instruction_score * fidelity_score / 100.0, 2)
            if instruction_score is not None and fidelity_score is not None
            else None
        )
        intent_by_modality = {}
        for modality in MODALITIES:
            instruction = dimension_modality["Instruction Following"][modality]["score"]
            fidelity = dimension_modality["Fidelity"][modality]["score"]
            intent_by_modality[modality] = (
                round(instruction * fidelity / 100.0, 2)
                if instruction is not None and fidelity is not None
                else None
            )

        return {
            "dimension_modality": dimension_modality,
            "checklist_summary": {
                "edit_response": response_stats,
                "dimension_scores": dimension_stats,
                "editing_intent_by_modality": intent_by_modality,
                "editing_intent": editing_intent,
                "primary_score_name": "checklist_summary.editing_intent",
                "primary_score": editing_intent,
            },
        }

    def evaluate_video(
        self,
        source_video_path: str,
        edited_video_path: str,
        source_caption: str,
        edit_prompt: str,
        edit_type: str,
        questions: List[Dict[str, Any]],
        edit_category: str = "joint",
    ) -> Dict[str, Any]:
        if not self.use_full_video:
            raise ValueError("Checklist evaluation requires use_full_video=true")
        normalized = self._normalize_questions(questions)
        grouped = {
            modality: [item for item in normalized if item["modality_tag"] == modality]
            for modality in MODALITIES
        }
        system_prompt = self.system_prompts.get(edit_category, self.system_prompts["default"])

        source_video_b64 = target_video_b64 = source_video_mime = target_video_mime = None
        if grouped["video"] or grouped["general"]:
            source_video_b64, source_video_mime = extract_and_encode_video(source_video_path)
            target_video_b64, target_video_mime = extract_and_encode_video(edited_video_path)

        source_audio_b64 = target_audio_b64 = source_audio_mime = target_audio_mime = None
        if grouped["audio"] or grouped["general"]:
            source_audio_b64, source_audio_mime = extract_and_encode_audio(source_video_path)
            target_audio_b64, target_audio_mime = extract_and_encode_audio(edited_video_path)

        split_outputs = {}
        for modality, subset in grouped.items():
            if not subset:
                continue
            media = []
            if modality in {"video", "general"}:
                media.extend(
                    [
                        {"type": "text", "text": "\n--- Source video ---"},
                        {
                            "type": "input_video",
                            "input_video": {"data": source_video_b64, "format": source_video_mime.split("/")[-1]},
                        },
                        {"type": "text", "text": "\n--- Target video ---"},
                        {
                            "type": "input_video",
                            "input_video": {"data": target_video_b64, "format": target_video_mime.split("/")[-1]},
                        },
                    ]
                )
            if modality in {"audio", "general"}:
                media.extend(
                    [
                        {"type": "text", "text": "\n--- Source audio ---"},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": source_audio_b64, "format": source_audio_mime.split("/")[-1]},
                        },
                        {"type": "text", "text": "\n--- Target audio ---"},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": target_audio_b64, "format": target_audio_mime.split("/")[-1]},
                        },
                    ]
                )
            prompt = self._render_prompt(source_caption, edit_prompt, edit_type, subset, modality)
            split_outputs[modality] = self._call_split(system_prompt, prompt, media, subset)

        returned = {}
        discrepancy = []
        for modality in MODALITIES:
            output = split_outputs.get(modality, {})
            analysis = str(
                output.get("discrepancy_analysis")
                or output.get("visual_discrepancy_analysis")
                or ""
            ).strip()
            if analysis:
                discrepancy.append(f"[{modality}] {analysis}")
            for item in output.get("evaluations", []):
                if isinstance(item, dict):
                    returned[str(item.get("question_id", "")).strip()] = item

        evaluations = []
        for question in normalized:
            answer = returned.get(question["question_id"], {})
            binary = "Yes" if str(answer.get("answer", "")).strip().lower() == "yes" else "No"
            evaluations.append(
                {
                    **question,
                    "observation": str(answer.get("observation", "")).strip(),
                    "answer": binary,
                    "justification": str(answer.get("justification", "")).strip(),
                }
            )

        yes_count = sum(item["answer"] == "Yes" for item in evaluations)
        result = {
            "discrepancy_analysis": "\n".join(discrepancy),
            "evaluations": evaluations,
            "questions": normalized,
            "edit_type": edit_type,
            "media_transport": {"mode": "tri_split_av", "calls": list(split_outputs)},
            "summary": {
                "total_questions": len(evaluations),
                "yes_count": yes_count,
                "no_count": len(evaluations) - yes_count,
                "score": _score(len(evaluations), yes_count),
            },
        }
        result.update(self._summarize(evaluations))
        result["summary"]["score"] = result["checklist_summary"]["editing_intent"]
        result["primary_score_name"] = "checklist_summary.editing_intent"
        result["primary_score"] = result["checklist_summary"]["editing_intent"]
        return result

    def evaluate_with_checklist(
        self,
        checklist_path: str,
        source_video_path: str,
        edited_video_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        checklist_data = json.loads(Path(checklist_path).read_text(encoding="utf-8-sig"))
        questions = checklist_data.get("questions")
        if not isinstance(questions, list) or not questions:
            raise ValueError("Checklist must contain a non-empty top-level questions list")

        edit_type = str(checklist_data.get("edit_type", "")).strip()
        edit_prompt = str(checklist_data.get("edit_prompt", "")).strip()
        edit_category = self._canonical_category(str(checklist_data.get("edit_category", "")))
        if not checklist_data.get("edit_category"):
            edit_category = self._infer_category(edit_type, edit_prompt)

        result = {
            "video_name": Path(edited_video_path).stem,
            "source_video": source_video_path,
            "edited_video": edited_video_path,
            "checklist_used": checklist_path,
            "edit_type": edit_type,
            "edit_category": edit_category,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        try:
            result["evaluation"] = self.evaluate_video(
                source_video_path=source_video_path,
                edited_video_path=edited_video_path,
                source_caption=str(checklist_data.get("caption", "")),
                edit_prompt=edit_prompt,
                edit_type=edit_type,
                questions=questions,
                edit_category=edit_category,
            )
            result["success"] = True
        except Exception as exc:
            logger.exception("Checklist evaluation failed")
            result["success"] = False
            result["error"] = str(exc)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{Path(edited_video_path).stem}_{result['timestamp']}_evaluation.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
