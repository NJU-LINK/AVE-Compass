"""
state_tracker.py - Linear state chain for tracking editing progress.

Maintains a list[StateSnapshot], one per executed step.
Supports rollback to any previous snapshot.

Simplified version of Agent Banana's Context Folding:
  - No graph (just a linear chain)
  - Each node stores: step_id, video_path, text description,
    tool call info, eval result, status.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from av_editor.schema import (
    EvalResult,
    StateSnapshot,
    StepStatus,
    ToolCall,
)

logger = logging.getLogger(__name__)


class StateTracker:
    """
    Tracks the editing state across sequential steps.

    The state is a simple ordered list (no graph).
    Each entry corresponds to one executed sub-task.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.snapshots: list[StateSnapshot] = []
        self._snapshot_dir = workspace / "states"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ── record ─────────────────────────────────────────────────────────

    def record(
        self,
        step_id: int,
        video_path: Path,
        description: str,
        tool_call: Optional[ToolCall] = None,
        eval_result: Optional[EvalResult] = None,
        status: StepStatus = StepStatus.SUCCESS,
    ) -> StateSnapshot:
        """Append a new snapshot to the history."""
        snap = StateSnapshot(
            step_id=step_id,
            video_path=video_path,
            description=description,
            tool_call=tool_call,
            eval_result=eval_result,
            status=status,
        )
        self.snapshots.append(snap)
        logger.info("[State] recorded step %d → %s (%s)", step_id, video_path.name, status.value)
        return snap

    # ── query ──────────────────────────────────────────────────────────

    @property
    def latest(self) -> Optional[StateSnapshot]:
        return self.snapshots[-1] if self.snapshots else None

    @property
    def current_video(self) -> Optional[Path]:
        return self.latest.video_path if self.latest else None

    def get(self, step_id: int) -> Optional[StateSnapshot]:
        for s in self.snapshots:
            if s.step_id == step_id:
                return s
        return None

    # ── rollback ───────────────────────────────────────────────────────

    def rollback_to(self, step_id: int) -> Path:
        """
        Rollback to the state after *step_id*.
        Removes all snapshots that come after that step.
        Returns the video path at the rollback point.
        """
        idx = None
        for i, s in enumerate(self.snapshots):
            if s.step_id == step_id:
                idx = i
                break
        if idx is None:
            raise ValueError(f"Step {step_id} not found in state history")

        removed = self.snapshots[idx + 1:]
        self.snapshots = self.snapshots[: idx + 1]
        logger.info("[State] rolled back to step %d, removed %d snapshots",
                     step_id, len(removed))
        return self.snapshots[-1].video_path

    # ── persistence ────────────────────────────────────────────────────

    def save(self) -> Path:
        """Dump the full state chain to a JSON file."""
        out = self._snapshot_dir / "state_chain.json"
        data = []
        for s in self.snapshots:
            entry = {
                "step_id": s.step_id,
                "video_path": str(s.video_path),
                "description": s.description,
                "status": s.status.value,
            }
            if s.tool_call:
                entry["tool"] = {
                    "name": s.tool_call.tool_name,
                    "success": s.tool_call.success,
                    "error": s.tool_call.error_msg,
                }
            if s.eval_result:
                entry["eval"] = {
                    "verdict": s.eval_result.verdict.value,
                    "quality": s.eval_result.quality_score,
                    "consistency": s.eval_result.consistency_score,
                    "reason": s.eval_result.reason,
                }
                if s.eval_result.checklist_scores:
                    entry["eval"]["checklist_scores"] = s.eval_result.checklist_scores
            data.append(entry)

        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("[State] saved %d snapshots → %s", len(data), out)
        return out

    def summary(self) -> str:
        """One-line-per-step summary for logging / LLM context."""
        lines = []
        for s in self.snapshots:
            verdict = ""
            if s.eval_result:
                verdict = f" eval={s.eval_result.verdict.value}(q={s.eval_result.quality_score:.2f})"
            lines.append(
                f"  step {s.step_id}: [{s.status.value}]{verdict} — {s.description}"
            )
        return "\n".join(lines)
