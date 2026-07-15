"""
base.py - Abstract base class for all video editing tools.

Every concrete tool inherits from BaseTool
and implements `execute()`.  This gives the Executor a uniform interface.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standardised return value from any tool."""
    success: bool
    output_path: Path | None = None      # path to the edited video file
    raw_response: dict[str, Any] = field(default_factory=dict)
    error_msg: str = ""


class BaseTool(abc.ABC):
    """
    Abstract video editing tool.

    Subclasses must implement:
      - `name`       : unique tool identifier
      - `actions`    : set of EditAction values this tool can handle
      - `execute()`  : actually call the external API and return a ToolResult
    """

    name: str = "base"
    actions: set[str] = set()

    @abc.abstractmethod
    async def execute(
        self,
        video_path: Path,
        action: str,
        params: dict[str, Any],
        output_dir: Path,
    ) -> ToolResult:
        """
        Run one atomic edit on *video_path* and write the result to *output_dir*.

        Parameters
        ----------
        video_path : Path to the current (video-only) file.
        action     : One of EditAction values.
        params     : Action-specific parameters produced by the Planner.
        output_dir : Directory where the output file should be saved.

        Returns
        -------
        ToolResult with the path to the new video.
        """
        ...

    def supports(self, action: str) -> bool:
        return action in self.actions
