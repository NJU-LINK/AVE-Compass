#!/usr/bin/env python3
"""
run.py - CLI entry point for AVE Agent.

Usage:
    python run.py --video input.mp4 --instruction "Make the clip cyberpunk"
    python run.py --video input.mp4 --instruction "Turn the sky overcast and add rain"

Environment variables required:
    GEMINI_API_KEY : for Gemini Planner/Evaluator/Captioner (AI Studio)
    FAL_KEY        : for Wan / Seedance / MMAudio / Qwen / Lipsync / SAM Audio
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

from av_editor.config import AppConfig, LLMConfig, PipelineConfig, ToolAPIConfig
from av_editor.pipeline import EditingPipeline


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AVE Agent",
    )
    parser.add_argument(
        "--video", "-v", type=str, default=None,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--instruction", "-i", type=str, default=None,
        help="Natural-language editing instruction",
    )
    parser.add_argument(
        "--reuse-session", type=str, default=None, metavar="SESSION_ID",
        help="Reuse an existing session's edited video; skip all video editing "
             "and re-run only the audio stage. Use with --instruction to override "
             "audio tasks, or omit to reuse plan.json audio tasks.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path for the output video (default: workspace/<session>/final_<name>.mp4)",
    )
    parser.add_argument(
        "--workspace", "-w", type=str, default="workspace",
        help="Workspace directory for intermediate files",
    )
    parser.add_argument(
        "--model", "-m", type=str,
        default=os.getenv("GEMINI_PLANNER_MODEL", "gemini-2.5-flash"),
        help="Gemini model name for the Planner",
    )
    parser.add_argument(
        "--video-backend", type=str,
        default=os.getenv("VIDEO_BACKEND", "wan"),
        choices=["wan", "seedance"],
        help="Primary video editor backend: wan or seedance (both fal.ai). "
             "seedance is reference-to-video regeneration, not true V2V edit.",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Run only preprocessing + captioning + planning, then exit",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.reuse_session and not args.video:
        print("error: --video is required unless --reuse-session is specified", file=sys.stderr)
        sys.exit(1)
    if not args.reuse_session and not args.instruction:
        print("error: --instruction is required unless --reuse-session is specified", file=sys.stderr)
        sys.exit(1)

    config = AppConfig(
        llm=LLMConfig(model=args.model),
        tools=ToolAPIConfig(
            video_backend=args.video_backend,
        ),
        pipeline=PipelineConfig(workspace_dir=Path(args.workspace)),
    )

    pipeline = EditingPipeline(config)

    if args.reuse_session:
        result = await pipeline.run_audio_only(
            session_id=args.reuse_session,
            audio_instruction=args.instruction,  # None means reuse plan.json.
        )
    elif args.plan_only:
        result = await pipeline.run(
            video_path=args.video,
            instruction=args.instruction,
            plan_only=True,
        )
    else:
        result = await pipeline.run(
            video_path=args.video,
            instruction=args.instruction,
        )

    if args.output:
        import shutil
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result, out)
        print(f"\nOutput saved to: {out}")
    else:
        print(f"\nOutput: {result}")


if __name__ == "__main__":
    asyncio.run(main())
