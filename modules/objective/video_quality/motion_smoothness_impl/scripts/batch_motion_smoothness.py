#!/usr/bin/env python3
"""Batch motion smoothness scoring using AMT-G frame interpolation.

The metric measures the mean absolute difference (d, in [0, 255]) between the
real middle frames of a video and the AMT-interpolated middle frames produced
from their neighboring frames. A smoother video -> smaller d -> higher score.

This script is self-contained: it loads the AMT model from a config + checkpoint,
processes every video in `--video_dir`, and writes a JSON summary to `--output`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, List

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

# Make the bundled AMT source package importable. The AMT code uses absolute
# imports rooted at `quality.amt.*` (e.g. `from quality.amt.networks.blocks.raft`),
# and the package is physically placed at:
#   <project_root>/models/video_quality/motion_smoothness/amt/
# So we register a virtual `quality` package whose path points to the
# motion_smoothness model directory, making `import quality.amt.xxx` resolve
# to the bundled `amt/` folder.
# Script lives at <project_root>/modules/objective/video_quality/motion_smoothness_impl/scripts/
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[4]
_MS_MODEL_DIR = _PROJECT_ROOT / "models" / "video_quality" / "motion_smoothness"
if not (_MS_MODEL_DIR / "amt").is_dir():
    # Fallback: search a few candidate locations.
    for _cand in (_HERE, _HERE.parents[1], _PROJECT_ROOT):
        if (_cand / "amt").is_dir():
            _MS_MODEL_DIR = _cand
            break

import importlib.util
import types

def _register_quality_alias(pkg_root: Path) -> None:
    """Register a `quality` namespace package pointing at `pkg_root` so that
    `quality.amt.*` imports resolve to `<pkg_root>/amt/`."""
    if "quality" in sys.modules and getattr(sys.modules["quality"], "__path__", None):
        # If real quality package exists elsewhere, just append our path.
        sys.modules["quality"].__path__.append(str(pkg_root))
        return
    pkg = types.ModuleType("quality")
    pkg.__path__ = [str(pkg_root)]
    sys.modules["quality"] = pkg

if (_MS_MODEL_DIR / "amt").is_dir():
    _register_quality_alias(_MS_MODEL_DIR)
    sys.path.insert(0, str(_MS_MODEL_DIR))

from quality.amt.utils.utils import img2tensor, tensor2img, check_dim_and_resize, InputPadder  # noqa: E402
from quality.amt.utils.build_utils import build_from_cfg  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def iter_videos(video_dir: Path) -> List[Path]:
    return sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


class FrameProcess:
    """Frame loading helpers mirroring the reference implementation."""

    @staticmethod
    def get_frames(video_path: str) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames extracted from {video_path}")
        return frames

    @staticmethod
    def extract_frame(frame_list: List[np.ndarray], start_from: int = 0) -> List[np.ndarray]:
        # Take every other frame starting from `start_from` (matches reference).
        return frame_list[start_from::2]


class MotionSmoothness:
    """AMT-based motion smoothness evaluator.

    Steps:
        1. Load every frame of the video.
        2. Keep every other frame starting at index 0 -> inputs[0, 2, 4, ...].
        3. Use AMT to interpolate the middle frame between each consecutive pair.
        4. Compare interpolated middle frames against the *real* middle frames
           (i.e. frames at odd indices 1, 3, 5, ... of the original video).
        5. d = mean abs difference in [0, 255]; score = (255 - d) / 255.
    """

    def __init__(self, config: str, ckpt: str, device: str = "cuda", niters: int = 1):
        self.device = device
        self.config = config
        self.ckpt = ckpt
        self.niters = niters
        self.model = None
        self.fp = FrameProcess()
        self._init_runtime()
        self._load_model()

    def _init_runtime(self) -> None:
        if self.device == "cuda" and torch.cuda.is_available():
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024 ** 2
            self.anchor_memory_bias = 2500 * 1024 ** 2
            self.vram_avail = torch.cuda.get_device_properties(0).total_memory
            logger.info("VRAM available: {:.1f} MB".format(self.vram_avail / 1024 ** 2))
        else:
            self.anchor_resolution = 8192 * 8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1
        self.embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(self.device)

    def _load_model(self) -> None:
        if not self.config or not os.path.exists(self.config):
            raise FileNotFoundError(f"AMT config not found: {self.config}")
        if not self.ckpt or not os.path.exists(self.ckpt):
            raise FileNotFoundError(f"AMT checkpoint not found: {self.ckpt}")
        network_cfg = OmegaConf.load(self.config).network
        network_name = network_cfg.name
        logger.info(f"Loading [{network_name}] from [{self.ckpt}]...")
        self.model = build_from_cfg(network_cfg)
        state = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        # Support both raw state_dict and wrapped {'state_dict': ...}.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("AMT model loaded successfully")

    def motion_score(self, video_path: str) -> float:
        if self.model is None:
            raise RuntimeError("AMT model is not loaded")

        frames = self.fp.get_frames(video_path)
        # Need at least 3 original frames so we have >=1 (in_0, middle, in_1) triplet.
        if len(frames) < 3:
            raise RuntimeError(f"motion smoothness needs >=3 frames, got {len(frames)}: {video_path}")

        frame_list = self.fp.extract_frame(frames, start_from=0)
        inputs = [img2tensor(f).to(self.device) for f in frame_list]
        if len(inputs) < 2:
            raise RuntimeError(f"need >=2 sampled frames after extraction, got {len(inputs)}")

        inputs = check_dim_and_resize(inputs)
        h, w = inputs[0].shape[-2:]
        scale = self.anchor_resolution / (h * w) * np.sqrt(
            (self.vram_avail - self.anchor_memory_bias) / self.anchor_memory
        )
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        if scale < 1:
            logger.debug(f"VRAM-limited, video scaled by {scale:.2f}")

        padding = int(16 / scale)
        padder = InputPadder(inputs[0].shape, padding)
        inputs = padder.pad(*inputs)

        # Iteratively interpolate middle frames.
        for _ in range(self.niters):
            outputs = [inputs[0]]
            for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                in_0 = in_0.to(self.device)
                in_1 = in_1.to(self.device)
                with torch.no_grad():
                    imgt_pred = self.model(in_0, in_1, self.embt, scale_factor=scale, eval=True)['imgt_pred']
                outputs += [imgt_pred.cpu(), in_1.cpu()]
            inputs = outputs

        outputs = padder.unpad(*outputs)
        outputs = [tensor2img(o) for o in outputs]
        vfi_score = self._vfi_score(frames, outputs)
        return float((255.0 - vfi_score) / 255.0)

    def _vfi_score(self, ori_frames: List[np.ndarray], interpolate_frames: List[np.ndarray]) -> float:
        # Real middle frames = odd indices of the original full frame list.
        ori = self.fp.extract_frame(ori_frames, start_from=1)
        interpolate = self.fp.extract_frame(interpolate_frames, start_from=1)
        n = min(len(ori), len(interpolate))
        if n == 0:
            return 255.0
        diffs = [self._abs_diff(ori[i], interpolate[i]) for i in range(n)]
        return float(np.mean(diffs))

    @staticmethod
    def _abs_diff(img1: np.ndarray, img2: np.ndarray) -> float:
        if img1.shape != img2.shape:
            # Resize to match if shapes differ (safety).
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return float(np.mean(cv2.absdiff(img1, img2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch motion smoothness scoring with AMT-G")
    parser.add_argument("--video_dir", required=True, help="Directory containing videos to evaluate")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--config", required=True, help="AMT config YAML (e.g. AMT-G.yaml)")
    parser.add_argument("--checkpoint", required=True, help="AMT checkpoint (e.g. amt-g.pth)")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--niters", type=int, default=1, help="interpolation iterations")
    args = parser.parse_args()

    video_dir = Path(args.video_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    videos = iter_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"no videos found in {video_dir}")

    evaluator = MotionSmoothness(args.config, args.checkpoint, device=args.device, niters=args.niters)

    results = []
    scores: List[float] = []
    for video_path in videos:
        try:
            score = evaluator.motion_score(str(video_path))
            scores.append(score)
            results.append({
                "file": str(video_path),
                "filename": video_path.name,
                "score": score,
                "error": None,
            })
            logger.info(f"{video_path.name}: score={score:.4f}")
        except Exception as exc:
            logger.error(f"failed on {video_path.name}: {exc}")
            results.append({
                "file": str(video_path),
                "filename": video_path.name,
                "score": None,
                "error": str(exc),
            })

    payload = {
        "metric": "motion_smoothness",
        "backend": "amt_frame_interpolation",
        "model_config": str(Path(args.config).expanduser().resolve()),
        "model_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "summary": {
            "mean_score": float(mean(scores)) if scores else None,
            "total_samples": len(results),
            "successful_samples": len(scores),
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Saved motion smoothness results to {output_path}")


if __name__ == "__main__":
    main()
