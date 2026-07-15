#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def iter_videos(video_dir: Path) -> list[Path]:
    return sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def sample_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0 or num_frames <= 0:
        return []
    count = min(total_frames, num_frames)
    return sorted({int(round(i)) for i in np.linspace(0, total_frames - 1, count)})


def load_sampled_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[Image.Image] = []
    for index in sample_frame_indices(total, num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    if not frames:
        raise RuntimeError(f"no readable frames: {video_path}")
    return frames


def manual_preprocess(frames: Iterable[Image.Image], image_size: int = 224) -> Any:
    import torch

    tensors = []
    for frame in frames:
        image = frame.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(tensors, dim=0)


def pool_model_output(output: Any) -> Any:
    import torch

    if isinstance(output, torch.Tensor):
        tensor = output
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        tensor = output.pooler_output
    elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        tensor = output.last_hidden_state[:, 0]
    elif isinstance(output, (tuple, list)) and output:
        tensor = output[0]
    else:
        raise RuntimeError(f"unsupported model output type: {type(output)!r}")
    if tensor.ndim > 2:
        tensor = tensor.flatten(1).mean(dim=1, keepdim=False)
    return tensor.float()


class DinoEncoder:
    def __init__(self, model_path: Path, device: str, model_name: str) -> None:
        import torch

        self.torch = torch
        self.device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.processor = None
        self.backend = ""

        if model_path.is_dir() and (model_path / "config.json").exists():
            try:
                from transformers import AutoImageProcessor, AutoModel

                self.processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
                self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(self.device)
                self.backend = "transformers"
                self.model.eval()
                return
            except Exception as exc:
                self._transformers_error = str(exc)

        if model_path.is_file():
            try:
                self.model = torch.jit.load(str(model_path), map_location=self.device).to(self.device)
                self.backend = "torchscript"
                self.model.eval()
                return
            except Exception as exc:
                self._torchscript_error = str(exc)

        if model_path.is_dir() and (model_path / "hubconf.py").exists():
            try:
                self.model = torch.hub.load(str(model_path), model_name, source="local").to(self.device)
                self.backend = "torch_hub_local"
                self.model.eval()
                return
            except Exception as exc:
                self._hub_error = str(exc)

        try:
            import timm

            ckpt = str(model_path) if model_path.is_file() else ""
            self.model = timm.create_model(model_name, pretrained=False, checkpoint_path=ckpt, num_classes=0).to(self.device)
            self.backend = "timm"
            self.model.eval()
            return
        except Exception as exc:
            self._timm_error = str(exc)

        errors = {
            key: value
            for key, value in self.__dict__.items()
            if key.endswith("_error") and isinstance(value, str)
        }
        raise RuntimeError(f"could not load DINO/DINOv2 model from {model_path}; loader errors: {errors}")

    def encode(self, frames: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            if self.processor is not None:
                inputs = self.processor(images=[f.convert("RGB") for f in frames], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                features = pool_model_output(self.model(**inputs))
            else:
                batch = manual_preprocess(frames).to(self.device)
                features = pool_model_output(self.model(batch))
            features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy().astype(np.float32)


def consistency_score(features: np.ndarray) -> dict[str, Any]:
    if features.shape[0] < 2:
        raise RuntimeError("subject consistency needs at least two frames")
    adjacent = np.sum(features[:-1] * features[1:], axis=1)
    pairwise_values = []
    for i in range(features.shape[0]):
        for j in range(i + 1, features.shape[0]):
            pairwise_values.append(float(np.dot(features[i], features[j])))
    adjacent_cos = float(np.mean(adjacent))
    pairwise_cos = float(np.mean(pairwise_values)) if pairwise_values else adjacent_cos
    score = max(0.0, min(1.0, (adjacent_cos + 1.0) / 2.0))
    return {
        "score": score,
        "adjacent_cosine": adjacent_cos,
        "pairwise_cosine": pairwise_cos,
        "num_frames": int(features.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DINO/DINOv2 subject consistency scoring")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default="dinov2_vitb14")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    video_dir = Path(args.video_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    videos = iter_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"no videos found in {video_dir}")

    encoder = DinoEncoder(Path(args.model_path).expanduser().resolve(), args.device, args.model_name)
    results = []
    scores = []
    for video_path in videos:
        try:
            frames = load_sampled_frames(video_path, args.num_frames)
            details = consistency_score(encoder.encode(frames))
            scores.append(details["score"])
            results.append(
                {
                    "file": str(video_path),
                    "filename": video_path.name,
                    "score": details["score"],
                    "backend": encoder.backend,
                    "details": details,
                    "error": None,
                }
            )
        except Exception as exc:
            results.append({"file": str(video_path), "filename": video_path.name, "score": None, "error": str(exc)})

    payload = {
        "metric": "subject_consistency",
        "summary": {
            "mean_score": float(mean(scores)) if scores else math.nan,
            "total_samples": len(results),
            "successful_samples": len(scores),
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved subject consistency results to {output_path}")


if __name__ == "__main__":
    main()
