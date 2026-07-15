#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


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
        # Fallback: re-encode with ffmpeg and retry
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(video_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an", tmp_path,
        ]
        subprocess.run(cmd, check=True)
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in sample_frame_indices(total, num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()
        Path(tmp_path).unlink(missing_ok=True)
    if not frames:
        raise RuntimeError(f"no readable frames: {video_path}")
    # Pad to exactly num_frames by repeating last frame if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])
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
    elif hasattr(output, "image_embeds") and output.image_embeds is not None:
        tensor = output.image_embeds
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


def normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


class VisualEncoder:
    def __init__(self, model_path: Path, device: str, open_clip_model: str) -> None:
        import torch

        self.torch = torch
        self.device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.processor = None
        self.preprocess = None
        self.encode_kind = "forward"
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

        # Try VideoCLIP-XL native format
        if model_path.is_file():
            try:
                ckpt = torch.load(str(model_path), map_location="cpu")
                if isinstance(ckpt, dict) and any(k.startswith("vision_model.") for k in ckpt.keys()):
                    videoclip_dir = model_path.parent
                    if (videoclip_dir / "modeling.py").exists():
                        sys.path.insert(0, str(videoclip_dir))
                        from utils.vision_encoder import get_vision_encoder
                        vision_model = get_vision_encoder()
                        # Load only vision_model keys
                        vision_state = {k.replace("vision_model.", "", 1): v for k, v in ckpt.items() if k.startswith("vision_model.")}
                        vision_model.load_state_dict(vision_state, strict=False)
                        self.model = vision_model.to(self.device)
                        self.encode_kind = "videoclip_xl"
                        self.backend = "videoclip_xl"
                        self.model.eval()
                        return
            except Exception as exc:
                self._videoclip_xl_error = str(exc)

        try:
            open_clip_src = (
                Path(__file__).resolve().parents[3]
                / "cross_modal"
                / "av_sync_impl"
                / "Objective"
                / "Similarity"
                / "Synchformer-main"
                / "model"
                / "modules"
                / "feat_extractors"
                / "train_clip_src"
            )
            if open_clip_src.exists():
                sys.path.insert(0, str(open_clip_src))
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(open_clip_model, pretrained=None)
            if model_path.is_file():
                ckpt = torch.load(str(model_path), map_location="cpu")
                self.model.load_state_dict(normalize_state_dict(ckpt), strict=False)
            self.model = self.model.to(self.device)
            self.encode_kind = "encode_image"
            self.backend = f"open_clip:{open_clip_model}"
            self.model.eval()
            return
        except Exception as exc:
            self._open_clip_error = str(exc)

        try:
            import timm

            ckpt = str(model_path) if model_path.is_file() else ""
            self.model = timm.create_model("vit_base_patch16_224", pretrained=False, checkpoint_path=ckpt, num_classes=0).to(self.device)
            self.backend = "timm:vit_base_patch16_224"
            self.model.eval()
            return
        except Exception as exc:
            self._timm_error = str(exc)

        errors = {
            key: value
            for key, value in self.__dict__.items()
            if key.endswith("_error") and isinstance(value, str)
        }
        raise RuntimeError(f"could not load visual similarity model from {model_path}; loader errors: {errors}")

    def encode(self, frames: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            if self.encode_kind == "videoclip_xl":
                # VideoCLIP-XL expects (B, T, C, H, W) input
                v_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
                v_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
                tensors = []
                for frame in frames:
                    fr = np.asarray(frame.convert("RGB").resize((224, 224)), dtype=np.float32)
                    fr = (fr / 255.0 - v_mean) / v_std
                    fr = np.expand_dims(fr, axis=(0, 1))  # (1, 1, H, W, C)
                    tensors.append(fr)
                vid_tube = np.concatenate(tensors, axis=1)  # (1, T, H, W, C)
                vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))  # (1, T, C, H, W)
                vid_tube = torch.from_numpy(vid_tube).float().to(self.device)
                features = self.model.get_vid_features(vid_tube)
            elif self.processor is not None:
                inputs = self.processor(images=[f.convert("RGB") for f in frames], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if hasattr(self.model, "get_image_features"):
                    features = self.model.get_image_features(**inputs)
                else:
                    features = pool_model_output(self.model(**inputs))
            elif self.preprocess is not None:
                batch = torch.stack([self.preprocess(f.convert("RGB")) for f in frames], dim=0).to(self.device)
                features = self.model.encode_image(batch) if self.encode_kind == "encode_image" else pool_model_output(self.model(batch))
            else:
                batch = manual_preprocess(frames).to(self.device)
                features = pool_model_output(self.model(batch))
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            video_feature = torch.nn.functional.normalize(features.mean(dim=0, keepdim=True), dim=-1)
        return video_feature.cpu().numpy().astype(np.float32)[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        raise RuntimeError("empty visual embedding")
    return float(np.dot(a, b) / denom)


def similarity_score(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, min(1.0, (cosine(a, b) + 1.0) / 2.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Video visual similarity scoring")
    parser.add_argument("--source_video", required=True)
    parser.add_argument("--edited_video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--metric", default="video_similarity")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--open_clip_model", default="ViT-L-14")
    args = parser.parse_args()

    source_video = Path(args.source_video).expanduser().resolve()
    edited_video = Path(args.edited_video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = VisualEncoder(Path(args.model_path).expanduser().resolve(), args.device, args.open_clip_model)
    src_feature = encoder.encode(load_sampled_frames(source_video, args.num_frames))
    edited_feature = encoder.encode(load_sampled_frames(edited_video, args.num_frames))
    raw_cosine = cosine(src_feature, edited_feature)
    score = max(0.0, min(1.0, (raw_cosine + 1.0) / 2.0))

    result = {
        "source_video": str(source_video),
        "edited_video": str(edited_video),
        "source_filename": source_video.name,
        "edited_filename": edited_video.name,
        "score": score,
        "details": {
            "raw_cosine": raw_cosine,
            "num_frames": args.num_frames,
            "backend": encoder.backend,
        },
        "error": None,
    }
    payload = {
        "metric": args.metric,
        "summary": {"mean_score": score, "total_samples": 1, "successful_samples": 1},
        "results": [result],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved video similarity results to {output_path}")


if __name__ == "__main__":
    main()
