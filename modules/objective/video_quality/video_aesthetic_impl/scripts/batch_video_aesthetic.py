#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

import cv2
from PIL import Image
import torch

# Add aesthetic_predictor_v2_5 src to path
_ap_v25_src = str(Path(__file__).resolve().parents[1] / 'Objective' / 'Video' / 'aesthetic-predictor-v2-5' / 'src')
import sys
if _ap_v25_src not in sys.path:
    sys.path.insert(0, _ap_v25_src)
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
DEFAULT_PREDICTOR_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / 'Objective'
    / 'Video'
    / 'aesthetic-predictor-v2-5'
    / 'models'
    / 'aesthetic_predictor_v2_5.pth'
)


def iter_videos(video_dir: Path) -> list[Path]:
    return sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def sample_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    if num_frames <= 1 or total_frames <= num_frames:
        return list(range(total_frames))
    step = (total_frames - 1) / float(num_frames - 1)
    return sorted({min(total_frames - 1, round(i * step)) for i in range(num_frames)})


def load_sampled_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total_frames, num_frames)
    frames: list[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


DEFAULT_SIGLIP_PATH = (
    Path(__file__).resolve().parents[5]
    / 'models'
    / 'video_quality'
    / 'video_aesthetic'
    / 'siglip-so400m-patch14-384'
)


def build_predictor(device: str, predictor_weights: Path):
    encoder_model = str(DEFAULT_SIGLIP_PATH) if DEFAULT_SIGLIP_PATH.exists() else "google/siglip-so400m-patch14-384"
    model, preprocessor = convert_v2_5_from_siglip(
        predictor_name_or_path=str(predictor_weights),
        encoder_model_name=encoder_model,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=DEFAULT_SIGLIP_PATH.exists(),
    )
    if device.startswith('cuda') and torch.cuda.is_available():
        model = model.to(torch.bfloat16).to(device)
    else:
        device = 'cpu'
        model = model.to(device)
    model.eval()
    return model, preprocessor, device


def score_frames(model, preprocessor, frames: Iterable[Image.Image], device: str) -> list[float]:
    scores: list[float] = []
    for frame in frames:
        pixel_values = preprocessor(images=frame.convert('RGB'), return_tensors='pt').pixel_values
        if device != 'cpu':
            pixel_values = pixel_values.to(torch.bfloat16).to(device)
        with torch.inference_mode():
            score = model(pixel_values).logits.squeeze().float().cpu().item()
        scores.append(float(score))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch video aesthetic scoring')
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_frames', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model_path', default=str(DEFAULT_PREDICTOR_WEIGHTS))
    args = parser.parse_args()

    video_dir = Path(args.video_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    videos = iter_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f'No videos found in {video_dir}')

    model, preprocessor, device = build_predictor(args.device, Path(args.model_path).expanduser().resolve())

    results = []
    scores = []
    for video_path in videos:
        frame_scores = score_frames(model, preprocessor, load_sampled_frames(video_path, args.num_frames), device)
        if not frame_scores:
            results.append(
                {
                    'file': str(video_path),
                    'filename': video_path.name,
                    'error': 'no readable frames',
                    'score': None,
                }
            )
            continue
        video_score = float(mean(frame_scores))
        scores.append(video_score)
        results.append(
            {
                'file': str(video_path),
                'filename': video_path.name,
                'score': video_score,
                'num_frames': len(frame_scores),
                'frame_scores': frame_scores,
            }
        )

    payload = {
        'metric': 'video_aesthetic',
        'summary': {
            'mean_score': float(mean(scores)) if scores else 0.0,
            'total_samples': len(results),
            'successful_samples': len(scores),
        },
        'results': results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Saved results to {output_path}')


if __name__ == '__main__':
    main()
