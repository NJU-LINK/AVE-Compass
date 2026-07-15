#!/usr/bin/env python3
"""Batch lip-sync evaluation using SyncNet.

Processes all videos in a directory and outputs results as JSON.
Reference: eval_lipsync.sh interface (--video_dir, --model_path, --output_file, --device, --temp_dir).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Allow importing syncnet_eval from the parent package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from syncnet_eval import SyncNetEval  # noqa: E402


def find_videos(video_dir: str) -> list:
    """Find all video files in a directory."""
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    videos = []
    for path in Path(video_dir).iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            videos.append(str(path))
    return sorted(videos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch lip-sync evaluation using SyncNet")
    parser.add_argument("--video_dir", type=str, required=True, help="Input directory containing videos")
    parser.add_argument("--model_path", type=str, required=True, help="Path to syncnet model (.model)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--temp_dir", type=str, default=None, help="Temporary directory for frame extraction")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for feature extraction")
    parser.add_argument("--vshift", type=int, default=15, help="Maximum frame shift for offset detection")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    videos = find_videos(args.video_dir)
    if not videos:
        print(f"ERROR: No videos found in {args.video_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(videos)} videos to evaluate")

    # Initialize evaluator
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    evaluator = SyncNetEval(device=device)
    evaluator.loadParameters(args.model_path)
    print(f"Model loaded from {args.model_path}, device={device}")

    # Temp directory
    temp_base = args.temp_dir or os.path.join(os.path.dirname(args.output_file), "temp_lipsync")
    os.makedirs(temp_base, exist_ok=True)

    results = []
    confidences = []
    for idx, video_path in enumerate(videos):
        filename = os.path.basename(video_path)
        print(f"[{idx + 1}/{len(videos)}] Evaluating: {filename}")

        result = {
            "file": video_path,
            "filename": filename,
            "av_offset": None,
            "min_dist": None,
            "confidence": None,
            "error": None,
        }

        try:
            temp_dir = os.path.join(temp_base, f"temp_{idx}")
            offset, min_dist, conf = evaluator.evaluate(
                video_path=video_path,
                temp_dir=temp_dir,
                batch_size=args.batch_size,
                vshift=args.vshift,
            )
            result["av_offset"] = offset
            result["min_dist"] = min_dist
            result["confidence"] = conf
            confidences.append(conf)
            print(f"  offset={offset}, min_dist={min_dist:.4f}, confidence={conf:.4f}")
        except Exception as exc:
            result["error"] = str(exc)
            print(f"  ERROR: {exc}")

        results.append(result)

    # Summary
    summary = {
        "total_samples": len(results),
        "successful": sum(1 for r in results if r["error"] is None),
        "failed": sum(1 for r in results if r["error"] is not None),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
    }

    output = {
        "metric": "lip_sync",
        "summary": summary,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output_file}")
    print(f"Total: {summary['total_samples']}, Success: {summary['successful']}, Failed: {summary['failed']}")
    if summary["mean_confidence"] is not None:
        print(f"Mean confidence: {summary['mean_confidence']:.4f}")


if __name__ == "__main__":
    main()
