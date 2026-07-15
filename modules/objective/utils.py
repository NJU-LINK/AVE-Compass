import csv
import json
import math
import os
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


OBJECTIVE_METRICS = [
    "lip_sync",
    "av_sync",
    "video_aesthetic",
    "subject_consistency",
    "motion_smoothness",
    "audio_aesthetic",
    "speech_quality",
    "videoclip_video_similarity",
    "audio_similarity",
]

ACTIVE_METRICS = {
    "audio-only": ["av_sync", "audio_aesthetic", "videoclip_video_similarity"],
    "video-only": [
        "video_aesthetic",
        "subject_consistency",
        "motion_smoothness",
        "audio_similarity",
    ],
    "joint": [
        "av_sync",
        "video_aesthetic",
        "subject_consistency",
        "motion_smoothness",
        "audio_aesthetic",
    ],
    "speech": [
        "lip_sync",
        "av_sync",
        "video_aesthetic",
        "subject_consistency",
        "motion_smoothness",
        "audio_aesthetic",
        "speech_quality",
    ],
}


def get_active_metrics(category: Optional[str]) -> List[str]:
    normalized = normalize_category(category)
    return list(ACTIVE_METRICS.get(normalized, []))


def normalize_category(category: Optional[str]) -> str:
    if not category:
        return "joint"
    text = str(category).strip().lower().replace("_", "-")
    aliases = {
        "audio": "audio-only",
        "audio-only": "audio-only",
        "audio_only": "audio-only",
        "video": "video-only",
        "video-only": "video-only",
        "video_only": "video-only",
        "av": "joint",
        "audio-video": "joint",
        "joint": "joint",
        "speech": "speech",
        "speech-editing": "speech",
    }
    return aliases.get(text, text if text in ACTIVE_METRICS else "joint")


def clip01(value: Optional[float]) -> float:
    if value is None or not isinstance(value, (int, float)) or math.isnan(float(value)):
        return math.nan
    return float(max(0.0, min(1.0, float(value))))


def nanmean(values: Iterable[Any]) -> Optional[float]:
    valid = []
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(float(value)):
            valid.append(float(value))
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def metric_result(name: str, score: Optional[float], warning: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    value = clip01(score)
    result = {"metric": name, "score": None if math.isnan(value) else value}
    if warning:
        result["warning"] = warning
    if details:
        result["details"] = details
    return result


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(project_dir: Path, path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def extract_audio(video_path: str, output_wav: str, sample_rate: int = 16000) -> str:
    output = Path(output_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(completed.stderr.strip() or f"ffmpeg failed for {video_path}")
    return str(output)


def read_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported")
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def is_silent_audio(video_path: str, threshold: float = 1.0) -> bool:
    """Check if the audio track of a video is (near) silent.

    Extracts audio to mono WAV and computes RMS. Returns True if RMS < threshold.
    """
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        extract_audio(video_path, tmp_wav)
        data, _ = read_wav_mono(tmp_wav)
        os.unlink(tmp_wav)
        if data.size == 0:
            return True
        rms = float(np.sqrt(np.mean(data ** 2)))
        return rms < threshold
    except Exception:
        return False


def zscore(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return data
    std = float(np.std(data))
    if std < 1e-8:
        return np.array([], dtype=np.float32)
    return ((data - float(np.mean(data))) / std).astype(np.float32)


def best_lag_pearson(a: np.ndarray, b: np.ndarray, sr: int, max_shift_sec: float = 3.0) -> Tuple[float, int]:
    a = zscore(a)
    b = zscore(b)
    if a.size == 0 or b.size == 0:
        raise ValueError("empty or silent waveform")
    max_lag = int(max_shift_sec * sr)
    max_lag = min(max_lag, max(0, min(a.size, b.size) - 2))
    if max_lag <= 0:
        raise ValueError("waveform is too short")

    step = max(1, sr // 100)
    coarse_lags = range(-max_lag, max_lag + 1, step)
    coarse_best = max(coarse_lags, key=lambda lag: _overlap_corr(a, b, lag))
    fine_start = max(-max_lag, coarse_best - step)
    fine_end = min(max_lag, coarse_best + step)
    best = max(range(fine_start, fine_end + 1), key=lambda lag: _overlap_corr(a, b, lag))
    return _overlap_corr(a, b, best), best


def _overlap_corr(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    if lag >= 0:
        x = a[lag:]
        y = b[: x.size]
    else:
        y = b[-lag:]
        x = a[: y.size]
    n = min(x.size, y.size)
    if n < 2:
        return -1.0
    x = x[:n]
    y = y[:n]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-8:
        return -1.0
    return float(np.dot(x, y) / denom)


def sample_video_frames(video_path: str, num_frames: int, resize: Optional[Tuple[int, int]] = None, repeat_last: bool = False) -> List[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for frame sampling") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames found in video: {video_path}")
    indices = np.linspace(0, max(0, total - 1), min(num_frames, total), dtype=int).tolist()
    frames: List[np.ndarray] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
    cap.release()
    if repeat_last and frames:
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
    if not frames:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frames


def run_json_command(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    completed = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged_env, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "subprocess failed")


def read_first_csv_row(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return {}


def aggregate_objective_scores(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, Dict[str, Any]] = {}
    by_category: Dict[str, Dict[str, Any]] = {}
    overall: Dict[str, Any] = {}

    for metric in OBJECTIVE_METRICS:
        overall[metric] = nanmean(row.get("objective_scores", {}).get(metric) for row in rows)

    for row in rows:
        model = row.get("model_name") or "unknown"
        category = row.get("category") or "unknown"
        by_model.setdefault(model, {})
        by_category.setdefault(category, {})
        for metric in OBJECTIVE_METRICS:
            by_model[model][metric] = nanmean(
                r.get("objective_scores", {}).get(metric) for r in rows if (r.get("model_name") or "unknown") == model
            )
            by_category[category][metric] = nanmean(
                r.get("objective_scores", {}).get(metric) for r in rows if (r.get("category") or "unknown") == category
            )

    missing = {metric: 0 for metric in OBJECTIVE_METRICS}
    active_missing = {metric: 0 for metric in OBJECTIVE_METRICS}
    for row in rows:
        active = set(row.get("active_metrics", []))
        for metric in OBJECTIVE_METRICS:
            value = row.get("objective_scores", {}).get(metric)
            if value is None:
                missing[metric] += 1
                if metric in active:
                    active_missing[metric] += 1

    return {
        "overall": overall,
        "by_model": by_model,
        "by_category": by_category,
        "missing_metric_count": missing,
        "active_missing_metric_count": active_missing,
    }
