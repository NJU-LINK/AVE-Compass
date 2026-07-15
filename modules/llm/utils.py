#!/usr/bin/env python3
"""
工具模块 - 视频和音频处理
"""
from __future__ import annotations

import os
import base64
import tempfile
import logging
import subprocess
import shutil
import mimetypes
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent for checklist-only runs
    np = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent for checklist-only runs
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise ModuleNotFoundError("cv2 is required for video frame processing. Install opencv-python.")
    return cv2


def _require_image():
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise ModuleNotFoundError("PIL is required for frame image encoding. Install Pillow.") from exc
    return Image


def resolve_ffmpeg_executable() -> str:
    """
    Resolve an ffmpeg executable path.
    Priority:
    1) env `FFMPEG_PATH`
    2) system PATH
    3) `imageio-ffmpeg` bundled binary
    """
    env_path = os.getenv("FFMPEG_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p)

    in_path = shutil.which("ffmpeg")
    if in_path:
        return in_path

    try:
        import imageio_ffmpeg  # type: ignore
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass

    return "ffmpeg"


def extract_frames(video_path: str, num_frames: int = 8) -> List[np.ndarray]:
    """
    从视频中提取指定数量的帧

    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数

    Returns:
        帧列表
    """
    cv = _require_cv2()
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"视频中没有帧: {video_path}")

    step = max(1, total_frames // num_frames)
    selected_indices = [i * step for i in range(num_frames)]

    frames = []
    for idx in selected_indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        if len(frames) >= num_frames:
            break

    cap.release()

    if not frames:
        raise ValueError(f"无法从视频中提取帧: {video_path}")

    return frames


def frame_to_base64(frame: np.ndarray, format: str = "JPEG", quality: int = 75) -> str:
    """
    将帧转换为base64编码

    Args:
        frame: numpy数组格式的帧
        format: 图像格式
        quality: JPEG压缩质量

    Returns:
        base64编码字符串
    """
    Image = _require_image()
    img = Image.fromarray(frame)
    import io
    buffer = io.BytesIO()
    if format == "JPEG":
        img.save(buffer, format=format, quality=quality, optimize=True)
    else:
        img.save(buffer, format=format)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64


def video_to_base64_frames(video_path: str, num_frames: int = 8) -> List[str]:
    """
    将视频转换为base64编码的帧列表

    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数

    Returns:
        base64编码的帧列表
    """
    frames = extract_frames(video_path, num_frames)
    return [frame_to_base64(resize_frame(frame)) for frame in frames]


def resize_frame(frame: np.ndarray, max_size: int = 512) -> np.ndarray:
    """
    调整帧大小，保持宽高比

    Args:
        frame: 输入帧
        max_size: 最大尺寸

    Returns:
        调整后的帧
    """
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return frame

    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))

    return _require_cv2().resize(frame, (new_w, new_h))


def get_video_info(video_path: str) -> dict:
    """
    获取视频信息

    Args:
        video_path: 视频文件路径

    Returns:
        视频信息字典
    """
    cv = _require_cv2()
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    info = {
        "fps": cap.get(cv.CAP_PROP_FPS),
        "width": int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv.CAP_PROP_FRAME_COUNT)) / cap.get(cv.CAP_PROP_FPS) if cap.get(cv.CAP_PROP_FPS) > 0 else 0
    }

    cap.release()
    return info


def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> Tuple[str, str]:
    """
    从视频中提取音频

    Args:
        video_path: 视频文件路径
        output_path: 输出音频文件路径（可选）

    Returns:
        (音频文件路径, 音频格式)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / f"{video_path.stem}.mp3"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Extracting audio from {video_path} to {output_path}")

        ffmpeg_exe = resolve_ffmpeg_executable()
        cmd = [
            ffmpeg_exe,
            '-i', str(video_path),
            '-vn',
            '-acodec', 'libmp3lame',
            '-ab', '192k',
            '-y',
            str(output_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.warning(f"ffmpeg stderr: {result.stderr}")

        if output_path.exists():
            return str(output_path), "mp3"
        else:
            raise Exception("Audio extraction failed")
    except Exception as e:
        logger.error(f"Failed to extract audio: {e}")
        raise


def audio_to_base64(audio_path: str) -> str:
    """
    将音频文件转换为base64编码

    Args:
        audio_path: 音频文件路径

    Returns:
        base64编码字符串
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(audio_path, 'rb') as f:
        audio_data = f.read()

    return base64.b64encode(audio_data).decode('utf-8')


def get_audio_mime_type(audio_format: str) -> str:
    """
    获取音频的MIME类型

    Args:
        audio_format: 音频格式（mp3, wav等）

    Returns:
        MIME类型字符串
    """
    mime_types = {
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'ogg': 'audio/ogg',
        'flac': 'audio/flac',
        'm4a': 'audio/mp4'
    }
    return mime_types.get(audio_format.lower(), 'audio/mpeg')


def get_video_mime_type(video_path: str) -> str:
    """
    Infer MIME type for a video file path.
    """
    guessed, _ = mimetypes.guess_type(str(video_path))
    if guessed and guessed.startswith("video/"):
        return guessed
    return "video/mp4"


def video_file_to_base64(video_path: str) -> str:
    """
    Read a video file and encode it to base64.
    """
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_and_encode_audio(video_path: str) -> Tuple[str, str]:
    """
    从视频中提取音频并编码为base64

    Args:
        video_path: 视频文件路径

    Returns:
        (base64编码的音频, MIME类型)
    """
    audio_path, audio_format = extract_audio_from_video(video_path)
    audio_base64 = audio_to_base64(audio_path)
    mime_type = get_audio_mime_type(audio_format)

    return audio_base64, mime_type


def extract_and_encode_video(video_path: str) -> Tuple[str, str]:
    """
    Encode full video to base64 and return MIME type.
    """
    video_base64 = video_file_to_base64(video_path)
    mime_type = get_video_mime_type(video_path)
    return video_base64, mime_type
