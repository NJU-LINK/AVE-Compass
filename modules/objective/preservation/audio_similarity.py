import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from modules.objective.utils import extract_audio, metric_result, read_wav_mono, zscore


def _stft_magnitude(signal: np.ndarray, sr: int, nperseg: int = 1024, hop: int = 512) -> np.ndarray:
    """Compute STFT magnitude spectrogram.

    Returns a (n_freq, n_time) array.
    """
    try:
        from scipy.signal import stft as scipy_stft
        _, _, Z = scipy_stft(signal, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
        return np.abs(Z).astype(np.float32)
    except ImportError:
        # Fallback: manual STFT with Hann window
        window = np.hanning(nperseg).astype(np.float32)
        n_frames = max(1, 1 + (len(signal) - nperseg) // hop)
        spec = np.zeros((nperseg // 2 + 1, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop
            frame = signal[start:start + nperseg]
            if len(frame) < nperseg:
                frame = np.pad(frame, (0, nperseg - len(frame)))
            frame = frame * window
            spectrum = np.fft.rfft(frame)
            spec[:, i] = np.abs(spectrum).astype(np.float32)
        return spec


def _spectral_corr_at_lag(spec_a: np.ndarray, spec_b: np.ndarray, lag_frames: int) -> float:
    """Pearson correlation of STFT magnitude spectrograms at a given frame lag."""
    if lag_frames >= 0:
        x = spec_a[:, lag_frames:]
        y = spec_b[:, :x.shape[1]]
    else:
        y = spec_b[:, -lag_frames:]
        x = spec_a[:, :y.shape[1]]
    n = min(x.shape[1], y.shape[1])
    if n < 2:
        return -1.0
    x = x[:, :n].flatten()
    y = y[:, :n].flatten()
    x = zscore(x)
    y = zscore(y)
    if x.size == 0 or y.size == 0:
        return -1.0
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-8:
        return -1.0
    return float(np.dot(x, y) / denom)


def _best_lag_spectral(spec_a: np.ndarray, spec_b: np.ndarray, max_lag_frames: int, min_overlap_frac: float = 0.5) -> Tuple[float, int]:
    """Search for best alignment using STFT magnitude correlation.

    Enforces a minimum overlap to avoid spurious high correlations from tiny overlaps.
    """
    min_frames = min(spec_a.shape[1], spec_b.shape[1])
    max_lag_frames = min(max_lag_frames, max(0, int(min_frames * (1.0 - min_overlap_frac))))

    best_rho = -2.0
    best_lag = 0
    # Coarse search
    step = max(1, max_lag_frames // 50) if max_lag_frames > 0 else 1
    for lag in range(-max_lag_frames, max_lag_frames + 1, step):
        rho = _spectral_corr_at_lag(spec_a, spec_b, lag)
        if rho > best_rho:
            best_rho = rho
            best_lag = lag

    # Fine search around best coarse lag
    fine_start = max(-max_lag_frames, best_lag - step)
    fine_end = min(max_lag_frames, best_lag + step)
    for lag in range(fine_start, fine_end + 1):
        rho = _spectral_corr_at_lag(spec_a, spec_b, lag)
        if rho > best_rho:
            best_rho = rho
            best_lag = lag

    return best_rho, best_lag


def compute_audio_similarity(sample: Dict[str, Any], config: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    metric = "audio_similarity"
    try:
        src_wav = extract_audio(sample["source_video_path"], str(work_dir / "audio_sim" / "source.wav"))
        edited_wav = extract_audio(sample["edited_video_path"], str(work_dir / "audio_sim" / "edited.wav"))
        src, src_sr = read_wav_mono(src_wav)
        edited, edited_sr = read_wav_mono(edited_wav)
        if src_sr != edited_sr:
            return metric_result(metric, None, f"sample-rate mismatch after extraction: {src_sr} vs {edited_sr}")

        # Handle silent audio: if both tracks are (near) silent, they are identical.
        src_rms = float(np.sqrt(np.mean(src ** 2))) if src.size else 0.0
        edt_rms = float(np.sqrt(np.mean(edited ** 2))) if edited.size else 0.0
        if src_rms < 1e-3 and edt_rms < 1e-3:
            return metric_result(metric, 1.0, details={"method": "both_silent", "src_rms": src_rms, "edt_rms": edt_rms})

        settings = config.get("settings", {})
        max_shift_sec = float(settings.get("audio_max_shift_sec", settings.get("waveform_max_shift_sec", 3.0)))

        # Compute STFT magnitude spectrograms
        nperseg = int(settings.get("audio_stft_nperseg", settings.get("waveform_stft_nperseg", 1024)))
        hop = nperseg // 2
        spec_src = _stft_magnitude(src, src_sr, nperseg=nperseg, hop=hop)
        spec_edt = _stft_magnitude(edited, edited_sr, nperseg=nperseg, hop=hop)

        if spec_src.size == 0 or spec_edt.size == 0:
            return metric_result(metric, None, "empty spectrogram")

        # Convert max_shift_sec to frames
        hop_sec = hop / src_sr
        max_lag_frames = int(max_shift_sec / hop_sec)

        # Search for best alignment using spectral correlation
        rho, lag_frames = _best_lag_spectral(spec_src, spec_edt, max_lag_frames, min_overlap_frac=0.5)

        if math.isnan(rho):
            return metric_result(metric, None, "spectral correlation is NaN")

        return metric_result(metric, (1.0 + rho) / 2.0, details={
            "rho": rho,
            "best_lag_frames": lag_frames,
            "best_lag_sec": lag_frames * hop_sec,
            "method": "stft_magnitude",
        })
    except Exception as exc:
        return metric_result(metric, None, str(exc))
