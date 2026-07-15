"""
fal_samaudio.py - SAM Audio visual-guided separation via fal.ai API.

Replaces the local-GPU AudioSep approach with a remote API call to
fal.ai's hosted SAM Audio visual-guided model
(fal-ai/sam-audio/visual-separate).

Workflow:
    1. Build a video input. If the caller provides a separate audio_path,
       mux it with the video-only visual track first.
    2. Upload MP4 to fal CDN via fal_client.upload_file → file URL
    3. Submit separation job via fal queue → request_id
    4. Poll for completion → queue status
    5. Fetch result → target.url + residual.url
    6. Download target + residual audio to local disk

The tool returns the *kept* audio path (target or residual) based on mode.

API reference: https://fal.ai/models/fal-ai/sam-audio/visual-separate/api
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

import httpx

from av_editor.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

FAL_QUEUE_BASE = "https://queue.fal.run"

SAMAUDIO_SUPPORTED_ACTIONS: set[str] = {
    "audio_remove",     # remove the described sound, keep the rest
    "audio_keep",       # keep only the described sound
}


def _rms_db(audio_path: Path) -> float | None:
    """Return the overall RMS level of *audio_path* in dBFS (ffmpeg
    astats). Used to sanity-check SAM Audio separation — a near-
    silent ``target`` paired with a full-level ``residual`` means
    the separator didn't find the requested sound and the caller
    should treat the call as failed."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(audio_path),
             "-af", "astats=measure_overall=RMS_level", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
        for line in r.stderr.splitlines():
            if "RMS level dB" in line and "Overall" not in line:
                try:
                    return float(line.split("RMS level dB:")[-1].strip())
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def _extract_audio_wav(
    video_path: Path, output_dir: Path, sample_rate: int = 48000,
) -> Path:
    """Extract audio from video as mono WAV at SAM Audio's native 48kHz."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{video_path.stem}_audio.wav"
    if wav_path.exists():
        return wav_path
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    logger.info("[FalSAMAudio] extracted audio → %s", wav_path)
    return wav_path


def _mux_video_with_audio(video_path: Path, audio_path: Path, output_dir: Path) -> Path:
    """Create a compact MP4 with visuals from *video_path* and audio from
    *audio_path* for SAM Audio visual prompting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    muxed = output_dir / f"{video_path.stem}_with_{audio_path.stem}.mp4"
    if muxed.exists():
        return muxed
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(muxed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg video/audio mux failed: {result.stderr[:800]}")
    logger.info("[FalSAMAudio] muxed visual input → %s", muxed)
    return muxed


class FalSAMAudioTool(BaseTool):
    """
    SAM Audio visual-guided separation via fal.ai API.

    Uses Meta's SAM Audio model hosted on fal.ai serverless infrastructure.
    File upload uses fal_client SDK; job submission uses REST queue API.

    Params expected from Planner:
        description : str  – text describing the sound to separate/remove
                             e.g. "dog barking", "rain", "background music"
        mode        : str  – "remove" (keep residual) or "keep" (keep target)
                             Default: "remove"
        audio_path  : str  – (optional) path to audio file to mux with the
                             visual video before calling visual-separate.
        use_audio_only : bool – (optional) force the text-guided
                             audio-only endpoint.
    """

    name = "fal_samaudio"
    actions = SAMAUDIO_SUPPORTED_ACTIONS
    is_audio_separation_tool = True

    MODEL_ID = "fal-ai/sam-audio/visual-separate"
    AUDIO_ONLY_MODEL_ID = "fal-ai/sam-audio/separate"

    def __init__(self, api_key: str):
        self.api_key = api_key

    # ── internal helpers ───────────────────────────────────────────────

    def _headers(self, content_type: str = "application/json") -> dict[str, str]:
        h: dict[str, str] = {"Authorization": f"Key {self.api_key}"}
        if content_type:
            h["Content-Type"] = content_type
        return h

    # ── Step 1: Upload via fal_client ──────────────────────────────────

    def _upload_file_sync(self, file_path: Path) -> str:
        """Upload local file to fal CDN using fal_client SDK (synchronous)."""
        import fal_client

        # fal_client reads FAL_KEY from env; ensure it's set
        os.environ.setdefault("FAL_KEY", self.api_key)

        logger.info("[FalSAMAudio] uploading %s to fal CDN …", file_path.name)
        url = fal_client.upload_file(str(file_path))
        logger.info("[FalSAMAudio] uploaded → %s", url)
        return url

    # ── Step 2: Submit job via queue ───────────────────────────────────

    async def _submit_job(
        self,
        media_url: str,
        prompt: str,
        params: dict[str, Any],
        client: httpx.AsyncClient,
        *,
        use_audio_only: bool = False,
    ) -> dict[str, str]:
        """Submit SAM Audio separation job to fal queue.

        Returns a dict with keys: request_id, status_url, response_url.
        """
        model_id = self.AUDIO_ONLY_MODEL_ID if use_audio_only else self.MODEL_ID
        url = f"{FAL_QUEUE_BASE}/{model_id}"

        if use_audio_only:
            payload: dict[str, Any] = {
                "audio_url": media_url,
                "prompt": prompt,
                "predict_spans": params.get("predict_spans", True),
                "acceleration": params.get("acceleration", "quality"),
                "output_format": "wav",
            }
        else:
            payload = {
                "video_url": media_url,
                "prompt": prompt,
                "acceleration": params.get("acceleration", "balanced"),
                "max_chunk_duration": params.get("max_chunk_duration", 60),
                "chunk_overlap": params.get("chunk_overlap", 5),
                "output_format": "wav",
            }
            if params.get("mask_video_url"):
                payload["mask_video_url"] = params["mask_video_url"]

        # Optional: reranking for better quality (costs more)
        reranking = params.get("reranking_candidates", 1)
        if reranking > 1:
            payload["reranking_candidates"] = min(reranking, 7)

        logger.info(
            "[FalSAMAudio] submitting %s separation (prompt='%s')",
            "audio-only" if use_audio_only else "visual",
            prompt,
        )
        resp = await client.post(url, headers=self._headers(), json=payload)
        if resp.status_code >= 400:
            logger.error("[FalSAMAudio] submit failed %d: %s", resp.status_code, resp.text[:1000])
        resp.raise_for_status()

        data = resp.json()
        request_id = data.get("request_id", "")
        if not request_id:
            raise RuntimeError(f"No request_id in submit response: {data}")

        logger.info("[FalSAMAudio] job submitted → request_id=%s", request_id)
        return {
            "request_id": request_id,
            "status_url": data.get("status_url", ""),
            "response_url": data.get("response_url", ""),
        }

    # ── Step 3: Poll status ────────────────────────────────────────────

    async def _poll_status(
        self, status_url: str, request_id: str, client: httpx.AsyncClient,
    ) -> None:
        """Poll queue status until COMPLETED or failed (max 5 min)."""
        import time as _time
        start = _time.time()

        while _time.time() - start < 600:
            resp = await client.get(
                status_url, headers=self._headers(content_type=""),
            )
            # 202 = IN_PROGRESS, 200 = COMPLETED
            if resp.status_code not in (200, 202):
                raise RuntimeError(
                    f"Unexpected status poll response: {resp.status_code} {resp.text[:200]}"
                )
            data = resp.json()
            status = data.get("status", "")

            if status == "COMPLETED":
                logger.info("[FalSAMAudio] job %s completed", request_id)
                return
            if status in ("FAILED", "CANCELLED"):
                error = data.get("error", "unknown")
                raise RuntimeError(f"Job {request_id} {status}: {error}")

            queue_pos = data.get("queue_position", "?")
            logger.debug(
                "[FalSAMAudio] job %s status=%s queue_pos=%s, waiting…",
                request_id, status, queue_pos,
            )
            await asyncio.sleep(3)

        raise TimeoutError(f"Job {request_id} timed out after 600s")

    # ── Step 4: Fetch result ───────────────────────────────────────────

    async def _fetch_result(
        self, response_url: str, client: httpx.AsyncClient,
    ) -> dict[str, Any]:
        """Fetch the completed job result."""
        resp = await client.get(response_url, headers=self._headers(content_type=""))
        if resp.status_code >= 400:
            logger.error("[FalSAMAudio] result failed %d: %s", resp.status_code, resp.text[:1000])
        resp.raise_for_status()
        return resp.json()

    # ── Step 5: Download ───────────────────────────────────────────────

    async def _download(
        self, url: str, output_path: Path, client: httpx.AsyncClient,
    ) -> Path:
        """Download an audio file to local disk."""
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        logger.info(
            "[FalSAMAudio] downloaded → %s (%.1f KB)",
            output_path.name,
            output_path.stat().st_size / 1e3,
        )
        return output_path

    # ── Main execute ───────────────────────────────────────────────────

    async def execute(
        self,
        video_path: Path,
        action: str,
        params: dict[str, Any],
        output_dir: Path,
    ) -> ToolResult:
        """
        Full flow:
            prepare video/audio-visual input → upload to fal CDN
            → submit separation → poll → download

        Returns ToolResult with output_path pointing to the *kept* audio:
            - mode="remove" → residual (everything except the described sound)
            - mode="keep"   → target (only the described sound)
        """
        description = params.get("description", params.get("prompt", ""))
        mode = "keep" if action == "audio_keep" else params.get("mode", "remove")

        if not description:
            return ToolResult(
                success=False,
                error_msg="No 'description' provided for audio separation",
            )

        logger.info(
            "[FalSAMAudio] action=%s description='%s' mode=%s",
            action, description, mode,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=300) as client:
            try:
                use_audio_only = bool(params.get("use_audio_only", False))
                if use_audio_only:
                    # Legacy text-guided endpoint for explicit fallback/debug.
                    explicit_audio = params.get("audio_path")
                    if explicit_audio:
                        src_path = Path(explicit_audio)
                        logger.info("[FalSAMAudio] using provided audio: %s", src_path)
                        if src_path.suffix.lower() != ".wav":
                            wav_path = output_dir / f"{src_path.stem}_converted.wav"
                            if not wav_path.exists():
                                await asyncio.to_thread(
                                    lambda: subprocess.run(
                                        ["ffmpeg", "-y", "-i", str(src_path),
                                         "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
                                         str(wav_path)],
                                        capture_output=True, text=True, check=True,
                                    ),
                                )
                                logger.info("[FalSAMAudio] converted %s → %s", src_path.name, wav_path.name)
                        else:
                            wav_path = src_path
                    else:
                        wav_path = await asyncio.to_thread(
                            _extract_audio_wav, video_path, output_dir,
                        )
                    media_path = wav_path
                else:
                    if not video_path.exists():
                        raise RuntimeError(f"video_path not found: {video_path}")

                    explicit_audio = params.get("audio_path")
                    if explicit_audio:
                        src_path = Path(explicit_audio)
                        logger.info(
                            "[FalSAMAudio] muxing provided audio into video input: %s",
                            src_path,
                        )
                        media_path = await asyncio.to_thread(
                            _mux_video_with_audio, video_path, src_path, output_dir,
                        )
                    else:
                        media_path = video_path

                    if params.get("mask_video_path") and not params.get("mask_video_url"):
                        mask_path = Path(params["mask_video_path"])
                        params["mask_video_url"] = await asyncio.to_thread(
                            self._upload_file_sync, mask_path,
                        )

                # 2. Upload to fal CDN via fal_client SDK
                media_url = await asyncio.to_thread(
                    self._upload_file_sync, media_path,
                )

                # 3. Submit separation job
                job = await self._submit_job(
                    media_url, description, params, client,
                    use_audio_only=use_audio_only,
                )
                request_id = job["request_id"]

                # 4. Poll until completed (use URLs returned by submit)
                await self._poll_status(
                    job["status_url"], request_id, client,
                )

                # 5. Fetch result
                result_data = await self._fetch_result(
                    job["response_url"], client,
                )

                # 6. Download target + residual
                #    API returns: { target: {url: ...}, residual: {url: ...} }
                target_info = result_data.get("target", {})
                residual_info = result_data.get("residual", {})

                target_url = target_info.get("url", "")
                residual_url = residual_info.get("url", "")

                if not target_url:
                    raise RuntimeError(
                        f"No target URL in result: {result_data}"
                    )

                tag = uuid.uuid4().hex[:8]
                target_path = output_dir / f"target_{tag}.wav"
                await self._download(target_url, target_path, client)

                residual_path: Path | None = None
                if residual_url:
                    residual_path = output_dir / f"residual_{tag}.wav"
                    await self._download(residual_url, residual_path, client)

                # Always log RMS for diagnostics.
                target_rms = _rms_db(target_path)
                residual_rms = _rms_db(residual_path) if residual_path else None
                logger.info(
                    "[FalSAMAudio] separation RMS target=%s dB residual=%s dB",
                    f"{target_rms:.1f}" if target_rms is not None else "n/a",
                    f"{residual_rms:.1f}" if residual_rms is not None else "n/a",
                )

                # Conditional sanity check, gated by the Planner.
                # Many legitimate target sounds (faint bird chirp, distant
                # moo) are quiet relative to the rest of the mix, so a
                # large residual−target gap can mean either "SAM missed"
                # or "the target really was that quiet". The Planner sets
                # `expect_prominent_target=true` only when the audio
                # caption indicates the target is loud/prominent in the
                # original — in which case an empty target stem is almost
                # certainly a SAM failure and we fail-fast so the caller
                # can retry with a different prompt.
                expect_prominent = bool(params.get("expect_prominent_target", False))
                if (
                    expect_prominent
                    and target_rms is not None
                    and residual_rms is not None
                    and residual_rms - target_rms > 25.0
                ):
                    logger.warning(
                        "[FalSAMAudio] separation appears empty "
                        "(target %.1f dB << residual %.1f dB) AND planner "
                        "expected the target to be prominent: prompt "
                        "'%s' did not match anything in the audio. "
                        "Returning failure so the caller can retry.",
                        target_rms, residual_rms, description,
                    )
                    return ToolResult(
                        success=False,
                        error_msg=(
                            f"SAM Audio separation empty (prominent-target "
                            f"check): target {target_rms:.1f} dB vs "
                            f"residual {residual_rms:.1f} dB — prompt "
                            f"{description!r} missed the target sound."
                        ),
                    )

                # Determine which to keep
                if mode == "keep":
                    kept_path = target_path
                else:
                    kept_path = residual_path if residual_path else target_path

                return ToolResult(
                    success=True,
                    output_path=kept_path,
                    raw_response={
                        "target_path": str(target_path),
                        "residual_path": str(residual_path) if residual_path else None,
                        "kept_path": str(kept_path),
                        "description": description,
                        "mode": mode,
                        "duration": result_data.get("duration"),
                        "sample_rate": result_data.get("sample_rate"),
                        "target_rms_db": target_rms,
                        "residual_rms_db": residual_rms,
                        "model_id": self.AUDIO_ONLY_MODEL_ID if use_audio_only else self.MODEL_ID,
                        "media_input": str(media_path),
                        "use_audio_only": use_audio_only,
                    },
                )

            except Exception as exc:
                logger.error("[FalSAMAudio] failed: %s", exc)
                return ToolResult(success=False, error_msg=str(exc))
