"""
fal_common.py - Small helpers shared by fal.ai-backed tools.

The fal queue REST API returns request URLs on submit; files are uploaded
through the official Python client so local media can be used as model input.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

FAL_QUEUE_BASE = "https://queue.fal.run"


class FalQueueClient:
    """Minimal async wrapper around fal queue submit/status/result."""

    def __init__(self, api_key: str, timeout: int = 600):
        self.api_key = api_key
        self.timeout = timeout

    def headers(self, content_type: str = "application/json") -> dict[str, str]:
        h: dict[str, str] = {"Authorization": f"Key {self.api_key}"}
        if content_type:
            h["Content-Type"] = content_type
        return h

    def upload_file_sync(self, file_path: Path) -> str:
        import fal_client

        os.environ.setdefault("FAL_KEY", self.api_key)
        logger.info("[fal] uploading %s …", file_path.name)
        url = fal_client.upload_file(str(file_path))
        logger.info("[fal] uploaded → %s", url)
        return url

    async def upload_file(self, file_path: Path) -> str:
        return await asyncio.to_thread(self.upload_file_sync, file_path)

    async def submit(
        self,
        model_id: str,
        payload: dict[str, Any],
        client: httpx.AsyncClient,
    ) -> dict[str, str]:
        resp = await client.post(
            f"{FAL_QUEUE_BASE}/{model_id}",
            headers=self.headers(),
            json=payload,
        )
        if resp.status_code >= 400:
            logger.error("[fal] submit failed %d: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
        data = resp.json()
        request_id = data.get("request_id", "")
        if not request_id:
            raise RuntimeError(f"No request_id in fal submit response: {data}")
        return {
            "request_id": request_id,
            "status_url": data.get("status_url", ""),
            "response_url": data.get("response_url", ""),
        }

    async def poll(
        self,
        job: dict[str, str],
        client: httpx.AsyncClient,
        poll_interval: float = 3.0,
    ) -> None:
        request_id = job["request_id"]
        status_url = job.get("status_url", "")
        if not status_url:
            raise RuntimeError(f"No status_url for fal request {request_id}")

        start = time.time()
        while time.time() - start < self.timeout:
            resp = await client.get(status_url, headers=self.headers(content_type=""))
            if resp.status_code not in (200, 202):
                raise RuntimeError(
                    f"Unexpected fal status response: {resp.status_code} {resp.text[:300]}"
                )
            data = resp.json()
            status = data.get("status", "")
            if status == "COMPLETED":
                logger.info("[fal] request %s completed", request_id)
                return
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"fal request {request_id} {status}: {data.get('error', data)}")
            logger.debug("[fal] request %s status=%s", request_id, status)
            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"fal request {request_id} timed out after {self.timeout}s")

    async def result(
        self,
        job: dict[str, str],
        client: httpx.AsyncClient,
    ) -> dict[str, Any]:
        response_url = job.get("response_url", "")
        if not response_url:
            raise RuntimeError(f"No response_url for fal request {job['request_id']}")
        resp = await client.get(response_url, headers=self.headers(content_type=""))
        if resp.status_code >= 400:
            logger.error("[fal] result failed %d: %s", resp.status_code, resp.text[:1000])
        resp.raise_for_status()
        return resp.json()

    async def run(
        self,
        model_id: str,
        payload: dict[str, Any],
        client: httpx.AsyncClient,
        poll_interval: float = 3.0,
    ) -> dict[str, Any]:
        job = await self.submit(model_id, payload, client)
        await self.poll(job, client, poll_interval=poll_interval)
        data = await self.result(job, client)
        data.setdefault("_fal_request_id", job["request_id"])
        return data


async def download_url(url: str, output_path: Path, client: httpx.AsyncClient) -> Path:
    resp = await client.get(url, follow_redirects=True)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    logger.info("[fal] downloaded → %s (%.1f KB)", output_path.name, output_path.stat().st_size / 1e3)
    return output_path


def extract_audio_wav(video_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[:500]}")
    return output_path
