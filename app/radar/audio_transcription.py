"""Optional ASR bridge for explicitly allowlisted, creator-authorized audio."""
from __future__ import annotations

import io
import hashlib
import re
from urllib.parse import urlsplit

import httpx

from app.radar.sources import fetch_public
from app.radar.spoken_media import format_time

YOUTUBE_HOSTS = {"youtube.com", "youtu.be", "googlevideo.com"}
AUDIO_MAX_BYTES = 25 * 1024 * 1024


def host_is_allowed(url: str, allowlist: tuple[str, ...]) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().rstrip(".")
    if parts.scheme.lower() != "https":
        return False
    if not host or any(host == domain or host.endswith("." + domain) for domain in YOUTUBE_HOSTS):
        return False
    return any(host == domain or host.endswith("." + domain.lower().lstrip(".")) for domain in allowlist)


def transcribe_audio(url: str, *, endpoint: str, token: str, model: str,
                     allowlist: tuple[str, ...], timeout: int, max_mb: int) -> dict | None:
    """Fetch and transcribe a permitted RSS enclosure; never retrieves YouTube media."""
    if not endpoint or not host_is_allowed(url, allowlist):
        return None
    max_bytes = min(AUDIO_MAX_BYTES, max(1, max_mb) * 1024 * 1024)
    body, content_type, final_url = fetch_public(url, timeout=min(timeout, 90), max_bytes=max_bytes)
    mime = content_type.split(";", 1)[0].strip().lower()
    suffix = urlsplit(final_url).path.rsplit(".", 1)[-1].lower()
    if not (mime.startswith("audio/") or suffix in {"mp3", "m4a", "wav", "ogg", "opus", "webm"}):
        return None
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = httpx.post(endpoint, headers=headers,
                          data={"model": model, "response_format": "verbose_json"},
                          files={"file": (f"audio.{suffix or 'bin'}", io.BytesIO(body), mime or "application/octet-stream")},
                          timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    raw_segments = payload.get("segments") if isinstance(payload, dict) else None
    lines = []
    if isinstance(raw_segments, list):
        for segment in raw_segments:
            try:
                start = float(segment.get("start", 0))
            except (TypeError, ValueError, AttributeError):
                start = 0
            text = re.sub(r"\s+", " ", str(segment.get("text", ""))).strip() if isinstance(segment, dict) else ""
            if text:
                lines.append(f"[{format_time(start)}] {text}")
    if not lines:
        text = re.sub(r"\s+", " ", str(payload.get("text", ""))).strip() if isinstance(payload, dict) else ""
        if text:
            lines = [text]
    if not lines:
        return None
    return {"text": "\n".join(lines), "language": payload.get("language"),
            "timestamped": bool(raw_segments), "transcript_format": "asr_verbose_json",
            "transcript_url": None, "audio_url": final_url,
            "audio_sha256": hashlib.sha256(body).hexdigest()}
