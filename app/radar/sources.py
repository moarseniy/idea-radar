from __future__ import annotations

import gzip
import hashlib
import http.client
import io
import ipaddress
import json
import re
import socket
import ssl
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from app.radar.resilience import CircuitBreaker
from app.radar.spoken_media import (discover_transcript_links, parse_caption_text,
                                    is_youtube_url, parse_html_transcript,
                                    resolve_podcast_feed)
from app.radar.store import now
from app.registry import classify

MAX_BYTES = 5 * 1024 * 1024
MAX_TEXT = 60_000
PARSER_VERSION = "source-parser-v2"
FETCH_RETRIES = 2


class TransientFetchError(RuntimeError):
    """A retryable upstream HTTP response while loading an article page."""


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("Only public HTTP(S) URLs without credentials are supported")
    if any(ord(c) < 32 for c in url) or parts.port not in {None, 80, 443}:
        raise ValueError("Unsupported URL")
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", urlencode(query), ""))


def _resolve_with_socket(host: str, timeout: float) -> set[str]:
    pool = ThreadPoolExecutor(1)
    try:
        infos = pool.submit(socket.getaddrinfo, host, None).result(timeout=max(0.1, timeout))
    except FutureTimeout as exc:
        raise TimeoutError("Истёк лимит DNS-разрешения источника") from exc
    except OSError as exc:
        raise ValueError("Адрес источника не разрешается") from exc
    finally:
        pool.shutdown(wait=False)
    return {info[4][0].split("%")[0] for info in infos}


def public_ip(host: str, port: int, timeout: float = 4.0) -> str:
    del port  # getent resolves the hostname once; the chosen IP is later pinned to the target port.
    try:
        resolved = subprocess.run(["getent", "ahosts", host], check=False, capture_output=True,
                                  text=True, timeout=max(0.1, timeout))
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Истёк лимит DNS-разрешения источника") from exc
    except FileNotFoundError:
        # getent есть в Linux (Docker), но не в macOS — тогда системный резолвер с тем же таймаутом.
        addresses = _resolve_with_socket(host, timeout)
    else:
        addresses = {line.split()[0] for line in resolved.stdout.splitlines() if line.split()}
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError("Адрес не является публичным интернет-источником")
    return sorted(addresses)[0]


def fetch_public(url: str, timeout: float = 12, *, max_bytes: int = MAX_BYTES) -> tuple[bytes, str, str]:
    """Resolve each redirect, reject private IPs and pin the actual TCP connection.

    TLS still verifies the original host. Environment proxies are not used.
    """
    deadline = time.monotonic() + timeout
    for _ in range(5):
        url = canonical_url(url)
        parts = urlsplit(url)
        host = parts.hostname.encode("idna").decode("ascii")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Истёк лимит загрузки страницы")
        ip = public_ip(host, port, remaining)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Истёк лимит загрузки страницы")
        if parts.scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=remaining, context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=remaining)
        # http.client uses this factory before HTTPS wraps the socket with host SNI.
        conn._create_connection = lambda address, timeout=remaining, source_address=None: socket.create_connection(
            (ip, port), timeout, source_address)
        try:
            path = urlunsplit(("", "", parts.path or "/", parts.query, ""))
            conn.request("GET", path, headers={"User-Agent": "IDEA-Research/1.0", "Accept-Encoding": "identity"})
            response = conn.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise ValueError("Редирект без адреса")
                url = urljoin(url, location)
                continue
            if response.status in {429, 500, 502, 503, 504}:
                raise TransientFetchError(f"HTTP {response.status}")
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}")
            if int(response.getheader("Content-Length") or 0) > max_bytes:
                raise ValueError(f"Источник превышает лимит {max_bytes // (1024 * 1024)} МБ")
            chunks, size = [], 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Истёк лимит загрузки страницы")
                if conn.sock:
                    conn.sock.settimeout(remaining)
                part = response.read(min(64 * 1024, max_bytes + 1 - size))
                if not part:
                    break
                chunks.append(part)
                size += len(part)
                if size > max_bytes:
                    raise ValueError(f"Источник превышает лимит {max_bytes // (1024 * 1024)} МБ")
            body = b"".join(chunks)
            if response.getheader("Content-Encoding", "").lower() == "gzip":
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as zipped:
                    body = zipped.read(MAX_BYTES + 1)
                if len(body) > max_bytes:
                    raise ValueError("Распакованный источник превышает лимит")
            return body, response.getheader("Content-Type", ""), url
        finally:
            conn.close()
    raise ValueError("Слишком много перенаправлений")


def source_type(url: str) -> tuple[str, str, str]:
    """Тип, доверие и основание — из реестра источников (app/registry/sources.json)."""
    c = classify(url)
    return c.type, c.trust, c.reason


def extract_document(body: bytes, content_type: str, url: str) -> dict:
    text = body.decode("utf-8", errors="replace")
    caption = parse_caption_text(text, content_type, url)
    if caption:
        return {**caption, "title": "", "date": None, "language": None,
                "extraction": f"transcript_{caption['transcript_format']}", "truncated": False}
    if "pdf" in content_type.lower() or body.startswith(b"%PDF"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(body))
        pages = [(page.extract_text() or "") for page in reader.pages[:20]]
        return {"text": "\n\n".join(f"[стр. {i+1}] {t}" for i, t in enumerate(pages))[:MAX_TEXT],
                "title": str((reader.metadata or {}).get("/Title") or ""), "date": None,
                "language": None, "extraction": "pdf_text", "truncated": len(reader.pages) > 20}
    if "text/plain" in content_type.lower():
        return {"text": text[:MAX_TEXT], "title": "", "date": None,
                "language": None, "extraction": "plain_text", "truncated": len(body) > MAX_TEXT}
    if "html" not in content_type.lower() and "xhtml" not in content_type.lower():
        raise ValueError("Поддерживаются HTML, текст и PDF")
    transcript = parse_html_transcript(body, url)
    if transcript:
        return {**transcript, "title": "", "date": None, "language": None,
                "extraction": "transcript_html", "truncated": False}
    import trafilatura
    raw = trafilatura.extract(body, url=url, output_format="json", with_metadata=True,
                              include_comments=False, include_tables=True)
    if not raw:
        raise ValueError("Не удалось извлечь содержательный текст страницы")
    item = json.loads(raw)
    text = item.get("text") or ""
    return {"text": text[:MAX_TEXT], "title": item.get("title") or "", "date": item.get("date"),
            "language": item.get("language"), "extraction": "html_text", "truncated": len(text) > MAX_TEXT}


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def quote_present(quote: str, text: str) -> bool:
    return len(quote.strip()) >= 20 and normalized_text(quote) in normalized_text(text)


def language_hint(text: str) -> str:
    ru = len(re.findall(r"[А-Яа-яЁё]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if ru > max(latin, 20):
        return "ru (эвристика)"
    # Latin script alone cannot distinguish English from other Latin languages.
    return "не определён"


class SourceFetcher:
    def __init__(self, cache_dir: Path, timeout=12, cache_hours=24, fetch=fetch_public,
                 audio_transcription_url: str = "", audio_transcription_token: str = "",
                 audio_transcription_model: str = "whisper-1", audio_transcription_allowlist=(),
                 audio_transcription_timeout: int = 180, audio_transcription_max_mb: int = 20):
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.cache_hours = cache_hours
        self.fetch = fetch
        self.audio_transcription_url = audio_transcription_url
        self.audio_transcription_token = audio_transcription_token
        self.audio_transcription_model = audio_transcription_model
        self.audio_transcription_allowlist = tuple(audio_transcription_allowlist)
        self.audio_transcription_timeout = audio_transcription_timeout
        self.audio_transcription_max_mb = audio_transcription_max_mb
        self.breakers: dict[str, CircuitBreaker] = {}
        self.breakers_lock = threading.Lock()

    def _fetch_with_retry(self, url: str, max_bytes: int | None = None) -> tuple[bytes, str, str]:
        host = urlsplit(canonical_url(url)).hostname or "unknown"
        with self.breakers_lock:
            breaker = self.breakers.setdefault(host, CircuitBreaker(threshold=4, reset_after=45))
        breaker.before_call()
        deadline = time.monotonic() + self.timeout
        for attempt in range(FETCH_RETRIES + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                breaker.failure()
                raise TimeoutError("Исчерпан общий лимит загрузки страницы")
            try:
                if max_bytes and self.fetch is fetch_public:
                    result = self.fetch(url, remaining, max_bytes=max_bytes)
                    if len(result[0]) > max_bytes:
                        raise ValueError("Источник превышает настроенный размер")
                else:
                    result = self.fetch(url, remaining)
                breaker.success()
                return result
            except (TransientFetchError, TimeoutError, OSError, http.client.HTTPException):
                if attempt >= FETCH_RETRIES or time.monotonic() >= deadline:
                    breaker.failure()
                    raise
                time.sleep(min(0.2 * (2 ** attempt), max(0.0, deadline - time.monotonic())))
            except Exception:
                # Parsing, URL-policy and permanent HTTP errors are not retried.
                breaker.success()
                raise
        raise RuntimeError("Исчерпаны повторные попытки загрузки страницы")

    def _spoken_document(self, reference: dict, url: str, body: bytes,
                         content_type: str, final_url: str) -> tuple[dict, str, dict] | None:
        if reference.get("media_kind") != "spoken":
            return None
        hints = discover_transcript_links(body, final_url) if "html" in content_type.lower() else {}
        feed_candidates = list(hints.get("feed_urls", []))
        is_feed = any(token in content_type.lower() for token in ("rss", "atom", "xml")) or \
            body.lstrip().startswith((b"<?xml", b"<rss", b"<feed"))
        if is_feed:
            feed_candidates.insert(0, final_url)
        for transcript_url in hints.get("transcript_urls", []):
            try:
                transcript_body, transcript_type, transcript_final_url = self._fetch_with_retry(transcript_url)
                caption = parse_caption_text(transcript_body.decode("utf-8", errors="replace"),
                                             transcript_type, transcript_final_url)
                if not caption and "html" in transcript_type.lower():
                    html_doc = extract_document(transcript_body, transcript_type, transcript_final_url)
                    if len(html_doc.get("text", "")) >= 150:
                        caption = {**html_doc, "transcript_format": "html", "timestamped": False}
                if caption and len(caption["text"].strip()) >= 150:
                    return ({**caption, "title": "", "date": reference.get("published_at"),
                             "language": reference.get("source_language"),
                             "extraction": f"transcript_{caption['transcript_format']}",
                             "truncated": False, "final_url": final_url}, transcript_type,
                            {"transcript_url": transcript_final_url, "transcript_origin": "page_caption_link",
                             "transcript_timestamped": caption.get("timestamped", False),
                             "transcript_sha256": hashlib.sha256(transcript_body).hexdigest()})
            except Exception:
                continue
        for feed_url in dict.fromkeys(feed_candidates):
            try:
                feed_body, feed_type, feed_final_url = self._fetch_with_retry(feed_url)
                feed = resolve_podcast_feed(
                    feed_body.decode("utf-8", errors="replace"), target_url=url,
                    target_title=reference.get("title", ""),
                    preferred_language=reference.get("source_language", ""),
                )
                if not feed:
                    continue
                for transcript in feed.get("transcripts", []):
                    transcript_url = urljoin(feed_final_url, transcript["url"])
                    try:
                        transcript_body, transcript_type, transcript_final_url = self._fetch_with_retry(transcript_url)
                        if transcript_type in {"application/octet-stream", "binary/octet-stream"}:
                            transcript_type = transcript.get("type", "")
                        caption = parse_caption_text(transcript_body.decode("utf-8", errors="replace"),
                                                     transcript_type or transcript.get("type", ""), transcript_final_url)
                        if not caption and "html" in transcript_type.lower():
                            html_doc = extract_document(transcript_body, transcript_type, transcript_final_url)
                            if len(html_doc.get("text", "")) >= 150:
                                caption = {**html_doc, "transcript_format": "html", "timestamped": False}
                    except Exception:
                        continue
                    if caption and len(caption["text"].strip()) >= 150:
                        return ({**caption, "title": feed.get("title") or reference.get("title", ""),
                                 "date": feed.get("published_at") or reference.get("published_at"),
                                 "language": transcript.get("language") or feed.get("language"),
                                 "extraction": f"podcast_transcript_{caption['transcript_format']}",
                                 "truncated": False, "final_url": final_url}, transcript_type,
                                {"transcript_url": transcript_final_url,
                                 "transcript_origin": "podcast_rss",
                                 "transcript_format": caption["transcript_format"],
                                 "transcript_timestamped": caption.get("timestamped", False),
                                 "transcript_sha256": hashlib.sha256(transcript_body).hexdigest(),
                                 "media_license": feed.get("license"),
                                 "audio_url": feed.get("audio_url"),
                                 "episode_url": feed.get("url") or url,
                                 "feed_url": feed_final_url})
                audio_url = feed.get("audio_url")
                if audio_url and self.audio_transcription_url:
                    from app.radar.audio_transcription import transcribe_audio
                    transcript = transcribe_audio(
                        urljoin(feed_final_url, audio_url), endpoint=self.audio_transcription_url,
                        token=self.audio_transcription_token, model=self.audio_transcription_model,
                        allowlist=self.audio_transcription_allowlist,
                        timeout=self.audio_transcription_timeout, max_mb=self.audio_transcription_max_mb,
                    )
                    if transcript and len(transcript["text"].strip()) >= 150:
                        return ({"text": transcript["text"], "title": feed.get("title") or reference.get("title", ""),
                                 "date": feed.get("published_at") or reference.get("published_at"),
                                 "language": transcript.get("language") or feed.get("language"),
                                 "extraction": "audio_asr_transcription", "truncated": False,
                                 "final_url": final_url}, "application/x-asr-transcript",
                                {"transcript_origin": "asr_allowlisted_audio",
                                 "transcript_format": transcript["transcript_format"],
                                 "transcript_timestamped": transcript["timestamped"],
                                 "transcript_sha256": transcript["audio_sha256"],
                                 "media_license": feed.get("license"),
                                 "audio_url": transcript["audio_url"],
                                 "episode_url": feed.get("url") or url,
                                 "feed_url": feed_final_url})
            except Exception:
                continue
        return None

    def load(self, reference: dict, as_of: date) -> dict:
        url = canonical_url(reference["url"])
        digest = hashlib.sha256(url.encode()).hexdigest()
        cache_path = self.cache_dir / f"{digest}.json"
        registry = classify(url)
        source = {"id": "src_" + digest[:16], "url": url, "title": reference.get("title") or url,
                  "publisher": registry.name or urlsplit(url).hostname, "type": registry.type,
                  "type_label": registry.type_label, "trust": registry.trust, "trust_reason": registry.reason,
                  "primary_only": registry.primary_only, "registry_match": registry.matched,
                  "language": "не определён", "published_at": reference.get("published_at"), "retrieved_at": now(),
                  "source_language": reference.get("source_language"),
                  "query_language": reference.get("query_language"),
                  "media_kind": reference.get("media_kind"),
                  "transcript_url": reference.get("transcript_url"),
                  "transcript_origin": reference.get("transcript_origin"),
                  "transcript_format": reference.get("transcript_format"),
                  "transcript_timestamped": reference.get("transcript_timestamped"),
                  "media_license": reference.get("media_license"),
                  "audio_url": reference.get("audio_url"),
                  "source_country": reference.get("source_country"),
                  "source_provider": reference.get("source_provider"),
                  "structured_event": reference.get("structured_event"),
                  "adoption_metrics": reference.get("adoption_metrics"),
                  "text": "", "status": "failed", "error": None, "cached": False,
                  "parse_status": "pending", "content_type": None,
                  "ingest_sha256": None, "parser_version": PARSER_VERSION,
                  "summary_generated": True, "discovered_by": reference.get("branch", ""),
                  "duplicate_of": None, "date_status": "unknown"}
        if reference.get("media_kind") == "spoken" and is_youtube_url(url):
            source.update(error=("Видео найдено на YouTube. Для чтения транскрипта нужен доступ владельца "
                                 "к субтитрам; содержимое видео не извлекалось."),
                          parse_status="unavailable")
            return source
        try:
            transcript_meta = {}
            if reference.get("document"):
                # Научная работа из API (arXiv, Crossref): документ — аннотация, страница не нужна.
                doc = dict(reference["document"])
                doc.setdefault("truncated", False)
                content_type = "application/x-api-document"
                ingest_sha256 = hashlib.sha256(str(doc.get("text", "")).encode("utf-8")).hexdigest()
            elif self.cache_hours and cache_path.exists() and time.time() - cache_path.stat().st_mtime < self.cache_hours * 3600:
                cached = json.loads(cache_path.read_text())
                if cached.get("parser_version") == PARSER_VERSION:
                    doc = cached["document"]
                    transcript_meta = cached.get("transcript_meta", {})
                    source.update(transcript_meta)
                    source["retrieved_at"] = cached["retrieved_at"]
                    source["cached"] = True
                    content_type = cached.get("content_type")
                    ingest_sha256 = cached.get("ingest_sha256")
                else:
                    body, content_type, final_url = self._fetch_with_retry(url)
                    ingest_sha256 = hashlib.sha256(body).hexdigest()
                    spoken = self._spoken_document(reference, url, body, content_type, final_url)
                    doc, content_type, transcript_meta = spoken if spoken else (
                        extract_document(body, content_type, final_url), content_type, {})
                    source.update(transcript_meta)
                    doc["final_url"] = final_url
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    temp = cache_path.with_suffix(f".{time.time_ns()}.tmp")
                    temp.write_text(json.dumps({"document": doc, "retrieved_at": source["retrieved_at"],
                                                "content_type": content_type, "ingest_sha256": ingest_sha256,
                                                "parser_version": PARSER_VERSION,
                                                "transcript_meta": transcript_meta}, ensure_ascii=False))
                    temp.replace(cache_path)
            else:
                direct_audio = reference.get("media_kind") == "spoken" and \
                    bool(re.search(r"\.(?:mp3|m4a|wav|ogg|opus|webm)(?:$|\?)", url, re.I))
                if direct_audio and self.audio_transcription_url:
                    from app.radar.audio_transcription import transcribe_audio
                    transcript = transcribe_audio(
                        url, endpoint=self.audio_transcription_url, token=self.audio_transcription_token,
                        model=self.audio_transcription_model, allowlist=self.audio_transcription_allowlist,
                        timeout=self.audio_transcription_timeout, max_mb=self.audio_transcription_max_mb,
                    )
                else:
                    transcript = None
                if transcript:
                    doc = {"text": transcript["text"], "title": "", "date": reference.get("published_at"),
                           "language": transcript.get("language") or reference.get("source_language"),
                           "extraction": "audio_asr_transcription", "truncated": False, "final_url": url}
                    content_type = "application/x-asr-transcript"
                    ingest_sha256 = transcript["audio_sha256"]
                    transcript_meta = {"transcript_origin": "asr_allowlisted_audio",
                                       "transcript_format": transcript["transcript_format"],
                                       "transcript_timestamped": transcript["timestamped"],
                                       "audio_url": transcript["audio_url"]}
                else:
                    body, content_type, final_url = self._fetch_with_retry(url)
                    ingest_sha256 = hashlib.sha256(body).hexdigest()
                    spoken = self._spoken_document(reference, url, body, content_type, final_url)
                    doc, content_type, transcript_meta = spoken if spoken else (
                        extract_document(body, content_type, final_url), content_type, {})
                    if doc.get("extraction", "").startswith("transcript_"):
                        transcript_meta.setdefault("transcript_url", final_url)
                        transcript_meta.setdefault("transcript_origin", "published_caption_file")
                        transcript_meta.setdefault("transcript_format", doc.get("transcript_format"))
                        transcript_meta.setdefault("transcript_timestamped", doc.get("timestamped", False))
                source.update(transcript_meta)
                doc.setdefault("final_url", url)
                if len(doc["text"].strip()) < 150:
                    raise ValueError("Слишком мало текста для проверки утверждений")
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                temp = cache_path.with_suffix(f".{time.time_ns()}.tmp")
                temp.write_text(json.dumps({"document": doc, "retrieved_at": source["retrieved_at"],
                                            "content_type": content_type, "ingest_sha256": ingest_sha256,
                                            "parser_version": PARSER_VERSION, "transcript_meta": transcript_meta},
                                           ensure_ascii=False))
                temp.replace(cache_path)
            if transcript_meta.get("transcript_sha256"):
                ingest_sha256 = transcript_meta["transcript_sha256"]
            source.update(text=doc["text"], title=doc["title"] or source["title"],
                          language=doc.get("language") or language_hint(doc["text"]),
                          published_at=doc.get("date") or source.get("published_at"), extraction=doc["extraction"],
                          content_type=content_type, ingest_sha256=ingest_sha256,
                          parse_status="parsed",
                          truncated=doc.get("truncated", False), final_url=doc.get("final_url", url), status="read")
            source["content_hash"] = hashlib.sha256(normalized_text(source["text"]).encode()).hexdigest()
            if source["published_at"]:
                try:
                    publication_date = date.fromisoformat(str(source["published_at"])[:10])
                    source["date_status"] = "future" if publication_date > as_of else "known"
                    if publication_date > as_of:
                        source.update(status="excluded_date", error="Публикация позднее даты оценки")
                except ValueError:
                    source["published_at"] = None
            if as_of < date.today() and source["date_status"] == "unknown":
                source.update(status="excluded_date", error="Дата не установлена: источник исключён из исторической оценки")
        except Exception as exc:
            source["error"] = str(exc)[:240]
            source["parse_status"] = "failed"
        return source


def public_source(source: dict) -> dict:
    return {key: value for key, value in source.items() if key != "text"}


def mark_duplicates(sources: list[dict]) -> None:
    """Conservative exact/near-copy detection; different domains ≠ independent evidence."""
    seen = []
    for source in sources:
        if source["status"] != "read":
            continue
        words = normalized_text(source["text"]).split()[:5000]
        shingles = {" ".join(words[i:i+5]) for i in range(0, len(words)-4, 3)}
        for other, other_shingles in seen:
            overlap = len(shingles & other_shingles) / max(1, min(len(shingles), len(other_shingles)))
            if source["content_hash"] == other["content_hash"] or (len(shingles) > 40 and overlap > .82):
                source["duplicate_of"] = other["id"]
                break
        if not source["duplicate_of"]:
            seen.append((source, shingles))
