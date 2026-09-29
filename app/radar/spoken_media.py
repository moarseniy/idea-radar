"""Parsing helpers for creator-published podcast and conference transcripts."""
from __future__ import annotations

import html
import json
import re
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

TIME = r"(?:\d{1,2}:)?\d{2}:\d{2}(?:[.,]\d{1,3})?"
TIMESTAMP_LINE = re.compile(rf"^\s*({TIME})\s+-->\s+({TIME})(?:\s+.*)?$")
SPEAKER_TAG = re.compile(r"<v(?:\s+([^>]+))?>", re.I)
MARKUP = re.compile(r"</?(?:v|c|b|i|u|ruby|rt)(?:\s+[^>]*)?>", re.I)
TRANSCRIPT_EXTENSIONS = (".vtt", ".srt", ".txt", ".json")
TRANSCRIPT_HINTS = re.compile(r"transcript|caption|subtitle|captions|transcription|расшифров|субтитр|транскрип", re.I)
DISCOVERY_TERMS = {
    "en": "conference talk seminar webinar podcast interview transcript captions",
    "ru": "доклад конференция лекция семинар вебинар подкаст расшифровка субтитры",
    "zh": "会议报告 讲座 研讨会 网络研讨会 播客 文字稿 字幕",
    "es": "ponencia conferencia seminario webinar podcast entrevista transcripción subtítulos",
    "fr": "conférence exposé séminaire webinaire podcast entretien transcription sous-titres",
    "de": "Konferenzvortrag Seminar Webinar Podcast Interview Transkript Untertitel",
    "pt": "palestra conferência seminário webinar podcast entrevista transcrição legendas",
    "ja": "会議 講演 セミナー ウェビナー ポッドキャスト 文字起こし 字幕",
    "ko": "학회 발표 세미나 웨비나 팟캐스트 인터뷰 대본 자막",
    "ar": "محاضرة مؤتمر ندوة عبر الإنترنت بودكاست مقابلة تفريغ ترجمة",
    "hi": "सम्मेलन व्याख्यान वेबिनार पॉडकास्ट साक्षात्कार प्रतिलेख उपशीर्षक",
    "it": "conferenza intervento seminario webinar podcast intervista trascrizione sottotitoli",
    "tr": "konferans konuşması seminer webinar podcast röportaj döküm altyazı",
    "id": "konferensi kuliah seminar webinar podcast wawancara transkrip teks",
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1].lower()


def is_youtube_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com") or \
        host == "youtube-nocookie.com" or host.endswith(".youtube-nocookie.com")


def parse_time(value: str) -> float:
    text = value.strip().replace(",", ".")
    parts = text.split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    hours = int(parts[-3]) if len(parts) > 2 else 0
    return hours * 3600 + minutes * 60 + seconds


def format_time(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def parse_caption_text(text: str, content_type: str = "", url: str = "") -> dict | None:
    """Convert VTT/SRT/plain/Podcasting 2.0 JSON captions into timestamped text."""
    content_type, path = content_type.lower(), urlsplit(url).path.lower()
    if "json" in content_type or path.endswith(".json"):
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        raw_segments = data.get("segments") if isinstance(data, dict) else None
        if not isinstance(raw_segments, list):
            return None
        lines = []
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            try:
                start = float(item.get("startTime", item.get("start", 0)))
            except (TypeError, ValueError):
                start = 0
            body = re.sub(r"\s+", " ", str(item.get("body", item.get("text", "")))).strip()
            speaker = str(item.get("speaker", "")).strip()
            if body:
                lines.append(f"[{format_time(start)}] {speaker + ': ' if speaker else ''}{body}")
        return {"text": "\n".join(lines), "transcript_format": "json", "timestamped": bool(lines)}

    if "vtt" in content_type or path.endswith(".vtt") or "subrip" in content_type or path.endswith(".srt"):
        lines, cue_start, cue_text = [], None, []

        def flush():
            if cue_start is None:
                return
            body_parts, speaker = [], ""
            for raw in cue_text:
                speaker_match = SPEAKER_TAG.search(raw)
                if speaker_match and not speaker:
                    speaker = html.unescape((speaker_match.group(1) or "").strip())
                cleaned = html.unescape(MARKUP.sub("", raw)).strip()
                if cleaned:
                    body_parts.append(cleaned)
            body = re.sub(r"\s+", " ", " ".join(body_parts)).strip()
            if body:
                lines.append(f"[{format_time(cue_start)}] {speaker + ': ' if speaker else ''}{body}")

        for raw in text.lstrip("\ufeff").splitlines():
            line = raw.strip()
            match = TIMESTAMP_LINE.match(line)
            if match:
                flush()
                cue_start, cue_text = parse_time(match.group(1)), []
            elif cue_start is not None:
                if not line:
                    flush()
                    cue_start, cue_text = None, []
                elif not line.isdigit():
                    cue_text.append(line)
        flush()
        if lines:
            return {"text": "\n".join(lines), "transcript_format": "vtt" if "vtt" in content_type or path.endswith(".vtt") else "srt",
                    "timestamped": True}
        return None

    if "text/plain" in content_type or path.endswith(".txt"):
        cleaned = re.sub(r"\s+", " ", html.unescape(text)).strip()
        if len(cleaned) >= 150:
            return {"text": cleaned, "transcript_format": "text", "timestamped": False}
    return None


class _MediaLinks(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.transcripts: list[str] = []
        self.feeds: list[str] = []
        self._anchor_url = ""
        self._anchor_hint = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        href = attrs.get("href") or attrs.get("src")
        if not href:
            return
        absolute = urljoin(self.base_url, href)
        path = urlsplit(absolute).path.lower()
        rel = (attrs.get("rel") or "").lower()
        kind = (attrs.get("type") or "").lower()
        if tag == "link" and ("rss+xml" in kind or "atom+xml" in kind):
            self.feeds.append(absolute)
        elif tag == "track" and (attrs.get("kind", "").lower() in {"captions", "subtitles"} or
                                  any(ext in path for ext in TRANSCRIPT_EXTENSIONS)):
            self.transcripts.append(absolute)
        elif tag == "link" and ("transcript" in rel or "captions" in rel or "subtitles" in rel):
            self.transcripts.append(absolute)
        elif tag == "a":
            self._anchor_url = absolute
            self._anchor_hint = " ".join((href, attrs.get("title", "")))

    def handle_data(self, data):
        if self._anchor_url:
            self._anchor_hint += " " + data

    def handle_endtag(self, tag):
        if tag == "a" and self._anchor_url:
            path = urlsplit(self._anchor_url).path.lower()
            if any(path.endswith(ext) for ext in TRANSCRIPT_EXTENSIONS) or TRANSCRIPT_HINTS.search(self._anchor_hint):
                self.transcripts.append(self._anchor_url)
            self._anchor_url, self._anchor_hint = "", ""


def discover_transcript_links(body: bytes, base_url: str) -> dict[str, list[str]]:
    parser = _MediaLinks(base_url)
    try:
        parser.feed(body.decode("utf-8", errors="replace"))
    except Exception:
        return {"transcript_urls": [], "feed_urls": []}
    return {
        "transcript_urls": list(dict.fromkeys(parser.transcripts))[:5],
        "feed_urls": list(dict.fromkeys(parser.feeds))[:2],
    }


class _TranscriptHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.capture: str | None = None
        self.buffer: list[str] = []
        self.speaker = ""
        self.timestamp = ""
        self.rows: list[str] = []
        self.timestamped = False

    def handle_starttag(self, tag, _attrs):
        if tag in {"cite", "time", "p"}:
            self.capture, self.buffer = tag, []

    def handle_data(self, data):
        if self.capture:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag != self.capture:
            return
        value = re.sub(r"\s+", " ", " ".join(self.buffer)).strip()
        if tag == "cite":
            self.speaker = value.rstrip(":")
        elif tag == "time":
            self.timestamp = value
        elif tag == "p" and value:
            time_value = ""
            if re.fullmatch(TIME, self.timestamp):
                time_value = f"[{format_time(parse_time(self.timestamp))}] "
                self.timestamped = True
            speaker = f"{self.speaker}: " if self.speaker else ""
            self.rows.append(f"{time_value}{speaker}{value}")
            self.speaker, self.timestamp = "", ""
        self.capture, self.buffer = None, []


def _item_text(element, name: str) -> str:
    child = next((item for item in element if local_name(item.tag) == name), None)
    return (child.text or "").strip() if child is not None else ""


def _item_link(element, relation: str | None = None) -> str:
    for child in element:
        if local_name(child.tag) != "link":
            continue
        rel = child.attrib.get("rel", "alternate").lower()
        href = child.attrib.get("href", "").strip() or (child.text or "").strip()
        if href and (relation is None and rel in {"alternate", ""} or relation == rel):
            return href
    return ""


def _normalize_title(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold()))


def _episode_similarity(item_title: str, target_title: str, target_url: str, item_url: str, guid: str) -> float:
    if target_url and target_url.rstrip("/") in {item_url.rstrip("/"), guid.rstrip("/")}:
        return 1.0
    if not target_title:
        return 0.0
    left, right = _normalize_title(item_title), _normalize_title(target_title)
    if not left or not right:
        return 0.0
    left_tokens, right_tokens = set(left.split()), set(right.split())
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return max(SequenceMatcher(None, left, right).ratio(), overlap)


def _iso_date(value: str) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return value[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", value) else None


def resolve_podcast_feed(xml_text: str, *, target_url: str = "", target_title: str = "",
                         preferred_language: str = "") -> dict | None:
    """Pick one matching RSS episode and return its published transcript/audio links."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return None
    channel = next((item for item in root.iter() if local_name(item.tag) == "channel"), root)
    channel_language = _item_text(channel, "language") or channel.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "")
    channel_license = ""
    for element in channel.iter():
        if local_name(element.tag) == "license":
            channel_license = (element.text or "").strip() or element.attrib.get("url", "")
            break
    items = [element for element in root.iter() if local_name(element.tag) in {"item", "entry"}]
    choices = []
    for item in items:
        title = _item_text(item, "title")
        link = _item_link(item) or _item_text(item, "link")
        guid = _item_text(item, "guid")
        score = _episode_similarity(title, target_title, target_url, link, guid)
        if score < 0.58:
            continue
        transcripts = []
        audio = ""
        for element in item.iter():
            name = local_name(element.tag)
            if name == "transcript":
                transcript_url = element.attrib.get("url", "").strip()
                if transcript_url:
                    transcripts.append({"url": transcript_url,
                                        "type": element.attrib.get("type", "").lower(),
                                        "language": element.attrib.get("language", "").lower(),
                                        "rel": element.attrib.get("rel", "").lower()})
            elif name == "enclosure" and not audio:
                kind = element.attrib.get("type", "").lower()
                if kind.startswith("audio/") or re.search(r"\.(mp3|m4a|wav|ogg|opus)(?:$|\?)", element.attrib.get("url", ""), re.I):
                    audio = element.attrib.get("url", "").strip()
            elif name == "link" and not audio and element.attrib.get("rel", "").lower() == "enclosure":
                kind = element.attrib.get("type", "").lower()
                href = element.attrib.get("href", "").strip()
                if href and (kind.startswith("audio/") or re.search(
                        r"\.(mp3|m4a|wav|ogg|opus)(?:$|\?)", href, re.I)):
                    audio = href
        if not transcripts and not audio:
            continue
        transcripts.sort(key=lambda transcript: (
            0 if preferred_language and transcript["language"] == preferred_language.lower() else 1,
            0 if transcript["rel"] == "captions" else 1,
            0 if any(token in transcript["type"] for token in ("vtt", "subrip", "json")) else 1,
        ))
        choices.append((score, {"title": title, "url": link or guid or target_url,
                                "published_at": _iso_date(_item_text(item, "pubdate") or _item_text(item, "date")
                                                           or _item_text(item, "published") or _item_text(item, "updated")),
                                "language": channel_language, "license": channel_license,
                                "transcripts": transcripts, "audio_url": audio}))
    if not choices:
        return None
    choices.sort(key=lambda value: value[0], reverse=True)
    return choices[0][1]


def parse_html_transcript(body: bytes, url: str) -> dict | None:
    """Parse accessible transcript markup that uses speaker/time/p tags."""
    text = body.decode("utf-8", errors="replace")
    parts = urlsplit(url)
    if not TRANSCRIPT_HINTS.search(parts.path + " " + parts.query):
        return None
    structured = _TranscriptHTML()
    try:
        structured.feed(text)
    except Exception:
        pass
    if structured.rows:
        transcript = "\n".join(structured.rows)
        if len(transcript) >= 150:
            return {"text": transcript, "transcript_format": "html",
                    "timestamped": structured.timestamped}
    try:
        from trafilatura import extract
        result = extract(text, url=url, include_comments=False, include_tables=False)
    except Exception:
        result = None
    cleaned = re.sub(r"\s+", " ", result or "").strip()
    if len(cleaned) < 150:
        return None
    return {"text": cleaned, "transcript_format": "html", "timestamped": False}
