from datetime import date

import httpx

from app.radar.audio_transcription import host_is_allowed, transcribe_audio
from app.radar.pipeline import spoken_media_search_task
from app.radar.sources import SourceFetcher, public_source
from app.radar.spoken_media import (discover_transcript_links, parse_caption_text,
                                    parse_html_transcript, resolve_podcast_feed)
from app.radar.models import Branch


VTT = """WEBVTT

00:00:05.000 --> 00:00:08.000
<v Ada>We built an experimental low-power sensor for industrial monitoring.

00:00:09.000 --> 00:00:13.000
The first pilot runs at two manufacturing sites and the team is measuring reliability.
"""


def test_vtt_transcript_keeps_timestamps_and_speaker():
    result = parse_caption_text(VTT, "text/vtt", "https://conference.example/talk.vtt")
    assert result["timestamped"] is True
    assert result["transcript_format"] == "vtt"
    assert "[00:05] Ada: We built an experimental" in result["text"]
    assert "[00:09] The first pilot runs" in result["text"]


def test_html_page_finds_caption_and_rss_links():
    html = b'''<html><head>
      <link rel="alternate" type="application/rss+xml" href="/feed.xml">
      <track kind="captions" src="/captions.vtt">
      </head></html>'''
    result = discover_transcript_links(html, "https://events.example/session")
    assert result["feed_urls"] == ["https://events.example/feed.xml"]
    assert result["transcript_urls"] == ["https://events.example/captions.vtt"]


def test_transcript_html_keeps_speaker_and_time_tags():
    body = ("<html><body><cite>Ada Lovelace</cite><time>01:02</time><p>" +
            "The team demonstrated a new technical process in a small factory pilot. " * 3 +
            "</p></body></html>").encode()
    result = parse_html_transcript(body, "https://podcast.example/transcript.html")
    assert result["timestamped"] is True
    assert result["text"].startswith("[01:02] Ada Lovelace:")


def test_rss_resolves_matching_episode_transcript_and_enclosure():
    feed = '''<?xml version="1.0"?><rss xmlns:podcast="https://podcastindex.org/namespace/1.0"><channel>
      <title>Emerging Systems Conference</title><language>en-us</language>
      <podcast:license>CC BY 4.0</podcast:license>
      <item><title>Low-power sensor pilot in manufacturing</title>
      <link>https://events.example/sessions/sensor-pilot</link>
      <pubDate>Thu, 20 Feb 2025 10:00:00 GMT</pubDate>
      <enclosure url="https://media.example/sensor-pilot.mp3" type="audio/mpeg" />
      <podcast:transcript url="https://events.example/sensor-pilot.vtt" type="text/vtt" language="en" rel="captions" />
      </item></channel></rss>'''
    result = resolve_podcast_feed(feed, target_url="https://events.example/sessions/sensor-pilot",
                                  target_title="Low-power sensor pilot in manufacturing",
                                  preferred_language="en")
    assert result["published_at"] == "2025-02-20"
    assert result["transcripts"][0]["url"].endswith("sensor-pilot.vtt")
    assert result["audio_url"].endswith("sensor-pilot.mp3")
    assert result["license"] == "CC BY 4.0"


def test_atom_feed_resolves_caption_link():
    feed = '''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:podcast="https://podcastindex.org/namespace/1.0" xml:lang="de">
      <title>Forschungskonferenz</title><entry><title>Industrie-Sensor Pilot</title>
      <link rel="alternate" href="https://uni.example/talks/sensor-pilot" />
      <published>2025-02-20T10:00:00Z</published>
      <podcast:transcript url="https://uni.example/sensor.vtt" type="text/vtt" language="de" />
      </entry></feed>'''
    result = resolve_podcast_feed(feed, target_title="Industrie-Sensor Pilot", preferred_language="de")
    assert result["language"] == "de"
    assert result["published_at"] == "2025-02-20"
    assert result["transcripts"][0]["language"] == "de"


def test_fetcher_reads_creator_published_rss_transcript(tmp_path):
    page = b'''<html><head><link rel="alternate" type="application/rss+xml" href="/feed.xml"></head></html>'''
    feed = b'''<rss xmlns:podcast="https://podcastindex.org/namespace/1.0"><channel>
      <title>Example conference</title><language>en</language><item>
      <title>Sensor pilot</title><link>https://events.example/sensor-pilot</link>
      <pubDate>Thu, 20 Feb 2025 10:00:00 GMT</pubDate>
      <podcast:transcript url="https://events.example/sensor-pilot.vtt" type="text/vtt" language="en" />
      </item></channel></rss>'''
    captions = VTT.encode() * 5
    documents = {
        "https://events.example/sensor-pilot": (page, "text/html; charset=utf-8"),
        "https://events.example/feed.xml": (feed, "application/rss+xml"),
        "https://events.example/sensor-pilot.vtt": (captions, "text/vtt"),
    }

    def fetch(url, _timeout):
        body, content_type = documents[url]
        return body, content_type, url

    fetcher = SourceFetcher(tmp_path / "cache", timeout=3, cache_hours=0, fetch=fetch)
    source = fetcher.load({"url": "https://events.example/sensor-pilot",
                           "title": "Sensor pilot", "branch": "Edge",
                           "media_kind": "spoken", "source_language": "en"},
                          date(2026, 9, 29))

    assert source["status"] == "read"
    assert source["extraction"] == "podcast_transcript_vtt"
    assert source["transcript_origin"] == "podcast_rss"
    assert source["transcript_timestamped"] is True
    assert source["transcript_url"] == "https://events.example/sensor-pilot.vtt"
    assert source["published_at"] == "2025-02-20"
    assert "[00:05] Ada:" in source["text"]
    assert "text" not in public_source(source)


def test_youtube_video_is_discovered_but_not_fetched_for_transcript(tmp_path):
    def fetch(*_args):
        raise AssertionError("YouTube watch page must not be fetched")

    fetcher = SourceFetcher(tmp_path / "cache", timeout=3, cache_hours=0, fetch=fetch)
    source = fetcher.load({"url": "https://www.youtube.com/watch?v=abc123",
                           "title": "Conference talk", "branch": "AI",
                           "media_kind": "spoken"}, date(2026, 9, 29))
    assert source["status"] == "failed"
    assert source["parse_status"] == "unavailable"
    assert "доступ владельца" in source["error"]


def test_spoken_search_uses_original_language_and_ru_en_terms():
    task = spoken_media_search_task(
        Branch(topic="Sensor deployments", query_ru="датчики на заводах", query_en="industrial sensors"),
        [{"language": "en", "query": "industrial sensors"},
         {"language": "ru", "query": "датчики на заводах"},
         {"language": "fr", "query": "capteurs industriels"}],
        "fr",
    )
    assert [item["language"] for item in task["queries"]] == ["ru", "en", "fr"]
    assert "transcription" in task["queries"][2]["query"]
    assert task["media_kind"] == "spoken"
    broad = spoken_media_search_task(
        Branch(topic="Sensor deployments", query_ru="датчики на заводах", query_en="industrial sensors"),
        [{"language": code, "query": f"topic {code}"}
         for code in ("en", "ru", "zh", "es", "fr", "de", "pt", "ja", "ko", "ar", "hi", "it", "tr", "id")],
        "fr", broad=True,
    )
    assert broad["max_queries"] == 14
    assert broad["per_query"] == 2


def test_asr_requires_allowlisted_non_youtube_host(monkeypatch):
    assert host_is_allowed("https://media.example/show.mp3", ("example",))
    assert not host_is_allowed("https://youtu.be/video", ("youtube.com", "youtu.be"))
    assert transcribe_audio("https://youtu.be/video", endpoint="http://whisper/v1/audio/transcriptions",
                            token="", model="whisper", allowlist=("youtube.com", "youtu.be"),
                            timeout=30, max_mb=20) is None

    monkeypatch.setattr("app.radar.audio_transcription.fetch_public",
                        lambda *_args, **_kwargs: (b"audio-bytes", "audio/mpeg", "https://media.example/e.mp3"))

    def mock_post(_url, **kwargs):
        assert kwargs["data"]["response_format"] == "verbose_json"
        return httpx.Response(200, json={"language": "en", "segments": [{"start": 65.0, "text": "A technical pilot."}]},
                              request=httpx.Request("POST", _url))

    monkeypatch.setattr("app.radar.audio_transcription.httpx.post", mock_post)
    result = transcribe_audio("https://media.example/e.mp3", endpoint="http://whisper/v1/audio/transcriptions",
                              token="", model="whisper", allowlist=("media.example",),
                              timeout=30, max_mb=20)
    assert result["text"] == "[01:05] A technical pilot."
    assert result["timestamped"] is True
