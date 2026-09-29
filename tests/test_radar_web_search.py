from datetime import date
from types import SimpleNamespace

from app.radar.web_search import search_task
from app.radar.pipeline import select_search_locales


def test_spoken_task_marks_media_refs_and_does_not_ingest_youtube_html(monkeypatch):
    monkeypatch.setattr("app.radar.web_search.load_search_settings", lambda: SimpleNamespace(max_queries=3))
    monkeypatch.setattr("app.radar.web_search.web_search", lambda query: {
        "providers": [{"name": "searxng", "count": 2}],
        "results": [
            {"url": "https://www.youtube.com/watch?v=example", "title": "Conference talk",
             "content": "Transcript-like indexed page text " * 20, "language": "en", "provider": "searxng"},
            {"url": "https://events.example/transcript", "title": "Published transcript",
             "content": "Transcript text " * 40, "language": "en", "provider": "searxng"},
        ],
    })
    result = search_task({"branch": "Edge", "query": "talk transcript", "media_kind": "spoken",
                          "connector_label": "Поиск докладов, подкастов и транскриптов",
                          "queries": [{"language": "en", "query": "talk transcript"}]},
                         date(2026, 9, 29).isoformat())

    youtube, transcript = result["references"]
    assert youtube["media_kind"] == "spoken" and "document" not in youtube
    assert transcript["media_kind"] == "spoken" and transcript["document"]["extraction"] == "searxng_content"
    assert result["connector"] == "Поиск докладов, подкастов и транскриптов"
    assert result["query_count"] == 1 and result["query_languages"] == ["en"]


def test_broad_pass_covers_all_locales_and_followup_keeps_ru_en_and_user_language():
    localized = [{"language": code, "query": f"q-{code}"}
                 for code in ("en", "ru", "zh", "es", "fr", "de", "pt", "ja", "ko", "ar", "hi", "it", "tr", "id")]
    broad = select_search_locales(localized, "fr", broad=True)
    focused = select_search_locales(localized, "fr")
    assert len(broad) == 14
    assert {item["language"] for item in focused} == {"ru", "en", "fr"}


def test_root_web_search_can_exceed_global_cap_and_prioritizes_user_language(monkeypatch):
    seen = []
    monkeypatch.setattr("app.radar.web_search.load_search_settings", lambda: SimpleNamespace(max_queries=3))

    def search(query):
        seen.append(query.language)
        return {"providers": [], "results": []}

    monkeypatch.setattr("app.radar.web_search.web_search", search)
    codes = ("en", "ru", "zh", "es", "fr", "de", "pt", "ja", "ko", "ar", "hi", "it", "tr", "id")
    result = search_task({"branch": "Основная тема", "query": "many locales",
                          "priority_languages": ["fr"], "max_queries": 14,
                          "queries": [{"language": code, "query": f"query {code}"} for code in codes]},
                         date(2026, 9, 29).isoformat())
    assert result["query_count"] == 14
    assert len(set(seen)) == 14
    assert seen[:3] == ["ru", "en", "fr"]
