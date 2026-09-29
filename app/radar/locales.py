"""Priority locales shared by live radar search and training-data collection."""
from __future__ import annotations

import re
from hashlib import sha256

# Broad global coverage for research and technology news. This is an extensible
# launch set, not a claim that every written language is covered.
LANGUAGES = (
    ("en", "English", "en-US", "US"),
    ("ru", "Russian", "ru-RU", "RU"),
    ("zh", "Simplified Chinese", "zh-CN", "CN"),
    ("es", "Spanish", "es-ES", "ES"),
    ("fr", "French", "fr-FR", "FR"),
    ("de", "German", "de-DE", "DE"),
    ("pt", "Portuguese", "pt-BR", "BR"),
    ("ja", "Japanese", "ja-JP", "JP"),
    ("ko", "Korean", "ko-KR", "KR"),
    ("ar", "Arabic", "ar-SA", "SA"),
    ("hi", "Hindi", "hi-IN", "IN"),
    ("it", "Italian", "it-IT", "IT"),
    ("tr", "Turkish", "tr-TR", "TR"),
    ("id", "Indonesian", "id-ID", "ID"),
)
LANGUAGE_NAMES = {code: name for code, name, _, _ in LANGUAGES}
LANGUAGE_MARKETS = {code: (market, country) for code, _, market, country in LANGUAGES}
LANGUAGE_CODES = tuple(LANGUAGE_NAMES)

_SIGNAL_META_TERMS = re.compile(r"\b(?:слаб\w*[\s-]+сигнал\w*|weak[\s-]+signals?)\b", re.I)
_EDGE_WORDS = re.compile(
    r"^(?:(?:и|а|в|во|на|для|по|о|об|про|of|in|on|for|about|and|the)\s+)+|"
    r"(?:\s+(?:и|а|в|во|на|для|по|о|об|про|of|in|on|for|about|and|the))+$",
    re.I,
)
_EMPTY_QUERY_FALLBACKS = {
    "ru": "технологические исследования и разработки",
    "en": "technology research and development",
    "zh": "技术研究与开发",
    "es": "investigación y desarrollo tecnológico",
    "fr": "recherche et développement technologique",
    "de": "Technologieforschung und -entwicklung",
    "pt": "pesquisa e desenvolvimento tecnológico",
    "ja": "技術研究開発",
    "ko": "기술 연구 개발",
    "ar": "البحث والتطوير التكنولوجي",
    "hi": "प्रौद्योगिकी अनुसंधान और विकास",
    "it": "ricerca e sviluppo tecnologico",
    "tr": "teknoloji araştırma ve geliştirme",
    "id": "penelitian dan pengembangan teknologi",
}


def clean_search_query(query: str, language: str = "und") -> str:
    """Remove classifier/meta wording from the text sent to web and source search."""
    cleaned = _SIGNAL_META_TERMS.sub(" ", str(query or ""))
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"^[\s,.;:!?–—-]+|[\s,.;:!?–—-]+$", "", cleaned)
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = _EDGE_WORDS.sub("", cleaned).strip(" ,.;:!?–—-")
    cleaned = " ".join(cleaned.split())
    if cleaned:
        return cleaned[:400]
    return _EMPTY_QUERY_FALLBACKS.get(language.lower(), _EMPTY_QUERY_FALLBACKS["en"])


def normalize_queries(query_ru: str, query_en: str, localized: list[dict] | None) -> list[dict]:
    """Validate model-generated locale variants and retain RU/EN fallbacks."""
    found: dict[str, str] = {}
    for entry in localized or []:
        code = str(entry.get("language", "")).lower().strip()
        query = clean_search_query(str(entry.get("query", "")), code)
        if re.fullmatch(r"[a-z]{2,3}", code) and query:
            found.setdefault(code, query[:400])
    found.setdefault("ru", clean_search_query(query_ru, "ru"))
    found.setdefault("en", clean_search_query(query_en, "en"))
    ordered = list(LANGUAGE_CODES) + [code for code in found if code not in LANGUAGE_NAMES]
    return [{"language": code, "language_name": LANGUAGE_NAMES.get(code, code.upper()), "query": found[code]}
            for code in ordered if found.get(code)]


def training_language_batch(key: str, width: int = 6) -> tuple[str, ...]:
    """Rotate training-search locales so the mined corpus spans all languages
    without multiplying every paid search angle by the full locale count.
    English and Russian are always included; other locales rotate deterministically.
    """
    width = max(2, min(int(width), len(LANGUAGE_CODES)))
    extra = [code for code in LANGUAGE_CODES if code not in {"en", "ru"}]
    start = int(sha256(key.encode("utf-8")).hexdigest()[:8], 16) % len(extra)
    need = width - 2
    rotating = [extra[(start + i) % len(extra)] for i in range(need)]
    return ("en", "ru", *rotating)
