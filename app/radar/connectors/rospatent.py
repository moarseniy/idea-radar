"""Роспатент: поисковая платформа ФИПС (патенты РФ, СНГ и мировые массивы), ключ API.

Поиск на естественном языке (``qn``) по короткой фразе: английские фразы плана и русские
формулировки темы. Публикация отсекается на стороне API фильтром ``date_published``
(окно LOOKBACK_YEARS до as_of включительно), поэтому date_support = "strict".

Сначала ищем в массивах РФ и СНГ — то, чего нет у Google Patents; мировые массивы
платформы — только на оставшиеся места. Семейство (US, EP, WO одного изобретения)
склеивается по названию.

Дата записи — дата приоритета (иначе подачи): когда идея была заявлена, как у
коннектора Google/EPO. Смысловой поиск находит и соседние изобретения ("prompt injection
detection" → контроль литья под давлением), поэтому слова полной фразы — не меньше
MIN_SHARE — должны стоять рядом в названии (ru или en). Фрагмент описания для этой
проверки не годится: в нём находится недостающее слово ("prompt the user…").

Формат запроса и ответа — по «Открытые API ИС Поисковая платформа» (online.rospatent.gov.ru,
раздел «Открытые данные»). Ключ — ROSPATENT_API_KEY; без него коннектор выключен.
"""
from __future__ import annotations

import html
import re
from datetime import date
from typing import Any

from app.ml.env import env
from app.ml.features.evidence import years_before
from app.radar.connectors import Connector
from app.radar.connectors._terms import (
    CYRILLIC,
    NEAR,
    ReadyPhrases,
    relevant,
    short_phrases,
)
from app.radar.research_apis import _document, _json_request, limiter

URL = "https://searchplatform.rospatent.gov.ru/patsearch/v0.2/search"
DOC_URL = "https://searchplatform.rospatent.gov.ru/docs/"
LOOKBACK_YEARS = 10
PER_QUERY = 10
MIN_SHARE = 0.75  # из трёх слов фразы — все три, из четырёх — три
RU_DATASETS = ["ru_since_1994", "cis"]
WORLD_DATASETS = ["us", "ep", "pct", "cn", "jp", "kr", "gb", "de", "fr", "ca", "au", "ch", "at", "ap"]

limiter("rospatent", 1.0)

_TAG = re.compile(r"<[^>]+>")


def _enabled(_terms: list[str]) -> bool:
    return bool(env("ROSPATENT_API_KEY"))


def _clean(text: Any) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", str(text or ""))).split())


def _day(value: Any) -> str | None:
    """Дата платформы ``2013.09.30`` → ISO; иначе None."""
    match = re.fullmatch(r"(\d{4})[.\-](\d{2})[.\-](\d{2})", str(value or "").strip())
    return "-".join(match.groups()) if match else None


def _phrases(terms: list[str]) -> list[str]:
    """До двух латинских фраз без общих слов и до двух русских формулировок как есть.

    Поиск смысловой: ядра из двух слов ("quantum key") ему не нужны и только множат запросы.
    """
    latin = short_phrases(ReadyPhrases(term for term in terms if not CYRILLIC.search(term)), limit=2)
    russian = [term.strip() for term in terms if CYRILLIC.search(term) and term.strip()][:2]
    return list(dict.fromkeys([*latin, *russian]))


def _body(phrase: str, as_of: date, datasets: list[str]) -> dict:
    window = {"gte": years_before(as_of, LOOKBACK_YEARS).strftime("%Y%m%d"), "lte": as_of.strftime("%Y%m%d")}
    return {"qn": phrase, "limit": PER_QUERY, "datasets": datasets,
            "filter": {"date_published": {"range": window}}}


def _ref(hit: dict, phrases: list[str], as_of: date) -> dict | None:
    common = hit.get("common") or {}
    biblio = hit.get("biblio") or {}
    snippet = hit.get("snippet") or {}
    doc_id = str(hit.get("id") or "").strip()
    title_ru = _clean((biblio.get("ru") or {}).get("title"))
    title_en = _clean((biblio.get("en") or {}).get("title"))
    title = title_ru or title_en or _clean(snippet.get("title"))
    published = _day(common.get("publication_date"))
    priority = _day((common.get("priority") or {}).get("filing_date") if isinstance(common.get("priority"), dict)
                    else None) or _day((common.get("application") or {}).get("filing_date")) or published
    if not (doc_id and title and published) or published > as_of.isoformat():
        return None
    description = _clean(snippet.get("description"))
    if not relevant(f"{title_ru}. {title_en}", phrases, min_share=MIN_SHARE, window=NEAR):
        return None
    holders = (biblio.get("ru") or {}).get("patentee") or (biblio.get("en") or {}).get("patentee") or []
    assignee = ", ".join(_clean(item.get("name")) for item in holders if isinstance(item, dict) and item.get("name"))
    number = f"{common.get('publishing_office') or ''}{common.get('document_number') or ''}{common.get('kind') or ''}"
    text = ". ".join(part for part in (
        title, title_en if title_en and title_en != title else "", description,
        f"Патентообладатель: {assignee}" if assignee else "",
        f"Публикация: {published}; приоритет: {priority}") if part)
    return _document(DOC_URL + doc_id, title, text, priority, snippet.get("lang") or "ru", "rospatent",
                     patent_number=number or doc_id, assignee=assignee, publication_date=published,
                     priority_date=priority, source_country=common.get("publishing_office"))


def _family_key(ref: dict) -> str:
    return re.sub(r"[^a-zа-яё0-9]", "", ref["title"].lower())


def search(terms: list[str], as_of: date, limit: int) -> list[dict]:
    if not _enabled(terms):
        return []
    headers = {"Authorization": f"Bearer {env('ROSPATENT_API_KEY')}"}
    phrases = _phrases(terms)
    refs: list[dict] = []
    seen: set[str] = set()
    for datasets in (RU_DATASETS, WORLD_DATASETS):
        for phrase in phrases:
            if len(refs) >= limit:
                return refs[:limit]
            payload = _json_request("rospatent", URL, body=_body(phrase, as_of, datasets), headers=headers,
                                    timeout=30) or {}
            for hit in payload.get("hits") or []:
                ref = _ref(hit, [*terms, *phrases], as_of)
                if ref and ref["url"] not in seen and _family_key(ref) not in seen:
                    seen.update((ref["url"], _family_key(ref)))
                    refs.append(ref)
    return refs[:limit]


CONNECTOR = Connector("Rospatent", "science", "strict", search, enabled=_enabled, language="multi",
                      radar_limit=5, default_on=False, scopes=("branch",))
