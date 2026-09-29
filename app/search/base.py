"""Общие типы сервиса поиска: запрос, результат, ошибки."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta


@dataclass(frozen=True)
class SearchQuery:
    text: str
    as_of: date | None = None      # дата оценки: результаты позже неё отбрасываются
    language: str = "en"           # ru | en | …: язык выдачи и регион, где провайдер это умеет
    limit: int = 10
    lookback_days: int = 3 * 365   # окно свежести до as_of для провайдеров с фильтром дат

    @property
    def start(self) -> date | None:
        return self.as_of - timedelta(days=self.lookback_days) if self.as_of else None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["as_of"] = self.as_of.isoformat() if self.as_of else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> SearchQuery:
        as_of = data.get("as_of")
        return cls(text=str(data["text"]), as_of=date.fromisoformat(as_of) if as_of else None,
                   language=str(data.get("language") or "en"), limit=int(data.get("limit") or 10),
                   lookback_days=int(data.get("lookback_days") or 3 * 365))


class SearchError(Exception):
    """Провайдер не ответил или ответил ошибкой — цепочка переходит к следующему."""


class NotConfigured(SearchError):
    """У провайдера нет ключа или адреса."""


def result(url: str, title: str, *, provider: str, snippet: str = "", published_at: str | None = None,
           language: str | None = None, content: str | None = None, engine: str | None = None) -> dict:
    """Единый формат результата: одинаковый для всех провайдеров и для HTTP API сервиса."""
    return {"url": url.strip(), "title": clean(title) or url.strip(), "snippet": clean(snippet),
            "published_at": iso_date(published_at), "language": language, "provider": provider,
            "engine": engine, "content": content or None}


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def iso_date(value) -> str | None:
    """Дата публикации в ISO (YYYY-MM-DD) из разных форматов провайдеров; иначе None."""
    if not value:
        return None
    text = str(value).strip()
    for pattern, fmt in ((r"^\d{4}-\d{2}-\d{2}", "%Y-%m-%d"), (r"^\d{8}T", "%Y%m%d")):
        match = re.match(pattern, text)
        if match:
            try:
                return datetime.strptime(match.group(0).rstrip("T"), fmt).date().isoformat()
            except ValueError:
                return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z", "%B %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def language_of(text: str) -> str:
    """ru, если в запросе кириллица, иначе en — для провайдеров, которым нужен язык выдачи."""
    return "ru" if re.search(r"[А-Яа-яЁё]", text) else "en"
