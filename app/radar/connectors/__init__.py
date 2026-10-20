"""Реестр прямых коннекторов: наука, деньги, рынок, внедрение.

Каждый коннектор — модуль этого пакета с объектом ``CONNECTOR`` (или кортежем
``CONNECTORS``). Реестр собирается автоматически, поэтому новый источник не требует
правок в общих файлах: достаточно положить модуль рядом.

Контракт коннектора:
  search(terms, as_of, limit) -> list[dict] — ссылки формата research_apis._document;
  channel      — канал признаков и квот: science | money | market | adoption;
  date_support — как источник соблюдает дату оценки:
                 strict — API фильтрует по дате на своей стороне;
                 post   — у каждой записи есть дата, отсекаем после as_of сами;
                 none   — даты нет или она недостоверна: в историческом режиме источник
                          не вызывается (машина времени, этап 3).
  enabled(terms) — дешёвая проверка до сетевого вызова (ключ задан, тема медицинская…);
  default_on   — новые источники выключены, пока квоты (этап 2) не решают, сколько им
                 давать: включаются через RADAR_EXTRA_CONNECTORS=all или списком имён.
                 RADAR_DISABLED_CONNECTORS выключает любые;
  scopes       — где вызывается: "branch" — поиск по веткам запроса (7 вызовов на прогон),
                 "candidate" — проверка каждого кандидата при подсчёте признаков (до 64).
                 Медленные источники (UKRI, SEDIA) только в "branch".

Денежные записи несут ``structured_event`` (см. ``events.make_event``), иначе признаки
финансирования их не увидят.
"""
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

CHANNELS = ("science", "money", "market", "adoption")
DATE_SUPPORT = ("strict", "post", "none")


def _always(_terms: list[str]) -> bool:
    return True


@dataclass(frozen=True)
class Connector:
    name: str
    channel: str
    date_support: str
    search: Callable[[list[str], date, int], list[dict]]
    enabled: Callable[[list[str]], bool] = _always
    language: str = "en"   # язык аудита задания в пайплайне: en | multi
    radar_limit: int = 4   # записей на ветку в живом поиске
    default_on: bool = True
    scopes: tuple[str, ...] = ("branch", "candidate")

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError(f"{self.name}: неизвестный канал {self.channel!r}")
        if self.date_support not in DATE_SUPPORT:
            raise ValueError(f"{self.name}: неизвестная поддержка даты {self.date_support!r}")

    def run(self, terms: list[str], as_of: date, limit: int) -> list[dict]:
        """Поиск с разметкой канала; записи, датированные позже as_of, отбрасываются."""
        cutoff = as_of.isoformat()
        refs = [ref for ref in self.search(terms, as_of, limit)
                if self.date_support == "none" or not ref.get("observed_at")
                or str(ref["observed_at"])[:10] <= cutoff]
        return tag(refs, self)


def tag(refs: list[dict], connector: Connector) -> list[dict]:
    for ref in refs:
        ref.setdefault("connector", connector.name)
        ref.setdefault("channel", connector.channel)
        ref.setdefault("date_support", connector.date_support)
    return refs


# Порядок модулей задаёт порядок вызова и, значит, очереди чтения в пайплайне.
_ORDER = ("builtin",)
_REGISTRY: dict[str, Connector] | None = None


def _discover() -> dict[str, Connector]:
    names = [m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_")
             and m.name not in {"events"}]
    names.sort(key=lambda n: (_ORDER.index(n) if n in _ORDER else len(_ORDER), n))
    found: dict[str, Connector] = {}
    for module_name in names:
        module = importlib.import_module(f"{__name__}.{module_name}")
        items = getattr(module, "CONNECTORS", None) or (
            (module.CONNECTOR,) if hasattr(module, "CONNECTOR") else ())
        for connector in items:
            if connector.name in found:
                raise ValueError(f"коннектор {connector.name!r} объявлен дважды")
            found[connector.name] = connector
    return found


def registry() -> dict[str, Connector]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _discover()
    return _REGISTRY


def connectors(terms: list[str] | None = None, *, channel: str | None = None,
               historical: bool = False, exclude: set[str] | frozenset[str] = frozenset(),
               scope: str | None = None, only: frozenset[str] | None = None) -> list[Connector]:
    """Коннекторы, применимые к теме; в историческом режиме — только с фильтром по дате."""
    extra = _names("RADAR_EXTRA_CONNECTORS")
    disabled = _names("RADAR_DISABLED_CONNECTORS")
    out = []
    for connector in registry().values():
        if connector.name in exclude or connector.name in disabled or (channel and connector.channel != channel):
            continue
        if only is not None and connector.name not in only:
            continue
        # Явный список (источники признаков обученной модели) включает и выключенные по
        # умолчанию: иначе признаки на инференсе считались бы не так, как на обучении.
        if only is None and not connector.default_on and "all" not in extra and connector.name not in extra:
            continue
        if scope and scope not in connector.scopes:
            continue
        if historical and connector.date_support == "none":
            continue
        if terms is not None and not connector.enabled(terms):
            continue
        out.append(connector)
    return out


def is_historical(as_of: date) -> bool:
    """Оценка на прошлую дату (машина времени): источники без достоверной даты выключаются.

    GitHub и Hugging Face отдают текущие звёзды и даты изменения, Bing News и Хабр — выдачу
    без фильтра по дате: в прошлом это сведения из будущего.
    """
    return as_of < date.today()  # noqa: DTZ011 — дата оценки в локальном календарном дне


def _names(variable: str) -> set[str]:
    from app.ml.env import env

    return {name.strip() for name in env(variable).split(",") if name.strip()}


def model_connectors() -> frozenset[str]:
    """Источники признаков модели без явного списка в артефакте: те, что были до реестра.

    Модель видела на обучении только их; новые денежные источники меняют распределение
    признаков, поэтому попадают в признаки лишь у модели, обученной вместе с ними
    (поле ``feature_connectors`` артефакта), а до тех пор — только в карточку.
    """
    return frozenset(name for name, connector in registry().items() if connector.default_on)


# Каналы коннекторов open_sources, которые ищут по одной локализованной фразе
# (не по списку терминов) и поэтому живут вне реестра. Ключ — префикс имени задания.
QUERY_SOURCES: dict[str, tuple[str, str]] = {
    "Bing News": ("market", "none"),
    "Google News (": ("market", "post"),  # локализованный, машина времени
    "Crossref": ("science", "strict"),
    "Patents": ("science", "strict"),
    "GDELT Global News": ("market", "none"),
    "arXiv": ("science", "strict"),
    "Хабр": ("science", "none"),
}


# Канал ссылки без коннектора (веб-поиск LLM, SearXNG) — по домену; прочее — рынок.
MONEY_HOSTS = ("techcrunch.com", "crunchbase.com", "sifted.eu", "eu-startups.com", "venturebeat.com",
               "pitchbook.com", "dealroom.co", "cbinsights.com", "ycombinator.com", "vc.ru", "rb.ru",
               "usaspending.gov", "grants.gov", "sam.gov", "sec.gov", "ted.europa.eu", "cordis.europa.eu",
               "reporter.nih.gov", "gtr.ukri.org", "nsf.gov", "fasie.ru", "rscf.ru", "zakupki.gov.ru")
SCIENCE_HOSTS = ("arxiv.org", "doi.org", "nature.com", "science.org", "sciencedirect.com", "springer.com",
                 "ieee.org", "acm.org", "wiley.com", "mdpi.com", "biorxiv.org", "medrxiv.org",
                 "openreview.net", "semanticscholar.org", "openalex.org", "pubmed.ncbi.nlm.nih.gov",
                 "patents.google.com", "freepatentsonline.com", "searchplatform.rospatent.gov.ru", "fips.ru", "elibrary.ru", "cyberleninka.ru", "habr.com")


def channel_for_url(url: str) -> str:
    from urllib.parse import urlsplit

    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")

    def on(hosts: tuple[str, ...]) -> bool:
        return any(host == known or host.endswith("." + known) for known in hosts)
    return "money" if on(MONEY_HOSTS) else "science" if on(SCIENCE_HOSTS) else "market"


def describe(task_name: str) -> tuple[str | None, str | None]:
    """Канал и поддержка даты для имени задания пайплайна; (None, None), если неизвестно."""
    connector = registry().get(task_name)
    if connector:
        return connector.channel, connector.date_support
    for prefix, meta in QUERY_SOURCES.items():
        if task_name.startswith(prefix):
            return meta
    return None, None
