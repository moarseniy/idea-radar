"""Источники фактов о зрелости технологии и общий вердикт.

Модуль используют и сборщик позитивов, и сборщик негативов — раньше у каждого была
своя копия логики, что уже приводило к расхождениям.

Источники и их роли:
  Semantic Scholar — объём публикаций всего и за последние 3 года. Основной измеритель.
  arXiv            — фразовый поиск по препринтам. Силён в ИИ, роботах, edge; слабее
                     в финтехе. Служит подтверждением и работает там, где S2 молчит.
  Wikipedia        — дата создания статьи как индикатор укоренённости термина.
  OpenAlex         — лучший из четырёх (точный фразовый фильтр и счётчики по годам),
                     но дневной бюджет общий на IP и выбивается за один рабочий день.
                     Используется, только если отвечает; блокировка не останавливает работу.

ГЛАВНОЕ ПРАВИЛО: недоступность источника никогда не превращается в «признаков зрелости
не обнаружено». Каждый источник явно сообщает, ответил он или нет, и вердикт
"unverified" отличается от "early". Первая версия этой логики глотала ошибки, и все
19 проверок прошли вхолостую, выглядя при этом успешными.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "ml" / ".cache" / "maturity"

CONTACT = os.getenv("OPENALEX_MAILTO", "").strip() or "hackathon@example.org"
UA = {"User-Agent": f"weak-signal-radar/0.1 (hackathon research; {CONTACT})"}

class HostLimiter:
    """Минимальный интервал между запросами к одному сервису, безопасно для потоков.

    Разные сервисы опрашиваются параллельно, а к одному и тому же — не чаще его лимита.
    Раньше все запросы шли последовательно с паузами, и 8 секунд на термин уходило
    просто на ожидание.
    """

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        time.sleep(max(0.0, slot - now))


class Breaker:
    """После нескольких отказов подряд источник отключается до конца прогона.

    Semantic Scholar без ключа стабильно отвечает 429, и его ретраи съедали больше
    половины времени проверки — 23 секунды на термин без единого результата.
    """

    def __init__(self, name: str, threshold: int = 3) -> None:
        self.name = name
        self.threshold = threshold
        self._failures = 0
        self._lock = threading.Lock()
        self.open = False

    def record(self, ok: bool) -> None:
        with self._lock:
            self._failures = 0 if ok else self._failures + 1
            if not self.open and self._failures >= self.threshold:
                self.open = True
                print(f"      [{self.name}] {self.threshold} отказа подряд — источник отключён до конца прогона")


# Интервалы по заявленным ограничениям сервисов. arXiv — самый строгий.
LIMITS = {"openalex": HostLimiter(0.25), "s2": HostLimiter(1.1),
          "arxiv": HostLimiter(3.1), "wikipedia": HostLimiter(0.2)}
BREAKERS = {name: Breaker(name) for name in ("openalex", "s2", "arxiv", "wikipedia")}
CACHE_VERSION = 2

MATURE_WIKI_AGE_YEARS = 7
MATURE_VOLUME_3Y = 5000
EARLY_VOLUME_3Y = 1200
RECENT_WINDOW = 3


@dataclass
class Maturity:
    """Факты о технологии и вердикт. Все поля _ok говорят, ответил ли источник."""
    name_en: str
    s2_total: int | None = None
    s2_recent: int | None = None
    s2_ok: bool = False
    arxiv_total: int | None = None
    arxiv_ok: bool = False
    wiki_created: str | None = None
    wiki_ok: bool = False
    openalex_recent: int | None = None
    openalex_ok: bool = False
    histogram: dict = field(default_factory=dict)
    cache_version: int = 0
    verdict: str = "unverified"          # mature | early | unverified
    note: str = ""
    sources_answered: list[str] = field(default_factory=list)


def _get(url: str, *, tries: int = 3, timeout: int = 40, raw: bool = False):
    last: Exception | None = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                return body if raw else json.loads(body)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            time.sleep(3 * (attempt + 1) + random.random())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3 * (attempt + 1) + random.random())
    raise RuntimeError(str(last)[:120])


def _semantic_scholar(phrase: str) -> tuple[int | None, int | None]:
    """Публикаций всего и за последние 3 года."""
    base = "https://api.semanticscholar.org/graph/v1/paper/search?"
    total = _get(base + urllib.parse.urlencode({"query": phrase, "limit": 1, "fields": "title"}), tries=1)
    LIMITS["s2"].wait()
    current = date.today().year
    recent = _get(base + urllib.parse.urlencode({
        "query": phrase, "limit": 1, "fields": "title",
        "year": f"{current - RECENT_WINDOW + 1}-{current}",
    }), tries=1)
    return total.get("total"), recent.get("total")


def _arxiv(phrase: str) -> int | None:
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": f'all:"{phrase}"', "start": 0, "max_results": 1,
    })
    body = _get(url, raw=True).decode(errors="replace")
    match = re.search(r"opensearch:totalResults[^>]*>(\d+)<", body)
    return int(match.group(1)) if match else None


def _wikipedia(title: str) -> str | None:
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "titles": title, "prop": "revisions", "rvlimit": 1,
        "rvdir": "newer", "rvprop": "timestamp", "format": "json", "redirects": 1,
    })
    page = next(iter(_get(url)["query"]["pages"].values()))
    if "missing" in page or not page.get("revisions"):
        return None
    return page["revisions"][0]["timestamp"][:10]


def _openalex(phrase: str) -> tuple[int, dict[int, int]]:
    """Публикаций за последние годы и полный ряд по годам."""
    params = {"filter": f'title_and_abstract.search:"{phrase}"',
              "group_by": "publication_year", "mailto": CONTACT}
    payload = _get("https://api.openalex.org/works?" + urllib.parse.urlencode(params), tries=2)
    current = date.today().year
    histogram = {int(g["key"]): g["count"] for g in payload.get("group_by", [])
                 if g["key"].isdigit() and 1990 <= int(g["key"]) <= current}
    recent = sum(v for y, v in histogram.items() if y > current - RECENT_WINDOW)
    return recent, histogram


def _decide(facts: Maturity) -> tuple[str, str]:
    """Вердикт по собранным фактам. Зрелость подтверждается, ранность — лишь допускается."""
    if facts.wiki_ok and facts.wiki_created:
        age = (datetime.now() - datetime.strptime(facts.wiki_created, "%Y-%m-%d")).days / 365.25
        if age >= MATURE_WIKI_AGE_YEARS:
            return "mature", f"статья в Википедии от {facts.wiki_created} ({age:.0f} лет) — термин укоренён"

    # Долголетие термина как признак зрелости ОТВЕРГНУТО проверкой на золоте:
    # правило «термин в литературе 15+ лет» ложно отсекло 15 из 100 записей методологов
    # (нейроморфные процессоры, роевая робототехника, оптические нейросети — старые
    # научные направления с новым коммерческим движением). Слабый сигнал у заказчика —
    # это стадия внедрения, а не возраст идеи; научные базы измеряют второе.

    volume = facts.openalex_recent if facts.openalex_ok else (facts.s2_recent if facts.s2_ok else None)
    if volume is not None and volume >= MATURE_VOLUME_3Y:
        return "mature", f"{volume} публикаций за {RECENT_WINDOW} года — тема массовая"

    if not facts.sources_answered:
        return "unverified", "ни один источник не ответил"

    # Ноль публикаций во всех источниках не означает раннюю стадию. Это означает,
    # что технологию не удалось найти: либо название не является поисковым термином
    # (длинная описательная фраза вместо канонического названия), либо её не
    # существует. Ранняя стадия — это МАЛО публикаций, а не НИ ОДНОЙ.
    volumes = [v for v in (facts.openalex_recent if facts.openalex_ok else None,
                           facts.s2_recent if facts.s2_ok else None,
                           facts.arxiv_total if facts.arxiv_ok else None) if v is not None]
    if volumes and max(volumes) == 0:
        return "unverified", ("ни одной публикации ни в одном источнике — "
                              "название не найдено как поисковый термин")

    parts = []
    if volume is not None:
        parts.append(f"{volume} публикаций за {RECENT_WINDOW} года")
        if volume < EARLY_VOLUME_3Y:
            parts.append("объём низкий")
    if facts.arxiv_ok and facts.arxiv_total is not None:
        parts.append(f"arXiv: {facts.arxiv_total}")
    if facts.wiki_ok:
        parts.append("статьи в Википедии нет" if not facts.wiki_created
                     else f"статья от {facts.wiki_created}")
    return "early", "; ".join(parts) or "признаков зрелости не найдено"


def _call(source: str, fn, *args):
    """Один источник: пропуск при открытом предохранителе, ограничитель, учёт отказа."""
    breaker = BREAKERS[source]
    if breaker.open:
        return None, False
    LIMITS[source].wait()
    try:
        result = fn(*args)
        breaker.record(True)
        return result, True
    except Exception:  # noqa: BLE001 — отказ источника фиксируется, а не проглатывается
        breaker.record(False)
        return None, False


def assess(name_en: str, *, use_openalex: bool = True, verbose: bool = False) -> Maturity:
    """Собрать факты из четырёх источников параллельно и вынести вердикт. Кэш на диске."""
    facts = Maturity(name_en=name_en, cache_version=CACHE_VERSION)
    if not name_en or not name_en.strip():
        facts.note = "нет английского названия для поиска"
        return facts

    slug = re.sub(r"[^a-z0-9]+", "_", name_en.lower())[:70]
    cached = CACHE / f"{slug}.json"
    if cached.exists():
        blob = json.loads(cached.read_text(encoding="utf-8"))
        if blob.get("cache_version") == CACHE_VERSION:
            blob["histogram"] = {int(k): v for k, v in blob.get("histogram", {}).items()}
            cached_facts = Maturity(**blob)
            # Кэшируются факты, а не выводы: при изменении правил вердикт пересчитывается
            # без повторных запросов к источникам.
            cached_facts.verdict, cached_facts.note = _decide(cached_facts)
            return cached_facts

    jobs = {
        "semantic_scholar": ("s2", _semantic_scholar),
        "wikipedia": ("wikipedia", _wikipedia),
    }
    if use_openalex:
        jobs["openalex"] = ("openalex", _openalex)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(_call, source, fn, name_en) for name, (source, fn) in jobs.items()}
        results = {name: future.result() for name, future in futures.items()}
    # arXiv — запасной источник объёма: его лимит (1 запрос в 3 с) был потолком скорости
    # всей проверки, а при ответившем OpenAlex он ничего не добавляет к вердикту.
    if not results.get("openalex", (None, False))[1]:
        results["arxiv"] = _call("arxiv", _arxiv, name_en)
    else:
        results["arxiv"] = (None, False)

    value, ok = results.get("openalex", (None, False))
    if ok:
        facts.openalex_recent, facts.histogram = value
        facts.openalex_ok = True
        facts.sources_answered.append("openalex")
    value, ok = results["semantic_scholar"]
    if ok and value and value[1] is not None:
        facts.s2_total, facts.s2_recent = value
        facts.s2_ok = True
        facts.sources_answered.append("semantic_scholar")
    value, ok = results["arxiv"]
    if ok and value is not None:
        facts.arxiv_total, facts.arxiv_ok = value, True
        facts.sources_answered.append("arxiv")
    value, ok = results["wikipedia"]
    if ok:
        facts.wiki_created, facts.wiki_ok = value, True
        facts.sources_answered.append("wikipedia")

    facts.verdict, facts.note = _decide(facts)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(asdict(facts), ensure_ascii=False), encoding="utf-8")
    return facts


# Термины с заранее известным ответом. Каждый ловит один из дефектов, которые уже
# случались: проглоченные ошибки источников (всё становится early), ноль публикаций
# как признак ранней стадии, сломанный детектор зрелости.
CONTROL_SET = (
    ("Kubernetes", "mature"),
    ("Blockchain", "mature"),
    ("HASEL actuator", "early"),
    ("Physics-informed neural process twins for quasi-orbital fabrication", "unverified"),
)


def preflight() -> bool:
    """Прогнать контрольный набор. False — сбор запускать нельзя."""
    print("контрольная проверка источников:")
    ok = True
    for term, expected in CONTROL_SET:
        result = assess(term)
        passed = result.verdict == expected
        ok &= passed
        mark = "ок" if passed else "НЕ СХОДИТСЯ"
        print(f"  {term[:48]:50} ожидали {expected:10} получили {result.verdict:10} {mark}")
    print("контроль пройден\n" if ok else "КОНТРОЛЬ НЕ ПРОЙДЕН — сбор остановлен\n")
    return ok


if __name__ == "__main__":
    import sys
    for term in sys.argv[1:] or ["HASEL actuator", "federated learning", "Kubernetes"]:
        result = assess(term, verbose=True)
        print(f"\n{term}")
        print(f"  вердикт: {result.verdict}")
        print(f"  факты  : {result.note}")
        print(f"  ответили: {', '.join(result.sources_answered) or 'никто'}")
