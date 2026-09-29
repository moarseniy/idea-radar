"""Сбор обучающей выборки: Tavily и общие структурированные API ищут; факты решают.

Одна программа для обоих классов. Данные, где позитивы и негативы собраны разными
скриптами, разделяются по отпечаткам пайплайна, а не по сути: у Арсения стадийные
признаки заполнены у 100% позитивов и 35% негативов, и модель учит «заполнено → позитив».
Здесь обе метки проходят через один и тот же поиск, извлечение и проверку.

Три ступени:
  1. Поиск.      Tavily возвращает статьи; API-адаптеры дополняют их публикациями, грантами,
                 репозиториями, моделями, тендерами и (для медицины) испытаниями/регуляторными записями.
  2. Извлечение. Модель получает пронумерованные статьи и выделяет технологии, ссылаясь
                 на НОМЕРА статей. Ссылка не может быть выдуманной: URL берётся из
                 выдачи Tavily по номеру, модель его не пишет.
  3. Проверка.   evidence_sources.assess — объём публикаций, ряд по годам, Википедия.
                 Позитив принимается, только если факты не показывают зрелости; негатив —
                 только если факты подтверждают заявленный тип.

Две независимые линии подтверждения у каждой записи: отраслевые медиа (ступень 1) и
научные базы (ступень 3).

Запуск:
    python ml/scripts/collect.py --label positive
    python ml/scripts/collect.py --label negative
    python ml/scripts/collect.py --label positive --domain Финтех
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.radar.locales import LANGUAGE_NAMES, training_language_batch


def load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, _, value = line.partition("=")
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


load_dotenv()

from app.radar.research_apis import search_structured_sources
from evidence_sources import HostLimiter, assess, preflight

GOLD = ROOT / "ml" / "datasets" / "gold.csv"
LABELED = ROOT / "ml" / "datasets" / "labeled.csv"
PENDING = ROOT / "ml" / "datasets" / "collect_pending.csv"
REJECTED = ROOT / "ml" / "datasets" / "collect_rejected.csv"
EXCLUSIONS = ROOT / "ml" / "datasets" / "exclusions"

MODEL = os.getenv("EXTRACTION_MODEL", "google/gemini-3.8-flash")
TAVILY_LIMIT = HostLimiter(0.3)

LABELED_FIELDS = ["id", "name", "domain", "label", "stage", "trend", "negative_type",
                  "rationale", "annotator", "evidence_urls", "added_at"]

DOMAINS = {
    "Индустриальный ИИ": "industrial AI manufacturing",
    "Роботы": "robotics",
    "Инфраструктура ИИ": "AI infrastructure chips datacenter",
    "Финтех": "fintech payments banking",
    "Защита ИИ": "AI security",
    "Edge": "edge AI on-device",
}

# Ракурс поиска: запросы к Tavily, тема выдачи и что просим извлечь.
POSITIVE_ANGLES = {
    "stealth": (["{d} startup emerges from stealth", "{d} stealth startup launches"], "news",
                "технологии, которые развивают компании, недавно вышедшие из stealth"),
    "seed": (["{d} startup raises seed round", "{d} Series A funding new technology"], "news",
             "технологии, под которые недавно привлечён seed или Series A"),
    "pilot": (["{d} first pilot deployment new technology", "{d} first commercial deployment startup"], "news",
              "технологии, дошедшие до первых пилотов или первых коммерческих поставок"),
    "spinout": (["{d} university spinout startup", "{d} research lab spinout funding"], "news",
                "технологии университетских и лабораторных спиноутов"),
    "research": (["{d} new approach prototype breakthrough", "{d} novel technique demonstrated prototype"], "news",
                 "новые технические подходы на стадии прототипа"),
}

# Для негативов тип заявляется при извлечении и затем проверяется фактами.
NEGATIVE_ANGLES = {
    "N1": (["{d} widely adopted industry standard technology", "{d} mature technology mass adoption"], "general",
           "ЗРЕЛЫЕ технологии: массово внедрены, есть отраслевые стандарты"),
    "N3d": (["{d} dominant technology market leaders market share", "{d} mainstream technology fast growing market"], "general",
            "технологии, которые быстро растут, но уже массовые, с выраженными лидерами рынка"),
    "N3a": (["{d} established niche technology specialized vendors decades", "{d} legacy specialized technology"], "general",
            "нишевые технологии, существующие давно, с устоявшимися поставщиками"),
    "N3e": (["{d} long-standing research problem decades unsolved", "{d} research area for decades little commercial"], "general",
            "направления, которые исследуются десятилетиями без выхода в применение"),
    "N2": (["{d} overhyped buzzword hype", "{d} hype cycle disappointment technology"], "news",
           "маркетинговые термины и хайп без технического содержания"),
}

POSITIVE_EXTRA = """- Не включай продукты крупных устоявшихся компаний (Cisco, Microsoft, Google, IBM, Amazon,
  Oracle, SAP, Palo Alto Networks и подобных): их запуск — признак зрелости рынка, а не
  ранний сигнал. Нужны технологии молодых компаний, лабораторий и первых пилотов.
"""

# ТЗ: соцсети, личные блоги и агрегаторы — только первичный индикатор, не основание.
LOW_TRUST_HOSTS = ("instagram.", "facebook.", "linkedin.", "twitter.", "x.com", "reddit.",
                   "youtube.", "tiktok.", "medium.com", "substack.com", "t.me")


def trusted(article: dict) -> bool:
    host = urllib.parse.urlparse(article.get("url", "")).netloc.lower()
    return bool(host) and not any(marker in host for marker in LOW_TRUST_HOSTS)


EXTRACT_PROMPT = """Ниже пронумерованные статьи из открытых источников. Выдели из них {what}.

Область: {domain}

Требования:
- Технология, а не компания, продукт или отрасль. Плохо: «Straiker», «кибербезопасность».
  Хорошо: «Поведенческий межсетевой экран для вызовов инструментов ИИ-агентами».
- КОНКРЕТНЫЙ ПОДХОД, а не зонтичная категория: название должно говорить, что именно
  делается и для чего, и быть уже, чем название области. Плохо: «Защита агентских систем
  ИИ», «Безопасность ИИ». Хорошо: «Выдача ИИ-агентам краткоживущих прав доступа с
  ограниченной областью действия». Детали реализации не обязательны.
- Одна технология — одна запись. Не выдавай одно и то же разными словами.
{extra}- Используй ТОЛЬКО сведения из статей. Для каждой технологии укажи номера статей в sources.
- name_ru — по-русски, 5-12 слов, «что именно делается для какой задачи».
- search_term — КАНОНИЧЕСКИЙ англоязычный термин из 2-4 слов для точного поиска в научных
  базах. Не описание, а устоявшееся название: «steerable microcatheter», а не
  «Magnetically Steered Endovascular Microcatheters for Neurovascular Intervention».
- companies — компании из статей, через запятую.
- note — одна фраза с фактом из статей: раунд, выход из stealth, пилот, стандарт, доля рынка.
- Не более {count} технологий. Если подходящих нет — пустой список.

Статьи:
{articles}

Ответ строго JSON без markdown:
{{"items":[{{"name_ru":"...","search_term":"...","companies":"...","note":"...","sources":[1,3]}}]}}"""


def _post_json(url: str, body: dict, headers: dict, tries: int = 3, timeout: int = 90) -> dict:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json", **headers})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(3 * (attempt + 1) + random.random())
    raise RuntimeError(f"{url}: {last}")


TAVILY_CACHE = ROOT / "ml" / ".cache" / "tavily"
LOCALIZED_QUERY_CACHE = ROOT / "ml" / ".cache" / "localized_search_queries"
CACHED_ONLY = False


def localize_search_query(query: str, domain: str, languages: tuple[str, ...]) -> list[dict]:
    """Create idiomatic native-language search variants; cache the LLM expansion."""
    cache_key = f"{domain}\n{query}\n{'|'.join(languages)}"
    path = LOCALIZED_QUERY_CACHE / f"{hashlib.sha256(cache_key.encode()).hexdigest()}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    fallback = [{"language": "en", "query": query}]
    if CACHED_ONLY or not os.environ.get("OPENROUTER_API_KEY"):
        return fallback
    language_list = ", ".join(f"{code} ({LANGUAGE_NAMES[code]})" for code in languages)
    prompt = f"""Create search-engine queries about emerging technical products and weak signals.
Domain context: {domain}
Search intent: {query}
Return one natural, concise search query in each requested language: {language_list}.
Use terms people in that language actually use; preserve company names, standards, and
technical acronyms. Do not translate literally and do not add invented claims.
Return strict JSON: {{"queries":[{{"language":"code","query":"native-language query"}}]}}"""
    try:
        response = _post_json("https://openrouter.ai/api/v1/chat/completions",
            {"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
             "messages": [{"role": "user", "content": prompt}]},
            {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or "{}"
        parsed = json.loads(content)
        by_language = {}
        for item in parsed.get("queries", []):
            language = str(item.get("language", "")).lower().strip()
            text = " ".join(str(item.get("query", "")).split())
            if language in languages and text:
                by_language.setdefault(language, text[:300])
        result = [{"language": language, "query": by_language[language]}
                  for language in languages if by_language.get(language)] or fallback
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result
    except Exception as exc:  # noqa: BLE001 — keep the original search usable
        print(f"  multilingual query expansion fallback ({type(exc).__name__})")
        return fallback


def tavily(query: str, topic: str) -> list[dict]:
    """Поиск Tavily с кэшем на диске: бюджет запросов ограничен, а полные тексты статей
    нужны и позже — для проверки стадии. Без кэша их приходилось доставать повторно."""
    digest = hashlib.sha256(f"{topic}\n{query}".encode()).hexdigest()
    cached = TAVILY_CACHE / f"{digest}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    # Reuse the old English-only cache names so existing Tavily spend is not repeated.
    legacy_key = re.sub(r"[^a-z0-9]+", "_", f"{topic}_{query}".lower())[:150]
    legacy = TAVILY_CACHE / f"{legacy_key}.json"
    if query.isascii() and legacy.exists():
        return json.loads(legacy.read_text(encoding="utf-8"))
    if CACHED_ONLY:
        return []  # повторный прогон по уже оплаченным запросам — новые не тратим
    TAVILY_LIMIT.wait()
    # Полный текст статьи: в 500-символьной выжимке новости о раунде механизм
    # технологии не описан, и модель честно не может его назвать.
    body = {"query": query, "topic": topic, "max_results": 8, "search_depth": "basic",
            "include_raw_content": True}
    if topic == "news":
        body["days"] = 540
    payload = _post_json("https://api.tavily.com/search", body,
                         {"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"}, timeout=40)
    results = payload.get("results", [])
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    return results


def extract(domain: str, what: str, articles: list[dict], count: int, extra: str = "") -> list[dict]:
    listing = "\n\n".join(
        f"[{i}] [язык поискового запроса: {a.get('search_language', 'unknown')}] "
        f"{a.get('title', '')} ({(a.get('published_date') or 'дата неизвестна')[:16]})\n"
        f"{a.get('url', '')}\n{(a.get('raw_content') or a.get('content') or '')[:1800]}"
        for i, a in enumerate(articles, start=1))
    prompt = EXTRACT_PROMPT.format(what=what, domain=domain, count=count, articles=listing, extra=extra)
    for attempt in range(2):
        response = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
             "messages": [{"role": "user", "content": prompt}]},
            {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        content = (response.get("choices") or [{}])[0].get("message", {}).get("content")
        if not content:
            continue  # пустой ответ модели — ровно то, что теряло целые ракурсы раньше
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            continue
        blob = match.group(0)
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            try:
                data = json.loads(re.sub(r",(\s*[}\]])", r"\1", blob))
            except json.JSONDecodeError:
                continue
        items = data.get("items", [])
        return [i for i in items if isinstance(i, dict) and i.get("name_ru")]
    raise RuntimeError("модель дважды вернула пустой или невалидный ответ")


SUBTOPIC_PROMPT = """Область: {domain}.
Перечисли {n} технических подтем этой области, в которых в 2025-2026 годах стартапы
привлекали seed и Series A и о которых пишут отраслевые медиа (TechCrunch, VentureBeat,
SiliconANGLE и подобные). Подтема — 2-4 английских слова, как их пишут в новостях,
например "agent identity management", "photonic interconnect".
Не академические ниши, о которых пишут только в статьях, и не зрелые рынки.
Ответ строго JSON: {{"subtopics": ["...", "..."]}}"""

SUBTOPIC_CACHE = ROOT / "ml" / ".cache" / "subtopics"


def new_subtopics(domain: str, n: int) -> list[str]:
    """Дополнительные подтемы, не повторяющие уже использованные. Старые запросы
    вернули бы те же статьи, и бюджет Tavily ушёл бы на дубли."""
    cached = SUBTOPIC_CACHE / f"{normalize(domain).replace(' ', '_')}.json"
    used = json.loads(cached.read_text(encoding="utf-8")) if cached.exists() else []
    prompt = SUBTOPIC_PROMPT.format(domain=f"{domain} ({DOMAINS[domain]})", n=n) + (
        "\nНе повторяй и не перефразируй уже использованные подтемы:\n" + "\n".join(f"- {s}" for s in used))
    response = _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {"model": MODEL, "temperature": 0.5, "response_format": {"type": "json_object"},
         "messages": [{"role": "user", "content": prompt}]},
        {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    content = response["choices"][0]["message"]["content"] or "{}"
    items = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group(0)).get("subtopics", [])
    used_norm = {normalize(s) for s in used}
    fresh = [s for s in items if isinstance(s, str) and s.strip() and normalize(s) not in used_norm][:n]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(used + fresh, ensure_ascii=False), encoding="utf-8")
    return fresh


def subtopics(domain: str, n: int) -> list[str]:
    """Поисковые затравки по области. Только направляют поиск: метку решают факты,
    а подтемы сохраняются на диск и попадают в отчёт, чтобы выбор был прозрачным."""
    cached = SUBTOPIC_CACHE / f"{normalize(domain).replace(' ', '_')}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))[:n]
    response = _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {"model": MODEL, "temperature": 0.3, "response_format": {"type": "json_object"},
         "messages": [{"role": "user", "content": SUBTOPIC_PROMPT.format(domain=f"{domain} ({DOMAINS[domain]})", n=n)}]},
        {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    content = response["choices"][0]["message"]["content"] or "{}"
    items = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group(0)).get("subtopics", [])
    items = [s for s in items if isinstance(s, str) and s.strip()][:n]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items


def normalize(name: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", " ", (name or "").lower()).strip()


class Seen:
    """Дедупликация по пересечению значимых слов, безопасно для потоков."""

    GENERIC_TERM_WORDS = {"ai", "artificial", "intelligence", "robot", "robots", "system", "systems",
                          "data", "model", "models"}

    def __init__(self, names: list[str], excluded_terms: list[str] | None = None) -> None:
        self._sets = [self._words(n) for n in names]
        self._terms: set[str] = set()
        # Термины золота и датасета Арсения. Сверка только по русским названиям пропустила
        # 28 технологий из золота: одна и та же технология по-русски звучит по-разному,
        # а английский термин у неё один.
        self._excluded = [w for w in (self._term_words(x) for x in (excluded_terms or [])) if len(w) >= 2]
        self._lock = threading.Lock()

    @classmethod
    def _term_words(cls, term: str) -> set[str]:
        return set(re.sub(r"[^a-z0-9 ]+", " ", (term or "").lower()).split()) - cls.GENERIC_TERM_WORDS

    def excluded_term(self, term: str) -> bool:
        words = self._term_words(term)
        if len(words) < 2:
            return False
        return any(words <= ex or ex <= words or len(words & ex) / len(words | ex) >= 0.6
                   for ex in self._excluded)

    @staticmethod
    def _words(name: str) -> set[str]:
        return {w for w in normalize(name).split() if len(w) > 3}

    def claim_term(self, term: str) -> bool:
        """Один поисковый термин — одна технология, как бы её ни назвали по-русски."""
        with self._lock:
            if term in self._terms:
                return False
            self._terms.add(term)
            return True

    def claim(self, name: str) -> bool:
        """True — имя новое и теперь занято; False — дубль."""
        words = self._words(name)
        if not words:
            return False
        with self._lock:
            for other in self._sets:
                if other and len(words & other) / len(words | other) > 0.6:
                    return False
            self._sets.append(words)
            return True


def judge(label: str, claimed_type: str, facts, stage: int | None = None,
          stage_quote: str = "") -> tuple[bool, str]:
    """Принять запись? Решают факты, а не заявление модели."""
    if facts.verdict == "unverified" and not (label == "negative" and stage is not None):
        return False, f"не проверено: {facts.note}"
    if label == "positive":
        if facts.verdict == "mature":
            return False, f"вето: {facts.note}"
        return True, facts.note

    # Негатив: заявленный тип должен подтверждаться фактами.
    hist = facts.histogram or {}
    years = sorted(hist)
    if claimed_type in ("N1", "N3d", "N3a"):
        # Зрелость по научным базам не видна (SCARA и хирургических роботов они зрелыми
        # не признали). Стадию внедрения читает модель по статьям — на золоте 0 из 49
        # ложных «зрелых», на эталонно зрелых 13 из 14 узнанных.
        if facts.verdict == "mature":
            return True, facts.note
        if stage is not None and stage == 5:
            return True, f"стадия 5 по статьям: {stage_quote[:150]}"
        if claimed_type == "N3a" and stage == 4 and years and years[0] <= date.today().year - 12:
            return True, f"стадия 4 по статьям, публикации с {years[0]} года: {stage_quote[:120]}"
        return False, f"зрелость не подтверждена (стадия по статьям: {stage}): {facts.note}"
    if claimed_type == "N3e":
        recent = [hist.get(y, 0) for y in range(date.today().year - 8, date.today().year)]
        if len(years) >= 8 and sum(recent) >= 40:
            half = len(recent) // 2
            early, late = sum(recent[:half]) or 1, sum(recent[half:])
            if late / early <= 1.3:
                return True, f"ряд за 8 лет почти плоский ({early} → {late}) — {facts.note}"
        return False, f"застой не подтверждён: {facts.note}"
    if claimed_type == "N2":
        # Хайп: мало науки при заявленном медийном шуме. Проверка слабее прочих —
        # отмечается в обосновании.
        volume = facts.openalex_recent if facts.openalex_ok else None
        if volume is not None and volume <= 30:
            return True, f"[слабая проверка] научных работ за 3 года: {volume} — {facts.note}"
        return False, f"хайп не подтверждён: {facts.note}"
    return False, f"неизвестный тип {claimed_type}"


class Writer:
    """Запись после каждого ракурса. Прежний сборщик писал только в конце, и любой
    сбой или перезапуск стоил бы всего прогона."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def append(self, path: Path, rows: list[dict], fields: list[str]) -> None:
        if not rows:
            return
        with self._lock:
            exists = path.exists() and path.stat().st_size > 0
            with path.open("a" if exists else "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                writer.writerows(rows)


def run_angle(label: str, domain: str, key: str, spec, count: int,
              seen: Seen, writer: Writer, stats: dict, stats_lock: threading.Lock) -> str:
    queries, topic, what = spec
    # Locale mix depends on the domain, not the positive/negative collection angle,
    # so query language cannot become a shortcut for the training label.
    languages = training_language_batch(domain)
    buckets = {language: [] for language in languages}
    api_terms = [template.format(d=DOMAINS[domain]) for template in queries]
    urls = set()
    for template in queries:
        base_query = template.format(d=DOMAINS[domain])
        for localized in localize_search_query(base_query, domain, languages):
            language = localized["language"]
            for article in tavily(localized["query"], topic):
                if article.get("url") and article["url"] not in urls and trusted(article):
                    urls.add(article["url"])
                    buckets.setdefault(language, []).append({**article, "search_language": language})
    # Round-robin by query language prevents English results from filling the
    # entire 12-document context passed to the technology extractor.
    articles = []
    for index in range(8):
        for language in languages:
            if index < len(buckets.get(language, [])):
                articles.append(buckets[language][index])
                if len(articles) >= 12:
                    break
        if len(articles) >= 12:
            break
    structured_articles = []
    if not CACHED_ONLY and os.getenv("FEATURE_STRUCTURED_SOURCES", "1") != "0":
        try:
            for reference in search_structured_sources(api_terms, date.today(), per_source=2):
                doc = reference.get("document") or {}
                text = str(doc.get("text") or "").strip()
                if not text or not reference.get("url"):
                    continue
                structured_articles.append({
                    "url": reference["url"], "title": reference.get("title") or doc.get("title") or "",
                    "published_date": doc.get("date") or reference.get("observed_at"),
                    "raw_content": text, "content": text,
                    "search_language": reference.get("source_language") or "en",
                })
        except Exception as exc:  # noqa: BLE001 — a disabled public registry must not halt data mining
            print(f"  structured source supplement unavailable ({type(exc).__name__})")
    if structured_articles:
        articles = articles[:8] + structured_articles[:4]
    if not articles:
        return f"[{domain}/{key}] Tavily ничего не нашёл"
    source_tag = "tavily+structured" if structured_articles else "tavily"
    return process_articles(label, domain, key, articles[:12], what, count, source_tag,
                            seen, writer, stats, stats_lock)


def process_articles(label: str, domain: str, key: str, articles: list[dict], what: str, count: int,
                     source_tag: str, seen: Seen, writer: Writer, stats: dict,
                     stats_lock: threading.Lock) -> str:
    """Общая обработка статей любого источника: извлечение технологий, исключения,
    проверка фактами. Источник (Tavily, RSS) меняет только то, откуда пришли статьи."""
    items = extract(domain, what, articles, count, POSITIVE_EXTRA if label == "positive" else "")
    accepted, pending, rejected = [], [], []
    claimed_type = key if label == "negative" else ""

    for item in items:
        name = item["name_ru"].strip()
        term = normalize(item.get("search_term", ""))
        if seen.excluded_term(item.get("search_term", "")):
            with stats_lock:
                stats["дубли"] += 1
            continue
        if not seen.claim(name) or (term and not seen.claim_term(term)):
            with stats_lock:
                stats["дубли"] += 1
            continue
        ids = [i for i in (item.get("sources") or []) if isinstance(i, int) and 1 <= i <= len(articles)]
        if not ids:
            rejected.append({"name": name, "reason": "модель не сослалась ни на одну статью"})
            continue
        evidence = [articles[i - 1] for i in ids]
        facts = assess(item.get("search_term") or "")
        stage, stage_quote = None, ""
        if label == "negative" and claimed_type in ("N1", "N3d", "N3a"):
            import stage_judge  # отложенный импорт: stage_judge сам импортирует collect
            try:
                verdict = stage_judge.ask(name, evidence)
                stage, stage_quote = verdict["stage"], verdict["stage_quote"]
            except Exception:  # noqa: BLE001
                pass
        ok, why = judge(label, claimed_type, facts, stage, stage_quote)
        media_only = False
        if not ok and label == "positive" and facts.verdict == "unverified":
            # Научных публикаций нет: у самых ранних сигналов их и не бывает (18% золота).
            # ТЗ относит профессиональные отраслевые медиа к доверенным источникам —
            # соцсети и блоги отфильтрованы раньше. Стадию читает проверенная на золоте
            # модель по тем же статьям; стадия 5 — отказ.
            import stage_judge  # отложенный импорт: stage_judge сам импортирует collect
            try:
                verdict = stage_judge.ask(name, evidence)
            except Exception:  # noqa: BLE001
                verdict = None
            if verdict and verdict["stage"] <= 4:
                ok, media_only = True, True
                why = (f"медиа-подтверждение, научных публикаций нет; стадия по статьям "
                       f"{verdict['stage']}: {verdict['stage_quote'][:120]}")
            elif verdict:
                why = f"стадия 5 по статьям: {verdict['stage_quote'][:120]}"
        if ok:
            dates = sorted({(a.get("published_date") or "")[:16] for a in evidence} - {""})
            source_query_languages = sorted({a.get("search_language", "") for a in evidence} - {""})
            accepted.append({
                "id": f"{label[:3]}-{abs(hash(normalize(name))) % 10**7:07d}",
                "name": name, "domain": domain, "label": 1 if label == "positive" else 0,
                "stage": stage if stage is not None else "", "trend": "", "negative_type": claimed_type,
                "rationale": f"{item.get('note', '')} | факты: {why} | термин: {item.get('search_term', '')}"
                             + (f" | языки поисковых запросов: {', '.join(source_query_languages)}"
                                if source_query_languages else "")
                             + (f" | даты публикаций: {', '.join(dates)}" if dates else ""),
                "annotator": f"collect:{MODEL}+{source_tag}" + (":media_only" if media_only else ""),
                "evidence_urls": ";".join(a["url"] for a in evidence[:4]),
                "added_at": date.today().isoformat(),
            })
        elif facts.verdict == "unverified":
            pending.append({"name": name, "domain": domain, "search_term": item.get("search_term", ""),
                            "label": label, "reason": why,
                            "evidence_urls": ";".join(a["url"] for a in evidence[:4])})
        else:
            rejected.append({"name": name, "reason": why})

    writer.append(LABELED, accepted, LABELED_FIELDS)
    writer.append(PENDING, pending, ["name", "domain", "search_term", "label", "reason", "evidence_urls"])
    writer.append(REJECTED, rejected, ["name", "reason"])
    with stats_lock:
        stats["принято"] += len(accepted)
        stats["не проверено"] += len(pending)
        stats["отклонено"] += len(rejected)
    return f"[{domain}/{key}] найдено {len(items)}, принято {len(accepted)}, " \
           f"не проверено {len(pending)}, отклонено {len(rejected)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", choices=["positive", "negative"], required=True)
    parser.add_argument("--domain", action="append")
    parser.add_argument("--per-angle", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--subtopics", type=int, default=8, help="подтем на область (позитивы)")
    parser.add_argument("--cached-only", action="store_true",
                        help="только уже закэшированные запросы Tavily, новых не тратить")
    parser.add_argument("--topup", type=int, default=0, metavar="N",
                        help="добор: N новых подтем на область, общие ракурсы не повторяются")
    args = parser.parse_args()

    global CACHED_ONLY
    CACHED_ONLY = args.cached_only
    for key in ("TAVILY_API_KEY", "OPENROUTER_API_KEY"):
        if not os.getenv(key):
            print(f"{key} не задан", file=sys.stderr)
            return 1
    if not preflight():
        return 2

    # Исключения: золото (валидация) и датасет Арсения — наши данные собираются
    # независимо и потом сливаются с его, поэтому пересекаться не должны.
    names = []
    for path in (GOLD, LABELED, *sorted(p for p in EXCLUSIONS.glob("*.csv") if not p.name.endswith("_terms.csv"))):
        if path.exists() and path.stat().st_size > 0:
            with path.open(encoding="utf-8") as fh:
                names += [r["name"] for r in csv.DictReader(fh) if r.get("name")]
    excluded_terms = []
    for path in sorted(EXCLUSIONS.glob("*_terms.csv")):
        with path.open(encoding="utf-8") as fh:
            excluded_terms += [r["term"] for r in csv.DictReader(fh) if r.get("term")]
    seen = Seen(names, excluded_terms)

    domains = args.domain or list(DOMAINS)
    if args.label == "positive":
        # --topup: только новые подтемы, без повторения общих ракурсов, уже отработанных
        tasks = [] if args.topup else [(d, k, spec) for d in domains for k, spec in POSITIVE_ANGLES.items()]
        for d in domains:
            for s in (new_subtopics(d, args.topup) if args.topup else subtopics(d, args.subtopics)):
                spec = ([f"{s} startup raises funding", f"{s} emerges from stealth OR first pilot"], "news",
                        f"технологии в подтеме «{s}», которые развивают молодые компании, "
                        f"лаборатории или стартапы: недавние раунды, выход из stealth, первые пилоты")
                tasks.append((d, s, spec))
    else:
        tasks = [(d, k, spec) for d in domains for k, spec in NEGATIVE_ANGLES.items()]
    stats = {"принято": 0, "не проверено": 0, "отклонено": 0, "дубли": 0}
    stats_lock, writer = threading.Lock(), Writer()
    started = time.time()

    print(f"{args.label}: {len(tasks)} поисковых задач, {args.workers} потока\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_angle, args.label, d, k, spec, args.per_angle,
                               seen, writer, stats, stats_lock): (d, k) for d, k, spec in tasks}
        for future in as_completed(futures):
            domain, key = futures[future]
            try:
                print(f"  {future.result()}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{domain}/{key}] СБОЙ: {str(exc)[:100]}", flush=True)

    print(f"\nитог за {(time.time() - started) / 60:.1f} мин: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
