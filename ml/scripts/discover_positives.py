"""Целенаправленный сбор позитивов: поиск по следам ранней стадии + структурное вето.

Две ступени, и решение о метке НЕ остаётся за моделью:

  1. Обнаружение. Веб-поиск находит технологии по следам ранней стадии — выходы из
     stealth, раунды seed/A, первые пилоты, спиноуты лабораторий. Модель возвращает
     название, компании и ссылки. Это поиск кандидатов, а не разметка.

  2. Структурное вето. По англоязычному названию считается фразовый запрос в OpenAlex
     (title_and_abstract.search) и возраст статьи в Википедии. Кандидат отбраковывается,
     если по объективным признакам он зрелый: большой объём публикаций или давняя статья
     в Википедии.

Позитив принимается только если обе ступени сошлись. Так работали и методологи заказчика:
нашли в отраслевых медиа, проверили по существу.

Почему не намайненный из заголовков пул: ранние индустриальные сигналы (выход из stealth
с раундом $38M) в научных базах не отражены в принципе, там находятся только области
исследований вроде "adaptive control". Проверено прогоном авторазметчика на золоте.

Запуск:
    python ml/scripts/discover_positives.py --per-angle 5           # весь набор
    python ml/scripts/discover_positives.py --domain "Финтех"       # одна область
    python ml/scripts/discover_positives.py --dry-run               # без записи
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "ml" / "datasets" / "gold.csv"
LABELED = ROOT / "ml" / "datasets" / "labeled.csv"
REJECTED = ROOT / "ml" / "datasets" / "discovery_rejected.csv"
CACHE = ROOT / "ml" / ".cache"

from evidence_sources import assess, preflight

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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

MODEL = os.getenv("DISCOVERY_MODEL", "google/gemini-3.8-flash")
CONTACT = os.getenv("OPENALEX_MAILTO", "").strip() or "hackathon@example.org"
UA = {"User-Agent": f"weak-signal-radar/0.1 (hackathon research; {CONTACT})"}

DOMAINS = ("Индустриальный ИИ", "Роботы", "Инфраструктура ИИ", "Финтех", "Защита ИИ", "Edge")

# Разные ракурсы поиска дают разные слои ранней стадии: от лабораторий до первых контрактов.
ANGLES = (
    "компании вышли из stealth за последние 18 месяцев",
    "подняли seed или Series A за последние 12 месяцев",
    "запустили первые пилоты или первые коммерческие поставки",
    "спиноуты университетов и исследовательских лабораторий",
    "новые технические подходы, обсуждаемые на профильных конференциях",
)

# Домены вендоров: их сайты не считаются независимым подтверждением.
VENDOR_HINTS = (".ai", ".io", ".dev", ".tech")
KNOWN_INDEPENDENT = (
    "techcrunch", "venturebeat", "siliconangle", "theregister", "ieee", "nature",
    "arxiv", "reuters", "bloomberg", "ft.com", "wired", "technologyreview",
    "thehackernews", "securityweek", "eetimes", "datacenterdynamics", "pulse2",
    "biometricupdate", "crunchbase", "prnewswire", "businesswire", "nist.gov",
    "europa.eu", "cbr.ru", "tadviser", "cnews", "habr",
)

# Структурное вето
MATURE_WIKI_AGE_YEARS = 7
MATURE_VOLUME_3Y = 5000       # работ за 3 года: столько бывает только у массовых тем
MIN_SOURCES = 2

DISCOVERY_PROMPT = """Ты аналитик технологической разведки. Найди в открытых источниках
конкретные технологии на РАННЕЙ стадии развития.

Область: {domain}
Критерий поиска: {angle}

Требования к результату:
- Технология, а не компания, продукт или отрасль. Плохо: «Straiker», «кибербезопасность».
  Хорошо: «Поведенческий межсетевой экран для вызовов инструментов ИИ-агентами».
- Название по-русски, 5-12 слов, в стиле «Что именно делается для какой задачи».
- НЕ включай зрелые технологии, отраслевые стандарты, массово внедрённые решения,
  а также общие темы исследований вроде «адаптивное управление».
- Минимум 2 ссылки, желательно не только на сайт самой компании.

Ответ строго JSON без markdown:
{{"items":[{{"name_ru":"...","name_en":"...","search_term":"...","companies":"...","stage_note":"...","sources":["url1","url2"]}}]}}

name_en — английское описательное название технологии.
search_term — КАНОНИЧЕСКИЙ отраслевой термин из 2-4 слов для точного поиска в научных
базах. Не описание, а устоявшееся название. Плохо: "Magnetically Steered Endovascular
Microcatheters for Neurovascular Intervention". Хорошо: "steerable microcatheter".
Поиск идёт по точной фразе, поэтому длинное описание не найдёт ничего.
stage_note — одна фраза о стадии со ссылкой на факт: раунд, выход из stealth, первый пилот.
Найди {count} технологий."""


def _http_json(url: str, *, data: dict | None = None, headers: dict | None = None,
               tries: int = 4, timeout: int = 240) -> dict:
    payload = json.dumps(data).encode() if data is not None else None
    last: Exception | None = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(
                url, data=payload, headers={**UA, **(headers or {})},
                method="POST" if payload else "GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            time.sleep(5 * (attempt + 1) + random.random())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (attempt + 1) + random.random())
    raise RuntimeError(f"{url}: не удалось за {tries} попыток: {last}")


def _parse_json_block(text: str) -> dict:
    """Модели иногда возвращают JSON с висящей запятой — чиним, а не падаем."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("в ответе нет JSON")
    blob = match.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", blob))


def discover(domain: str, angle: str, count: int, _retry: bool = False) -> list[dict]:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY не задан")
    response = _http_json(
        OPENROUTER_URL,
        data={
            "model": MODEL,
            "temperature": 0.4,
            "plugins": [{"id": "web", "max_results": 6}],
            "messages": [{"role": "user", "content": DISCOVERY_PROMPT.format(
                domain=domain, angle=angle, count=count)}],
        },
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        items = _parse_json_block(response["choices"][0]["message"]["content"]).get("items", [])
    except (ValueError, json.JSONDecodeError):
        # Модель иногда рвёт JSON на длинных строках. Один повтор дешевле, чем
        # потерять целый ракурс поиска, как это случалось в прошлых прогонах.
        if _retry:
            raise
        return discover(domain, angle, count, _retry=True)
    return [i for i in items if isinstance(i, dict) and i.get("name_ru")]


def normalize(name: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", " ", name.lower()).strip()


def is_duplicate(name: str, seen: set[str]) -> bool:
    """Грубая проверка на повтор: пересечение значимых слов выше 60%."""
    words = {w for w in normalize(name).split() if len(w) > 3}
    if not words:
        return True
    for other in seen:
        other_words = {w for w in other.split() if len(w) > 3}
        if other_words and len(words & other_words) / len(words | other_words) > 0.6:
            return True
    return False


def structural_veto(name_en: str) -> tuple[str, str]:
    """Обёртка над общим модулем проверки зрелости (evidence_sources.assess)."""
    facts = assess(name_en, verbose=True)
    return facts.verdict, facts.note


def independent_sources(sources: list[str]) -> int:
    count = 0
    for url in sources:
        host = urllib.parse.urlparse(url).netloc.lower()
        if any(k in host for k in KNOWN_INDEPENDENT):
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", action="append")
    parser.add_argument("--per-angle", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    domains = args.domain or list(DOMAINS)
    if not preflight():
        return 2

    # Золото — валидационная выборка, её записи в обучение попасть не должны.
    with GOLD.open(encoding="utf-8") as fh:
        seen = {normalize(r["name"]) for r in csv.DictReader(fh)}
    gold_size = len(seen)
    if LABELED.exists() and LABELED.stat().st_size > 0:
        with LABELED.open(encoding="utf-8") as fh:
            seen |= {normalize(r["name"]) for r in csv.DictReader(fh) if r.get("name")}

    accepted, rejected = [], []
    stats = {"найдено": 0, "дубли": 0, "вето": 0, "не проверено": 0, "мало источников": 0, "принято": 0}
    pending: list[dict] = []

    for domain in domains:
        for angle in ANGLES:
            print(f"[{domain}] {angle}")
            try:
                items = discover(domain, angle, args.per_angle)
            except Exception as exc:  # noqa: BLE001
                print(f"    пропуск: {exc}", file=sys.stderr)
                continue
            for item in items:
                stats["найдено"] += 1
                name = item["name_ru"].strip()
                if is_duplicate(name, seen):
                    stats["дубли"] += 1
                    continue
                sources = [s for s in (item.get("sources") or []) if isinstance(s, str)]
                if len(sources) < MIN_SOURCES:
                    stats["мало источников"] += 1
                    rejected.append({"name": name, "reason": f"источников {len(sources)}"})
                    continue
                verdict, note = structural_veto(item.get("search_term") or item.get("name_en", ""))
                if verdict == "mature":
                    stats["вето"] += 1
                    rejected.append({"name": name, "reason": note})
                    continue
                if verdict == "unverified":
                    stats["не проверено"] += 1
                    pending.append({
                        "id": f"pos-{abs(hash(normalize(name))) % 10**6:06d}",
                        "name": name, "domain": domain, "name_en": item.get("name_en", ""), "search_term": item.get("search_term", ""),
                        "companies": item.get("companies", ""),
                        "stage_note": item.get("stage_note", ""),
                        "sources": ";".join(sources[:4]), "reason": note,
                    })
                    continue
                seen.add(normalize(name))
                stats["принято"] += 1
                accepted.append({
                    "id": f"pos-{abs(hash(normalize(name))) % 10**6:06d}",
                    "name": name, "domain": domain, "label": 1, "stage": "", "trend": "",
                    "negative_type": "",
                    "rationale": f"{item.get('stage_note','')} | структура: {note} | "
                                 f"независимых источников: {independent_sources(sources)}",
                    "annotator": f"discovery:{MODEL}",
                    "evidence_urls": ";".join(sources[:4]),
                    "added_at": date.today().isoformat(),
                })
            print(f"    принято {stats['принято']}, вето {stats['вето']}, "
                  f"не проверено {stats['не проверено']}, дублей {stats['дубли']}")

    print(f"\nитог: {stats}")
    print(f"(в дубли входят совпадения с {gold_size} золотыми записями — они валидационные)")

    if args.dry_run:
        for item in accepted[:10]:
            print(f"\n  • {item['name']}\n    {item['rationale'][:150]}")
        return 0

    if accepted:
        exists = LABELED.exists() and LABELED.stat().st_size > 0
        with LABELED.open("a" if exists else "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(accepted[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(accepted)
        print(f"\nзаписано {len(accepted)} позитивов -> {LABELED.relative_to(ROOT)}")
    if pending:
        pending_file = ROOT / "ml" / "datasets" / "discovery_pending.csv"
        with pending_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(pending[0]))
            writer.writeheader()
            writer.writerows(pending)
        print(f"НЕ ПРОВЕРЕНО {len(pending)} -> {pending_file.relative_to(ROOT)}")
        print("  в обучение не идут: структурная проверка не отработала. Повторить прогон позже.")
    if rejected:
        with REJECTED.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["name", "reason"])
            writer.writeheader()
            writer.writerows(rejected)
        print(f"отбраковано {len(rejected)} -> {REJECTED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
