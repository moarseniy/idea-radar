"""RSS-коннектор отраслевых изданий: бесплатный источник статей для сбора позитивов.

Ленты на WordPress листаются назад параметром ?paged=N, поэтому за один бесплатный запрос
на страницу можно пройти 12-18 месяцев публикаций. Страницы кэшируются на диск.

Отбор по ключевым словам (раунды, выход из stealth, пилоты, спиноуты) — дальше статьи
идут в ту же обработку, что и выдача Tavily (collect.process_articles): одинаковые
извлечение, исключения и проверка стадии для любого источника.

Проверка лент:  python ml/scripts/rss_source.py --probe
Сбор:           python ml/scripts/rss_source.py --collect --months 18
"""
from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402
from evidence_sources import HostLimiter, preflight  # noqa: E402

CACHE = collect.ROOT / "ml" / ".cache" / "rss"
UA = {"User-Agent": "Mozilla/5.0 (compatible; weak-signal-radar/0.1; research)"}

# Отраслевые издания по областям. Доверенность уровня B по реестру: профессиональные
# отраслевые медиа. Состав проверяется режимом --probe.
FEEDS = {
    "Роботы": ["https://www.therobotreport.com/feed/", "https://roboticsandautomationnews.com/feed/"],
    "Защита ИИ": ["https://www.securityweek.com/feed/", "https://securityboulevard.com/feed/"],
    "Инфраструктура ИИ": ["https://www.nextplatform.com/feed/", "https://www.servethehome.com/feed/"],
    "Финтех": ["https://fintech.global/feed/", "https://thefintechtimes.com/feed/", "https://www.pymnts.com/feed/"],
    "Edge": ["https://www.edgeir.com/feed", "https://www.electronicsweekly.com/feed/"],
    "Индустриальный ИИ": ["https://iiot-world.com/feed/", "https://www.themanufacturer.com/feed/"],
}

# Следы ранней стадии в заголовке или анонсе.
SIGNAL_RE = re.compile(
    r"\b(raise[sd]?|raising|seed|series a|pre-seed|funding|funded|backed|stealth|"
    r"launch(?:es|ed)?|unveil(?:s|ed)?|debut(?:s|ed)?|pilot|spin[- ]?out|spins out|start-?up)\b", re.I)

_limiters: dict[str, HostLimiter] = {}
_limiters_lock = threading.Lock()


def _limiter(url: str) -> HostLimiter:
    host = urllib.parse.urlparse(url).netloc
    with _limiters_lock:
        return _limiters.setdefault(host, HostLimiter(1.5))  # вежливо к бесплатным источникам


def fetch_page(feed: str, page: int) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", feed.lower())[:80]
    cached = CACHE / f"{key}_p{page}.xml"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    url = feed if page == 1 else feed + ("&" if "?" in feed else "?") + f"paged={page}"
    _limiter(feed).wait()
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(body, encoding="utf-8")
    return body


def _text(fragment: str) -> str:
    fragment = re.sub(r"(?is)<!\[CDATA\[(.*?)\]\]>", r"\1", fragment or "")
    fragment = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse(xml: str) -> list[dict]:
    items = []
    for block in re.findall(r"(?is)<item[ >].*?</item>", xml):
        def tag(name):
            m = re.search(rf"(?is)<{name}[^>]*>(.*?)</{name}>", block)
            return m.group(1) if m else ""
        try:
            published = parsedate_to_datetime(_text(tag("pubDate")))
        except (TypeError, ValueError):
            published = None
        items.append({
            "title": _text(tag("title")),
            "url": _text(tag("link")),
            "published": published,
            "published_date": published.strftime("%a, %d %b %Y") if published else "",
            "summary": _text(tag("description"))[:600],
            "raw_content": _text(tag("content:encoded")) or _text(tag("description")),
        })
    return items


def walk(feed: str, cutoff: datetime, max_pages: int = 120) -> list[dict]:
    """Листать ленту назад до даты cutoff. Отказ страницы — конец ленты, а не ошибка."""
    out, urls = [], set()
    for page in range(1, max_pages + 1):
        try:
            items = parse(fetch_page(feed, page))
        except Exception:  # noqa: BLE001
            break
        fresh = [i for i in items if i["url"] and i["url"] not in urls]
        if not fresh:
            # Лента без листания отдаёт на каждой странице одно и то же (Next Platform:
            # 120 одинаковых страниц). Страница без новых статей — конец ленты.
            break
        urls.update(i["url"] for i in fresh)
        out += fresh
        dated = [i["published"] for i in items if i["published"]]
        if dated and min(dated) < cutoff:
            break
    return [i for i in out if not i["published"] or i["published"] >= cutoff]


def probe() -> int:
    for domain, feeds in FEEDS.items():
        for feed in feeds:
            try:
                p1 = parse(fetch_page(feed, 1))
                p5 = parse(fetch_page(feed, 5)) if p1 else []
                oldest = min((i["published"] for i in p1 + p5 if i["published"]), default=None)
                full = sum(1 for i in p1 if len(i["raw_content"]) > 1500)
                print(f"  [{domain[:10]:10}] {feed[:48]:50} стр.1: {len(p1):2}, стр.5: {len(p5):2}, "
                      f"до {oldest:%Y-%m-%d}" if oldest else f"  [{domain[:10]}] {feed[:48]} пусто",
                      f"| полный текст в ленте: {full}/{len(p1)}" if p1 else "")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{domain[:10]:10}] {feed[:48]:50} СБОЙ: {str(exc)[:50]}")
    return 0


def collect_rss(months: int, batch: int, workers: int, domains: list[str] | None) -> int:
    if not preflight():
        return 2
    import stage_judge  # тексты статей без полного содержания в ленте — прямым запросом

    cutoff = datetime.now(timezone.utc) - timedelta(days=30 * months)
    names = []
    for path in (collect.GOLD, collect.LABELED,
                 *sorted(p for p in collect.EXCLUSIONS.glob("*.csv") if not p.name.endswith("_terms.csv"))):
        if path.exists() and path.stat().st_size > 0:
            names += [r["name"] for r in csv.DictReader(path.open(encoding="utf-8")) if r.get("name")]
    terms = [r["term"] for f in sorted(collect.EXCLUSIONS.glob("*_terms.csv"))
             for r in csv.DictReader(f.open(encoding="utf-8")) if r.get("term")]
    seen = collect.Seen(names, terms)
    used_urls = {u for r in csv.DictReader(collect.LABELED.open(encoding="utf-8"))
                 for u in r["evidence_urls"].split(";")}
    stats = {"принято": 0, "не проверено": 0, "отклонено": 0, "дубли": 0}
    stats_lock, writer = threading.Lock(), collect.Writer()
    what = ("технологии, которые развивают молодые компании, стартапы и лаборатории: "
            "недавние раунды, выход из stealth, первые пилоты и поставки")

    # Разные сайты листаются параллельно; к одному сайту — не чаще раза в 1,5 с.
    feeds = [(d, f) for d, fs in FEEDS.items() if not domains or d in domains for f in fs]
    with ThreadPoolExecutor(max_workers=len(feeds)) as pool:
        walked = dict(zip(feeds, pool.map(lambda df: walk(df[1], cutoff), feeds)))

    by_domain: dict[str, list[dict]] = {}
    for (domain, feed), got in walked.items():
        picked = [a for a in got if SIGNAL_RE.search(a["title"] + " " + a["summary"])
                  and a["url"] not in used_urls]
        used_urls.update(a["url"] for a in picked)  # одна статья — в одну область и один раз
        print(f"  [{domain}] {feed[:45]}: статей {len(got)}, со следами ранней стадии {len(picked)}", flush=True)
        by_domain.setdefault(domain, []).extend(picked)

    # В ленте только анонс — берём полный текст статьи прямым запросом.
    short = [a for arts in by_domain.values() for a in arts if len(a["raw_content"]) < 800]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for a, text in zip(short, pool.map(lambda a: stage_judge.fetch_text(a["url"]), short)):
            a["raw_content"] = text or a["raw_content"]
    print(f"  полных текстов дозагружено: {sum(1 for a in short if len(a['raw_content']) >= 800)} из {len(short)}")

    tasks = []
    for domain, articles in by_domain.items():
        for i in range(0, len(articles), batch):
            tasks.append((domain, f"rss#{i // batch + 1}", articles[i:i + batch]))

    print(f"\nпакетов на извлечение: {len(tasks)}\n", flush=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(collect.process_articles, "positive", d, k, arts, what, batch, "rss",
                               seen, writer, stats, stats_lock): (d, k) for d, k, arts in tasks}
        for future in as_completed(futures):
            d, k = futures[future]
            try:
                print(f"  {future.result()}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{d}/{k}] СБОЙ: {str(exc)[:90]}", flush=True)
    print(f"\nитог за {(time.time() - started) / 60:.1f} мин: {stats}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--domain", action="append")
    args = parser.parse_args()
    if args.probe:
        return probe()
    if args.collect:
        return collect_rss(args.months, args.batch, args.workers, args.domain)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
