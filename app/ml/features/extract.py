"""Разметка технологии признаками общей схемы обучения и сервиса.

Один и тот же код размечает обучающий датасет и кандидатов в сервисе:

    from app.ml.features.extract import featurize
    row = featurize("Нейроморфные чипы для edge-устройств", "Edge", snapshot=date(2026, 9, 22))

Конвейер:
  1. query  — LLM переводит название в короткий канонический английский термин и до двух
              синонимов (поиск по ним одинаков для всех строк, независимо от того, откуда
              строка пришла в датасет);
  2. evidence — наука (Crossref + OpenAlex), патенты (FPO), гранты (NIH RePORTER/CORDIS),
              медиа (Google News)
              в окнах относительно даты среза;
  3. events — LLM читает пронумерованные заголовки и выписывает события (пилот, внедрение,
              раунд, стандарт...) со ссылкой на номер заголовка. Дата события берётся
              из заголовка, а не из ответа модели; события без валидной ссылки отбрасываются.
              Стадию модель оценивает по той же ленте;
  4. compute — чистые формулы от собранного: счётчики, логарифмы, рост, HHI.

Детерминизм: temperature=0 и seed у LLM, все сетевые и LLM-ответы кэшируются по хэшу
запроса. Повторная разметка на тот же срез даёт те же значения. Признаки, которые не
удалось измерить, остаются пустыми (""), ноль ставится только при измеренном нуле.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from urllib.parse import urlsplit

from app import registry
from app.ml.env import feature_llm, strip_thinking
from app.ml.features import evidence as ev
from app.ml.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from app.radar.connectors._terms import relevant

# LLM разметки: по умолчанию openai/gpt-5.6-luna через OpenRouter; при RADAR_PROVIDER=local —
# локальный сервер; FEATURE_MODEL / FEATURE_LLM_URL / FEATURE_LLM_EXTRA — только для разметки
# (app/ml/env.py: feature_llm).
_LLM = feature_llm()
MODEL, LLM_URL, LLM_EXTRA, LLM_KEY = _LLM["model"], _LLM["url"], _LLM["extra"], _LLM["key"]
EXTRACTOR_VERSION = "featurize-v2"  # v2: доверие к источникам из app/registry

# Доверие к источникам — из общего реестра app/registry/sources.json (его же использует радар).

MARKETING = {"revolutionary", "breakthrough", "game-changing", "unprecedented", "transformative",
             "disruptive"}
TECHNICAL = {"model", "sensor", "processor", "computing", "learning", "electrochemical", "photonic",
             "quantum", "robot", "network", "protocol", "algorithm", "controller", "semiconductor",
             "battery", "electrolyte", "actuator", "metamaterial", "microfluidic", "federated"}


# ------------------------------------------------------------------------------ LLM

def llm_json(prompt: str) -> dict:
    """Вызов LLM с temperature=0, seed и дисковым кэшем по хэшу (модель, промпт)."""
    # max_tokens с запасом на рассуждения: без него часть провайдеров OpenRouter просит
    # весь контекст модели под ответ и отвечает 400 (превышение окна).
    body = {"model": MODEL, "temperature": 0, "seed": 7, "max_tokens": 16000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}], **LLM_EXTRA}

    def parse(text: str) -> dict:
        data = json.loads(text)
        content = strip_thinking((data.get("choices") or [{}])[0].get("message", {}).get("content") or "")
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("пустой ответ модели")
        return json.loads(match.group(0))

    # Keep each feature request bounded so one slow provider cannot hold a radar run
    # past its deadline. Local inference can override both limits in .env.
    try:
        timeout = max(15, min(600, float(os.getenv("FEATURE_LLM_TIMEOUT", "60"))))
    except ValueError:
        timeout = 60
    try:
        tries = max(1, min(2, int(os.getenv("FEATURE_LLM_TRIES", "1"))))
    except ValueError:
        tries = 1
    return ev.cached_fetch("llm", f"{LLM_URL}/chat/completions", parse=parse,
                           body=body, headers={"Authorization": f"Bearer {LLM_KEY}"},
                           cache_key=f"{MODEL}\n{json.dumps(LLM_EXTRA, sort_keys=True)}\n{prompt}" if LLM_EXTRA
                           else f"{MODEL}\n{prompt}", tries=tries, timeout=timeout)


QUERY_PROMPT = """Технология: «{name}» (область: {domain}).

Дай поисковый запрос для англоязычных научных статей, патентов и новостей:
- "term": каноническое английское название самой технологии, 1-4 слова, так, как его пишут
  в статьях (без области применения, без слов technology/system/solution/platform,
  если они не часть устойчивого названия);
- "aliases": до 2 распространённых синонимов или аббревиатур (может быть пусто);
- "is_technology": true, если это технология или технический подход; false, если это
  конкретный программный продукт/бренд, маркетинговое обещание или бизнес-модель.

Ответ строго JSON: {{"term":"...","aliases":["..."],"is_technology":true}}"""


def derive_query(name: str, domain: str) -> dict:
    data = llm_json(QUERY_PROMPT.format(name=name.strip(), domain=domain.strip()))
    term = re.sub(r"\s+", " ", str(data.get("term") or "")).strip().strip('"')
    aliases = [re.sub(r"\s+", " ", str(a)).strip().strip('"') for a in data.get("aliases") or []]
    aliases = [a for a in aliases if a and a.lower() != term.lower()][:2]
    return {"term": term, "aliases": aliases, "is_technology": bool(data.get("is_technology", True))}


EVENTS_PROMPT = """Технология: «{term}» ({name}). Срез: {snapshot}.

Ниже пронумерованные заголовки новостей о ней за 4 года (дата, издание, заголовок) и
названия недавних научных работ и патентов.

{headlines}

Научные работы: {papers}
Патенты: {patents}
Структурированные API-свидетельства: {structured_sources}

Задача 1 — события. Выпиши только события, прямо описанные в заголовке, про ЭТУ технологию
(не про соседние). Типы:
  pilot — пилот, испытание у заказчика, PoC с названным заказчиком;
  deployment — промышленное/коммерческое внедрение, запуск в эксплуатацию;
  product — выпуск коммерческого продукта;
  funding — раунд финансирования компании, чей продукт основан на технологии;
  grant — государственный грант или программа финансирования;
  standard_final — принятый стандарт; regulation — законопроект, регулирование, проект стандарта;
  procurement — госзакупка или тендер.
Для каждого: "h" — номер заголовка, "type", "company" — компания-разработчик/поставщик,
"customer" — названный заказчик (если есть), для funding: "amount_usd_m" (миллионы USD,
число или null), "round" (pre-seed/seed/A/B/C/D+/grant/other), "strategic" (true, если
среди инвесторов корпорация, а не фонд).

Официальные записи из грантовых, тендерных, клинических и регуляторных реестров, а также
репозитории и модели — структурированные данные. Используй их для контекста и стадии.
Статистика GitHub/Hugging Face сама по себе не доказывает коммерческое внедрение.
Не дублируй структурированные события как новостные события: приложение добавит доступные
grant/procurement/regulation записи напрямую.

Задача 2 — оценки по всей ленте:
"stage": 1 — исследования; 2 — прототип/PoC; 3 — пилоты у заказчиков; 4 — раннее
коммерческое внедрение (единицы-десятки внедрений); 5 — масштабирование/зрелая массовая
технология с лидерами рынка и стандартами.
"mass_market": true, если технология уже массово доступна потребителям или бизнесу.
"rebranding_similarity": 0..1 — насколько это новое название давно зрелой технологии.

Ответ строго JSON:
{{"events":[{{"h":1,"type":"funding","company":"...","customer":null,"amount_usd_m":12,"round":"A","strategic":false}}],
"stage":3,"mass_market":false,"rebranding_similarity":0.1}}"""


def extract_events(query: dict, name: str, snapshot: date, headlines: list[dict],
                   sci: ev.Science, pat: ev.Patents,
                   structured_refs: list[dict] | None = None) -> dict:
    listing = "\n".join(f"{i}. {h['date']} | {h['publisher']} | {h['title']}"
                        for i, h in enumerate(headlines, 1)) or "(новостей не найдено)"
    structured_listing = "\n".join(
        f"- {ref.get('source_provider')}: {ref.get('title')} | {ref.get('observed_at') or 'дата неизвестна'} | "
        f"{ref.get('url')}\n  {(ref.get('document') or {}).get('text', '')[:900]}"
        for ref in (structured_refs or [])[:12]) or "(нет)"
    prompt = EVENTS_PROMPT.format(
        term=query["term"], name=name.strip(), snapshot=snapshot.isoformat(), headlines=listing,
        papers="; ".join(sci.titles[:15]) or "нет", patents="; ".join(pat.titles[:10]) or "нет",
        structured_sources=structured_listing)
    data = llm_json(prompt)
    events = []
    for ref in structured_refs or []:
        raw = ref.get("structured_event")
        if not isinstance(raw, dict) or raw.get("type") not in EVENT_TYPES:
            continue
        try:
            event_date = date.fromisoformat(str(raw.get("date") or "")[:10])
        except ValueError:
            continue
        if event_date <= snapshot:
            events.append({**raw, "date": event_date.isoformat(), "source_url": ref.get("url"),
                           "structured": True})
    for raw in data.get("events") or []:
        try:
            idx = int(raw.get("h"))
        except (TypeError, ValueError):
            continue
        if not 1 <= idx <= len(headlines) or raw.get("type") not in EVENT_TYPES:
            continue
        item = dict(raw)
        item["date"] = headlines[idx - 1]["date"]  # дата из источника, не из модели
        events.append(item)
    deduplicated, seen_events = [], set()
    for event in events:
        key = (event.get("type"), str(event.get("company") or "").casefold().strip(),
               str(event.get("customer") or "").casefold().strip(), event.get("date"))
        if key not in seen_events:
            seen_events.add(key)
            deduplicated.append(event)
    try:
        stage = int(data.get("stage"))
    except (TypeError, ValueError):
        stage = None
    try:
        rebrand = min(1.0, max(0.0, float(data.get("rebranding_similarity"))))
    except (TypeError, ValueError):
        rebrand = None
    return {"events": deduplicated, "stage": stage if stage in (1, 2, 3, 4, 5) else None,
            "mass_market": data.get("mass_market"), "rebranding_similarity": rebrand}


EVENT_TYPES = {"pilot", "deployment", "product", "funding", "grant", "standard_final",
               "regulation", "procurement"}


def _needs_patent_metrics(requested_features: set[str] | None) -> bool:
    if requested_features is None:
        return True
    patent_inputs = {feature for feature in FEATURE_NAMES if feature.startswith("patent_")} | {"missing_patents"}
    return bool(requested_features & patent_inputs)


def _publication_visualization(science: ev.Science) -> dict:
    available = science.ok
    latest_year = max((int(year) for year in science.year_counts if str(year).isdigit()), default=None)
    recent_year_counts = ({year: count for year, count in science.year_counts.items()
                           if latest_year is not None and str(year).isdigit()
                           and int(year) >= latest_year - 5}
                          if available else {})
    return {
        "science_status": ("unavailable" if not available else
                           "complete" if science.history_complete else "partial"),
        "publication_total_found": science.total if available else None,
        "publication_total_complete": science.history_complete,
        "publication_recent_3y_found": science.count_3y if science.recent_3y_complete else None,
        "publication_recent_share_found": (round(science.count_3y / science.total, 4)
                                            if science.history_complete and science.recent_3y_complete
                                            and science.total else None),
        # The compact list sparkline stays at six annual bins; the full timeline
        # uses evidence_timeline and expands to every available dated year.
        "publication_year_counts": recent_year_counts,
        "publication_total_capped": science.older_results_capped if available else False,
        "publication_years_capped": science.recent_results_capped if available else False,
    }


def _evidence_timeline_visualization(snapshot: date, science: ev.Science, media: ev.Media,
                                    structured_refs: list[dict]) -> dict:
    """Aggregate per-channel search results; do not derive the chart from citation links."""
    science_counts = science.year_counts if science.ok else {}
    news_counts: dict[str, int] = {}
    found: dict[str, dict[str, int]] = {
        key: {} for key in ("funding", "adoption", "community", "other", "news")
    }
    start = ev.years_before(snapshot, 4)
    seen_news: set[tuple[str, str]] = set()

    # Google News exposes dated hits rather than an annual total. Count the unique
    # returned title/publisher pairs; years outside its four-year query window stay unknown.
    if media.ok:
        for items in media.windows.values():
            for item in items:
                item_date = str(item.get("date") or "")
                title = re.sub(r"\W+", " ", str(item.get("title") or "").casefold()).strip()
                host = str(item.get("host") or item.get("publisher") or "").casefold().strip()
                try:
                    published = date.fromisoformat(item_date[:10])
                except ValueError:
                    continue
                key = (title, host)
                if published > snapshot or published < start or not title or key in seen_news:
                    continue
                seen_news.add(key)
                news_counts[str(published.year)] = news_counts.get(str(published.year), 0) + 1

    # Structured connectors return capped record lists. These are counts of rows actually
    # returned by those channels, not totals from an unbounded world-wide catalogue.
    for ref in structured_refs:
        raw_event = ref.get("structured_event") or {}
        raw_date = ((ref.get("document") or {}).get("date") or ref.get("observed_at")
                    or raw_event.get("date"))
        try:
            event_date = date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            continue
        if event_date > snapshot:
            continue
        event_type = str(raw_event.get("type") or "").lower()
        provider = str(ref.get("source_provider") or "").lower()
        source_channel = str(ref.get("channel") or "").lower()
        if event_type in {"pilot", "deployment", "product", "procurement", "trial", "regulation", "standard_final"}:
            channel = "adoption"
        elif event_type in {"grant", "funding", "funding_round", "grant_call"} or source_channel == "money":
            channel = "funding"
        elif "github" in provider or "huggingface" in provider or source_channel == "community":
            channel = "community"
        elif source_channel == "market" or event_type in {"news", "media"}:
            channel = "news"
        else:
            channel = "other"
        key = str(event_date.year)
        if channel == "news":
            title = re.sub(r"\W+", " ", str(ref.get("title") or "").casefold()).strip()
            host = (urlsplit(str(ref.get("url") or "")).hostname or provider).casefold()
            identity = (title, host)
            if title and identity in seen_news:
                continue
            if title:
                seen_news.add(identity)
        found[channel][key] = found[channel].get(key, 0) + 1

    observed_years = [int(year) for year in science_counts]
    observed_years.extend(int(year) for values in found.values() for year in values)
    observed_years.extend(int(year) for year in news_counts)
    first_year = min(observed_years, default=snapshot.year)
    years = [str(year) for year in range(first_year, snapshot.year + 1)]

    def series(values: dict[str, int | None] | None = None):
        values = values or {}
        return {year: values.get(year) for year in years}

    channels = {
        "science": {"label": "Научные работы", "counts": series(science_counts),
                    "capped": science.older_results_capped or science.recent_results_capped},
        "patents": {"label": "Патенты", "counts": series(), "capped": False},
        "news": {"label": "Новости и медиа", "counts": series(),
                 "capped": any(len(items) >= 100 for items in media.windows.values())},
        "funding": {"label": "Финансирование и гранты", "counts": series(), "capped": True},
        "adoption": {"label": "Пилоты, внедрения и закупки", "counts": series(), "capped": True},
        "community": {"label": "Технические сообщества", "counts": series(), "capped": True},
        "other": {"label": "Другие источники", "counts": series(), "capped": True},
    }

    if media.ok:
        for year in years:
            year_start = date(int(year), 1, 1)
            if year_start >= start:
                channels["news"]["counts"][year] = news_counts.get(year, 0)
            elif news_counts.get(year):
                channels["news"]["counts"][year] = news_counts[year]

    for channel, values in found.items():
        if values:
            first_channel_year = min(map(int, values))
            channel_counts = {
                year: (values.get(year, 0) if int(year) >= first_channel_year else None)
                for year in years
            }
            if channel == "news":
                channel_counts = {
                    year: (None if count is None and channels["news"]["counts"][year] is None
                           else (count or 0) + (channels["news"]["counts"][year] or 0))
                    for year, count in channel_counts.items()
                }
            channels[channel]["counts"] = channel_counts

    return {"years": years, "channels": channels,
            "note": "Показаны найденные датированные результаты по каждому доступному каналу; шкала не обрезается по фиксированному числу лет. Это не полный мировой подсчёт: у источников свои периоды покрытия и лимиты выдачи, а один материал может попасть в несколько каналов."}


def select_headlines(med: ev.Media, per_window: int = 30) -> list[dict]:
    """Детерминированная выборка ленты для LLM: по окнам, свежие первыми, без дублей."""
    seen, out = set(), []
    for window in ("0-6m", "6-12m", "12-24m", "24-48m"):
        items = sorted(med.windows.get(window, []), key=lambda h: (h["date"], h["title"]), reverse=True)
        taken = 0
        for h in items:
            key = re.sub(r"\W+", " ", h["title"].lower()).strip()
            if key in seen or taken >= per_window:
                continue
            seen.add(key)
            out.append(h)
            taken += 1
    return out


def _direct_structured_events(refs: list[dict], snapshot: date) -> list[dict]:
    events, seen = [], set()
    for ref in refs:
        raw = ref.get("structured_event")
        if not isinstance(raw, dict) or raw.get("type") not in EVENT_TYPES:
            continue
        try:
            event_date = date.fromisoformat(str(raw.get("date") or "")[:10])
        except ValueError:
            continue
        if event_date > snapshot:
            continue
        event = {**raw, "date": event_date.isoformat(), "source_url": ref.get("url"), "structured": True}
        key = (event.get("type"), str(event.get("company") or "").casefold().strip(),
               str(event.get("customer") or "").casefold().strip(), event["date"])
        if key not in seen:
            seen.add(key)
            events.append(event)
    return events


# ------------------------------------------------------------------------- формулы

def _log(x: float) -> float:
    return round(math.log1p(max(0.0, x)), 6)


def _growth(cur: int, prev: int) -> float:
    return round(math.log1p(cur) - math.log1p(prev), 6)


def _hhi(labels: list[str]):
    counts = Counter(x.lower() for x in labels if x)
    return _hhi_counts(counts)


def _hhi_counts(counts: dict[str, int] | Counter):
    total = sum(counts.values())
    return round(sum((c / total) ** 2 for c in counts.values()), 6) if total else ""


def _age(snapshot: date, iso: str | None):
    if not iso:
        return ""
    try:
        return round((snapshot - date.fromisoformat(iso[:10])).days / 365.25, 3)
    except ValueError:
        return ""


def _months_between(a: date, b: date) -> int:
    return max(1, (b.year - a.year) * 12 + b.month - a.month)


def compute(snapshot: date, query: dict, sci: ev.Science, pat: ev.Patents, med: ev.Media,
            ext: dict | None, structured_refs: list[dict] | None = None, historical: bool = False) -> dict:
    """Признаки по собранным свидетельствам. historical — срез в прошлом (машина времени)."""
    f: dict = {name: "" for name in FEATURE_NAMES}
    s = snapshot.isoformat()
    cut = {m: ev.years_before(snapshot, m / 12).isoformat() for m in (6, 12, 24, 36, 48)}

    # ---- стадия (LLM по ленте)
    if ext and ext.get("stage"):
        for i, name in enumerate(["stage_research", "stage_poc", "stage_pilot",
                                  "stage_early_adoption", "stage_scaling_or_mature"], 1):
            f[name] = int(ext["stage"] == i)

    # ---- наука
    if sci.ok:
        w0, w1, w2 = sci.windows
        f["papers_log_3y"] = _log(sci.count_3y)
        f["paper_growth_2y"] = _growth(w0, w1)
        f["paper_acceleration"] = round(_growth(w0, w1) - _growth(w1, w2), 6)
        # Crossref отдаёт число цитирований на сегодня: в прошлом это сведения из будущего
        # (статьи по темам, которые потом выстрелили, набрали цитирования после среза) —
        # признак не считается, модель подставляет медиану.
        velocities = []
        for w in ([] if historical else sci.recent):
            try:
                published = date.fromisoformat((w.get("publicationDate") or f"{w['year']}-07-01")[:10])
            except (KeyError, ValueError):
                continue
            velocities.append((w.get("citationCount") or 0) / _months_between(published, snapshot))
        f["citation_velocity_median"] = round(statistics.median(velocities), 6) if velocities else ""
        if sci.recent:
            preprints = sum(1 for w in sci.recent if w.get("preprint"))
            f["preprint_share_2y"] = round(preprints / len(sci.recent), 6)
        if sci.institutions_3y:
            f["institution_count_log_3y"] = _log(len(sci.institutions_3y))
        if sci.industry_affiliation_share_3y != "":
            f["industry_affiliation_share_3y"] = sci.industry_affiliation_share_3y
        if sci.countries_3y:
            f["science_country_count_log_3y"] = _log(len(sci.countries_3y))

    # ---- патенты
    if pat.ok:
        p0, p1, p2 = pat.windows
        f["patent_families_log_3y"] = _log(pat.count_3y)
        f["patent_growth_2y"] = _growth(p0, p1)
        f["patent_acceleration"] = round(_growth(p0, p1) - _growth(p1, p2), 6)
        f["patent_first_priority_age_years"] = _age(snapshot, pat.first_priority)
        assignee_counts = {name.lower(): count for name, count in pat.assignee_counts_3y.items() if count > 0}
        if assignee_counts:
            before = {a.lower() for a in pat.assignees_before}
            distinct = set(assignee_counts)
            f["patent_new_assignee_share_3y"] = round(len(distinct - before) / len(distinct), 6)
            f["patent_assignee_count_log_3y"] = _log(len(distinct))
            f["patent_assignee_hhi_3y"] = _hhi_counts(assignee_counts)
        elif pat.assignees_3y:
            before = {a.lower() for a in pat.assignees_before}
            distinct = {a.lower() for a in pat.assignees_3y}
            f["patent_new_assignee_share_3y"] = round(len(distinct - before) / len(distinct), 6)
            f["patent_assignee_count_log_3y"] = _log(len(distinct))
            f["patent_assignee_hhi_3y"] = _hhi(pat.assignees_3y)

    # ---- медиа
    news_12m = med.windows.get("0-6m", []) + med.windows.get("6-12m", []) if med.ok else []
    if med.ok:
        n0, n1 = len(med.windows["0-6m"]), len(med.windows["6-12m"])
        # Google News отдаёт до 100 записей на окно, поэтому объём нормирован на потолок 200.
        f["media_volume_normalized_12m"] = round(math.log1p(n0 + n1) / math.log1p(200), 6)
        f["media_growth_6m"] = _growth(n0, n1)
        monthly = Counter(h["date"][:7] for h in news_12m)
        months = []
        cursor = date.fromisoformat(cut[12])
        while cursor <= snapshot:
            months.append(monthly.get(cursor.isoformat()[:7], 0))
            cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
        if news_12m:
            mu, sd = statistics.mean(months), statistics.pstdev(months)
            f["media_burstiness_12m"] = round((sd - mu) / (sd + mu), 6) if sd + mu else ""
            hosts = [h["host"] or h["publisher"].lower() for h in news_12m]
            independent = {h for h in hosts if not registry.is_press_release(h)}
            f["independent_media_domains_log_12m"] = _log(len(independent))
            f["press_release_share_12m"] = round(sum(registry.is_press_release(h) for h in hosts) / len(hosts), 6)
            f["independent_high_trust_source_count_log"] = _log(len({h for h in hosts if registry.is_high_trust(h)}))
            f["unverifiable_source_share"] = round(sum(registry.is_unverifiable(h) for h in hosts) / len(hosts), 6)
            f["source_organization_hhi"] = _hhi(hosts)
        else:
            f["independent_media_domains_log_12m"] = 0.0
            f["independent_high_trust_source_count_log"] = 0.0

    # ---- события (LLM со ссылкой на заголовок; дата из заголовка)
    events = list((ext or {}).get("events") or [])
    structured_refs = structured_refs or []
    existing = {(e.get("type"), str(e.get("company") or "").casefold().strip(),
                 str(e.get("customer") or "").casefold().strip(), e.get("date")) for e in events}
    events.extend(e for e in _direct_structured_events(structured_refs, snapshot)
                  if (e.get("type"), str(e.get("company") or "").casefold().strip(),
                      str(e.get("customer") or "").casefold().strip(), e.get("date")) not in existing)
    have_events = bool(events) or (ext is not None and med.ok)

    def within(e, months):
        return e["date"] >= cut[months]

    if have_events:
        by_type = {t: [e for e in events if e["type"] == t] for t in EVENT_TYPES}
        commercial = [e["date"] for e in events if e["type"] in ("pilot", "deployment", "product", "funding")]
        f["first_commercial_event_age_years"] = _age(snapshot, min(commercial)) if commercial else ""
        f["verified_pilots_log_24m"] = _log(len({(e.get("company") or "", e.get("customer") or "", e["date"][:7])
                                                 for e in by_type["pilot"] if within(e, 24)}))
        f["production_deployments_log"] = _log(len({(e.get("company") or "", e.get("customer") or "", e["date"][:7])
                                                    for e in by_type["deployment"]}))
        vendors = {(e.get("company") or "").strip().lower() for e in events
                   if e["type"] in ("pilot", "deployment", "product", "funding") and e.get("company")}
        customers = {(e.get("customer") or "").strip().lower() for e in events if e.get("customer")}
        f["commercial_vendor_count_log"] = _log(len(vendors))
        f["named_customer_count_log"] = _log(len(customers))
        f["final_standard_flag"] = int(bool(by_type["standard_final"]))
        f["regulatory_precursor_count"] = len(by_type["regulation"])
        f["procurement_presence"] = int(bool(by_type["procurement"]))
        funding = by_type["funding"]
        f24 = [e for e in funding if within(e, 24)]
        f48 = [e for e in funding if not within(e, 24)]

        def amount(items):
            total = 0.0
            for e in items:
                try:
                    total += float(e.get("amount_usd_m") or 0)
                except (TypeError, ValueError):
                    pass
            return total
        f["funding_log_24m"] = _log(amount(f24))
        f["funding_growth_24m"] = round(math.log1p(amount(f24)) - math.log1p(amount(f48)), 6)
        f["funding_round_count_24m"] = len(f24)
        rounds = [str(e.get("round") or "").lower() for e in f24]
        known = [r for r in rounds if r in ("pre-seed", "seed", "a", "b", "c", "d+")]
        f["early_stage_funding_share"] = round(sum(r in ("pre-seed", "seed", "a") for r in known) / len(known), 6) \
            if known else ""
        f["strategic_investor_count"] = sum(1 for e in f24 if e.get("strategic") is True)
        f["funding_company_hhi"] = _hhi([e.get("company") or "" for e in f24])
        f["public_grants_log_36m"] = _log(len([e for e in by_type["grant"] if within(e, 36)]))
    if ext is not None:
        if ext.get("mass_market") is not None:
            f["mass_market_flag"] = int(bool(ext["mass_market"]))
        if ext.get("rebranding_similarity") is not None:
            f["rebranding_similarity_to_mature"] = ext["rebranding_similarity"]
    f["is_technology_flag"] = int(bool(query.get("is_technology", True)))

    # ---- текст (формулы как у build_positive_dataset: словари маркетинга и техники)
    text = " ".join(sci.titles + pat.titles + [h["title"] for h in news_12m])
    if text.strip():
        words = re.findall(r"[a-z][a-z0-9-]+", text.lower())
        f["marketing_claim_density"] = round(1000 * sum(w in MARKETING for w in words) / max(1, len(words)), 6)
        f["technical_term_density"] = round(sum(w in TECHNICAL for w in words) / max(1, len(words)), 6)
        docs = max(1, len(sci.titles) + len(pat.titles) + len(news_12m))
        hits = (len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|x|nm|µm|mm|w|kw|mw|kwh|wh|ms|ghz|mhz)\b", text.lower()))
                + len(re.findall(r"compared with|outperform|reduced|increased|efficiency|accuracy", text.lower())))
        f["claim_specificity"] = round(min(1.0, hits / docs), 6)

    # ---- первые свидетельства и межканальные признаки
    firsts = [d for d in (sci.first_date, pat.first_priority,
                          min((h["date"] for ws in med.windows.values() for h in ws), default=None))
              if d and d <= s]
    f["first_evidence_age_years"] = _age(snapshot, min(firsts)) if firsts else ""
    if sci.ok and pat.ok:
        f["patent_to_paper_ratio"] = round(math.log1p(pat.count_3y) - math.log1p(sci.count_3y), 6)
        f["patent_minus_paper_growth"] = round(f["patent_growth_2y"] - f["paper_growth_2y"], 6)
    if med.ok and (sci.ok or pat.ok):
        technical = (sci.count_3y if sci.ok else 0) + (pat.count_3y if pat.ok else 0)
        f["media_to_technical_evidence_ratio"] = round(math.log1p(len(news_12m)) - math.log1p(technical), 6)

    channels = {
        "science": sci.ok and sci.total > 0,
        "patents": pat.ok and sum(pat.windows) + pat.count_3y + pat.older_6y > 0,
        "media": med.ok and any(med.windows.values()),
        "funding": any(e["type"] in ("funding", "grant") for e in events),
        "adoption": (any(e["type"] in ("pilot", "deployment", "product", "procurement") for e in events)
                     or any(bool(ref.get("adoption_metrics")) for ref in structured_refs)),
    }
    f["source_type_diversity"] = sum(channels.values())
    growing = [sci.ok and sci.windows[0] >= 2 and sci.windows[0] > sci.windows[1],
               pat.ok and pat.windows[0] >= 2 and pat.windows[0] > pat.windows[1],
               med.ok and len(med.windows["0-6m"]) >= 2 and len(med.windows["0-6m"]) > len(med.windows["6-12m"]),
               have_events and any(e["type"] == "funding" and within(e, 24) for e in events)]
    f["cross_channel_confirmation_count"] = sum(growing)
    if sci.ok and pat.ok and med.ok:
        active = [sci.windows[0] > 0, pat.windows[0] > 0, len(news_12m) > 0]
        was_absent = [sci.windows[1] == 0, pat.windows[1] == 0,
                      not med.windows["12-24m"] and not med.windows["24-48m"]]
        f["first_time_convergence_flag"] = int(all(active) and sum(was_absent) >= 2)
    for name in ("science", "patents", "media", "funding", "adoption"):
        f[f"missing_{name}"] = int(not channels[name])
    return f


# -------------------------------------------------------------------------- конвейер

MONEY_EVENT_LIMIT = 12
MONEY_REVIEW_LIMIT = 40
MONEY_SEARCH_SECONDS = float(os.getenv("FEATURE_MONEY_SEARCH_SECONDS", "45"))

MONEY_PROMPT = """Кандидат в технологические сигналы: «{name}» (поисковый термин «{term}»), тема: «{domain}».

Ниже денежные события, найденные по поисковому термину: гранты, контракты, тендеры,
раунды, отчётность компаний. Поиск словарный, поэтому многие события про другое
(например, медицинский грант для банковской темы).

{events}

Оставь только события, которые прямо касаются ЭТОЙ технологии или её применения в теме:
финансируют её разработку, закупают или внедряют её, компания строит на ней продукт.
Совпадения отдельных слов недостаточно. Событие про тему в целом (например, раунд
компании, которая занимается безопасностью ИИ-агентов вообще) подходит, только если в
нём назван именно этот подход. Ответ — JSON: {{"keep": [номера событий]}}."""


def _snippet(ref: dict) -> str:
    """Начало описания без служебной шапки коннектора (у NIH и CORDIS она до пустой строки)."""
    text = str((ref.get("document") or {}).get("text") or "")
    _, sep, body = text.partition("\n\n")
    return " ".join((body if sep and body.strip() else text).split())[:240]


def _relevant_money(events: list[dict], refs: dict[str, dict], name: str, domain: str,
                    terms: list[str]) -> list[dict]:
    """Проверка LLM, что событие про кандидата; без ответа — строгий фильтр по заголовку."""
    if not events:
        return events
    lines = "\n".join(f'{i}. [{e["kind"]}] {e["title"]} — {_snippet(refs.get(e["url"]) or {})}'
                      for i, e in enumerate(events, 1))
    try:
        keep = llm_json(MONEY_PROMPT.format(name=name, term=terms[0], domain=domain or "не задана",
                                            events=lines)).get("keep")
        if not isinstance(keep, list):
            raise ValueError("нет списка keep")
        chosen = {int(i) for i in keep if str(i).strip().isdigit()}
        return [e for i, e in enumerate(events, 1) if i in chosen]
    except Exception:  # noqa: BLE001 — карточка не должна падать из-за проверки
        return [e for e in events if relevant(e["title"], terms, min_share=1.0)]


def money_events(refs: list[dict], snapshot: date, *, name: str = "", domain: str = "",
                 terms: list[str] | None = None) -> list[dict]:
    """Денежные и закупочные события кандидата для карточки: свежие первыми, не позже snapshot.

    С terms события проходят проверку релевантности (_relevant_money): словарный поиск
    API по короткому термину приносит гранты на чужие темы. Денежные документы, которые
    радар прочитал по веткам, приходят в refs вместе с ответами коннекторов.
    """
    events, seen = [], set()
    for ref in refs:
        raw = ref.get("structured_event")
        if not isinstance(raw, dict) or ref.get("channel") != "money" or not raw.get("date"):
            continue
        if str(raw["date"])[:10] > snapshot.isoformat():
            continue
        # Один грант приходит из нескольких API (NSF и USAspending): склеиваем по началу
        # заголовка, USAspending дописывает к названию NSF хвост описания.
        key = " ".join(re.findall(r"\w+", str(ref.get("title") or ref.get("url")).lower())[:6])
        if key in seen:
            continue
        seen.add(key)
        events.append({"kind": raw.get("kind") or raw.get("type"), "date": str(raw["date"])[:10],
                       "amount_usd_m": raw.get("amount_usd_m"), "company": raw.get("company") or "",
                       "customer": raw.get("customer"), "funder": raw.get("funder"),
                       "title": ref.get("title") or "", "url": ref.get("url"),
                       "connector": ref.get("connector") or ref.get("source_provider")})
    events.sort(key=lambda e: e["date"], reverse=True)
    if terms:
        events = _relevant_money(events[:MONEY_REVIEW_LIMIT], {r.get("url"): r for r in refs},
                                 name, domain, terms)
    return events[:MONEY_EVENT_LIMIT]


def featurize(name: str, domain: str, snapshot: date, query: dict | None = None,
              requested_features: list[str] | tuple[str, ...] | None = None,
              feature_connectors: frozenset[str] | set[str] | None = None,
              with_money_events: bool = False, money_refs: list[dict] | None = None,
              historical: bool | None = None) -> dict:
    """Разметка одной технологии; serving may request only the selected model inputs.

    feature_connectors — источники, по которым модель обучалась: ссылки остальных
    источников в признаки не попадают (другое распределение), но с with_money_events
    возвращаются в поле money_events для карточки. None — все включённые источники.
    money_refs — денежные документы запуска радара: только для карточки, не для признаков.
    historical — срез в прошлом (без источников без достоверной даты); по умолчанию
    snapshot < сегодня. False — для пересчёта среза, снятого вживую (обучающая выборка).
    """
    query = query or derive_query(name, domain)
    if not query["term"]:
        raise ValueError(f"не удалось получить поисковый термин для «{name}»")
    terms = [query["term"], *query["aliases"]]
    from app.radar.connectors import is_historical
    from app.radar.research_apis import search_structured_sources

    if historical is None:
        historical = is_historical(snapshot)

    needed = set(requested_features) if requested_features is not None else None
    # The deployed LR has no patent-specific inputs, and its training data has
    # missing_patents=1 throughout. Keep patent collection for offline dataset
    # building and future models, but don't make live scoring depend on FPO today.
    needs_patent_metrics = _needs_patent_metrics(needed)
    affiliation_features = {"institution_count_log_3y", "industry_affiliation_share_3y",
                            "science_country_count_log_3y"}
    include_affiliations = needed is None or bool(needed & affiliation_features)
    with ThreadPoolExecutor(5) as pool:
        sci_f = pool.submit(ev.science, terms, snapshot, include_affiliations=include_affiliations)
        pat_f = pool.submit(ev.patents, terms, snapshot) if needs_patent_metrics else None
        med_f = pool.submit(ev.media, terms, snapshot)
        # Science features already query OpenAlex inside ev.science(); avoid a
        # second independent OpenAlex lookup for the same technology.
        structured_f = pool.submit(search_structured_sources, terms, snapshot,
                                   per_source=3, include_openalex=False, historical=historical,
                                   only=frozenset(feature_connectors) if feature_connectors is not None else None)
        # Источники вне обучения модели нужны только карточке: ограничены по времени и
        # не задерживают признаки (иначе кандидат чаще уходит в медианную подстановку).
        money_f = (pool.submit(search_structured_sources, terms, snapshot, per_source=3,
                               include_openalex=False, historical=historical,
                               skip=frozenset(feature_connectors),
                               deadline=MONEY_SEARCH_SECONDS)
                   if with_money_events and feature_connectors is not None else None)
        sci = sci_f.result()
        pat = pat_f.result() if pat_f else ev.Patents(source="not_used_by_active_model")
        med, structured_refs = med_f.result(), structured_f.result()
        extra_refs = money_f.result() if money_f else []
    headlines = select_headlines(med) if med.ok else []
    with ThreadPoolExecutor(1) as pool:
        # Проверка денежных событий карточки — отдельный вызов LLM: идёт параллельно с
        # разметкой событий, иначе кандидат ждёт её последовательно (прогон 7 не уложился).
        events_f = (pool.submit(money_events, [*(money_refs or []), *structured_refs, *extra_refs],
                                snapshot, name=name, domain=domain, terms=terms)
                    if with_money_events else None)
        try:
            ext = extract_events(query, name, snapshot, headlines, sci, pat, structured_refs)
        except RuntimeError:
            ext = None
        card_events = events_f.result() if events_f else []
    features = compute(snapshot, query, sci, pat, med, ext, structured_refs, historical=historical)
    if needed is not None:
        features = {name: value for name, value in features.items() if name in needed}
    status = {"science": sci.ok, "patents": pat.ok, "media": med.ok, "llm": ext is not None}
    extra = {"money_events": card_events} if with_money_events else {}
    result = {
        **extra,
        "search_term": query["term"], "search_aliases": "; ".join(query["aliases"]),
        "snapshot_date": snapshot.isoformat(), "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "extractor_version": f"{EXTRACTOR_VERSION}:{MODEL}",
        "patent_source": pat.source,
        "patent_warning": pat.warning,
        "extraction_status": ",".join(k for k, ok in status.items() if not ok) or "ok",
        "llm_stage": (ext or {}).get("stage") or "",
        "evidence_digest": hashlib.sha256(json.dumps(
            [sci.windows, sci.count_3y, pat.windows, pat.count_3y,
             {k: len(v) for k, v in med.windows.items()},
             sorted(ref.get("url", "") for ref in structured_refs)], sort_keys=True).encode()).hexdigest()[:12],
        **features,
    }
    if requested_features is not None:
        result["visualization"] = {**_publication_visualization(sci),
                                    "evidence_timeline": _evidence_timeline_visualization(
                                        snapshot, sci, med, structured_refs)}
    return result
