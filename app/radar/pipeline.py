from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date

from app.radar.analysis import (
    checked_assessment,
    merge_candidates,
    rank_candidates,
    rank_results,
    unassessed,
)
from app.radar.config import RadarSettings
from app.radar.locales import clean_search_query, normalize_queries
from app.radar.model_scoring import ModelScorer, apply_model, supporting_sources
from app.radar.models import SearchRequest
from app.radar.open_sources import bing_news, branch_tasks, run_task
from app.radar.sources import (
    SourceFetcher,
    canonical_url,
    mark_duplicates,
    public_source,
)
from app.radar.spoken_media import DISCOVERY_TERMS, is_youtube_url
from app.radar.store import RadarStore, now
from app.radar.web_search import search_task

logger = logging.getLogger("idea.radar")
FINALIZATION_RESERVE_SECONDS = 60
STAGES = ["План поиска", "Поиск источников", "Чтение документов", "Выделение технологий",
          "Проверка зрелости", "Анализ доказательств", "Готово"]


def branch_coverage_warning(actual: int, requested: int) -> str | None:
    if actual >= requested:
        return None
    return (f"План сформировал {actual} из {requested} запрошенных веток. "
            "Исходная формулировка всё равно была включена в поиск.")


def select_search_locales(localized: list[dict], preferred_language: str, *, broad: bool = False) -> list[dict]:
    if broad:
        return list(localized)
    by_language = {item["language"]: item for item in localized if item.get("query")}
    selected = []
    for language in dict.fromkeys(["ru", "en", preferred_language]):
        if language in by_language and by_language[language] not in selected:
            selected.append(by_language[language])
    for item in localized:
        if len(selected) >= 3:
            break
        if item.get("query") and item not in selected:
            selected.append(item)
    return selected


def spoken_media_search_task(branch, localized: list[dict], preferred_language: str,
                             *, broad: bool = False) -> dict:
    """Search the branch for transcripts, talks, webinars and podcast episodes."""
    chosen = select_search_locales(localized, preferred_language, broad=broad)
    queries = [{"language": code,
                "query": f"{item['query']} {DISCOVERY_TERMS.get(code, DISCOVERY_TERMS['en'])}"}
               for item in chosen if (code := item["language"]) in DISCOVERY_TERMS or item.get("query")]
    if not queries:
        queries = [{"language": "en", "query": f"{branch.query_en} {DISCOVERY_TERMS['en']}"}]
    return {"branch": branch.topic, "query": "\n".join(item["query"] for item in queries),
            "queries": queries, "media_kind": "spoken",
            "priority_languages": [preferred_language],
            "max_queries": len(queries), "per_query": 2 if broad else 3,
            "max_refs": 28 if broad else 9,
            "connector_label": "Поиск докладов, подкастов и транскриптов"}


class SearchCancelled(Exception):
    pass


class _DeadlineThreadPoolExecutor(ThreadPoolExecutor):
    """Do not let ThreadPoolExecutor's context manager overrun the run deadline.

    ThreadPoolExecutor.__exit__ waits for every running future, even if the main
    pipeline has already hit its deadline. Upstream HTTP calls cannot always be
    interrupted once submitted, so abandon those futures and let the pipeline
    persist its partial result promptly.
    """

    def __init__(self, max_workers: int, deadline: float, cancel: threading.Event):
        super().__init__(max_workers=max_workers)
        self.deadline = deadline
        self.cancel = cancel

    def __exit__(self, exc_type, exc_value, traceback):
        aborted = (self.cancel.is_set() or time.monotonic() >= self.deadline
                   or (exc_type is not None and issubclass(exc_type, (TimeoutError, SearchCancelled))))
        if aborted:
            self.shutdown(wait=False, cancel_futures=True)
            return False
        return super().__exit__(exc_type, exc_value, traceback)


def new_run(request: SearchRequest, settings: RadarSettings) -> dict:
    stamp = now()
    return {"id": str(uuid.uuid4()), "query": request.query, "directions": request.directions,
            "as_of": request.as_of.isoformat(),
            "limit": request.limit, "created_at": stamp, "updated_at": stamp,
            "finished_at": None, "status": "queued", "stage": "В очереди", "stage_index": -1,
            "plan": None, "sources": [], "candidates": [], "signals": [], "top_candidates": [], "rejected": [],
            "assessments": [], "searches": [], "calls": [], "warnings": [], "events": [],
            "counters": {"found": 0, "fetched": 0, "read": 0, "duplicates": 0, "candidates": 0, "assessed": 0,
                         "confident": 0},
            "config": {"model": settings.active_model, "research_model": settings.active_research_model,
                       "provider": settings.provider_label, "branches": settings.branches, "max_sources": settings.max_sources,
                       "max_candidates": settings.max_candidates, "deadline_seconds": settings.deadline_seconds},
            "score_note": ("Скоринг — калиброванная вероятность обученной логистической регрессии "
                           "(app/ml/model/logreg_v1.json); проверка источников отсеивает нерелевантное "
                           "и неподтверждённое." if settings.use_model else
                           "Оценка по фиксированной рубрике; не калиброванная вероятность."),
            "mode": "live_search"}


def match_assessments(batch: list[dict], assessments: list) -> dict:
    """Оценки к кандидатам по id; нераспознанные — по порядку, если их столько же, сколько
    кандидатов без оценки. Открытые модели иногда искажают длинные id («cand_d1977acd71c8»)."""
    by_id = {a.candidate_id: a for a in assessments if a.candidate_id in {c["id"] for c in batch}}
    rest = [a for a in assessments if a.candidate_id not in by_id]
    missing = [c for c in batch if c["id"] not in by_id]
    if rest and len(rest) == len(missing):
        by_id.update({c["id"]: a for c, a in zip(missing, rest)})
    return by_id


def make_provider(settings: RadarSettings, deadline: float):
    if settings.provider == "local":
        from app.radar.provider_local import LocalProvider
        return LocalProvider(settings, deadline)
    from app.radar.provider_openrouter import OpenRouterProvider
    return OpenRouterProvider(settings, deadline)


class SearchPipeline:
    def __init__(self, store: RadarStore, settings: RadarSettings, provider_factory=make_provider, fetcher=None):
        self.store = store
        self.settings = settings
        self.provider_factory = provider_factory
        self.fetcher = fetcher or SourceFetcher(settings.storage_dir / "source_cache",
            timeout=settings.fetch_timeout, cache_hours=settings.cache_hours,
            audio_transcription_url=settings.audio_transcription_url,
            audio_transcription_token=settings.audio_transcription_token,
            audio_transcription_model=settings.audio_transcription_model,
            audio_transcription_allowlist=settings.audio_transcription_allowlist,
            audio_transcription_timeout=settings.audio_transcription_timeout,
            audio_transcription_max_mb=settings.audio_transcription_max_mb)

    def execute(self, run: dict, cancel: threading.Event):
        started = time.monotonic()
        deadline = started + self.settings.deadline_seconds
        finalization_reserve = min(FINALIZATION_RESERVE_SECONDS,
                                   max(15, int(self.settings.deadline_seconds * 0.06)))
        work_deadline = deadline - finalization_reserve
        provider = None
        scorer = None
        sources: list[dict] = []
        as_of = date.fromisoformat(run["as_of"])

        def check():
            if cancel.is_set():
                raise SearchCancelled()
            if time.monotonic() >= work_deadline:
                raise TimeoutError("Исчерпан общий бюджет времени")

        def save():
            if provider:
                run["calls"] = provider.audit()
            run["elapsed_seconds"] = round(time.monotonic() - started, 1)
            self.store.save(run)

        def stage(index, message=None):
            check()
            run.update(status="running", stage=STAGES[index], stage_index=index)
            run["events"].append({"at": now(), "stage": STAGES[index], "message": message or STAGES[index]})
            save()

        def completed(futures, label):
            """Yield finished tasks while persisting a heartbeat during slow upstream calls."""
            pending = set(futures)
            finished = 0
            while pending:
                check()
                timeout = min(30.0, max(0.0, work_deadline - time.monotonic()))
                done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                if not done:
                    check()
                    if run["events"]:
                        run["events"][-1].update(
                            at=now(),
                            message=f"{label}: выполнено {finished} из {len(futures)} задач; ожидаю завершения остальных.",
                        )
                    save()
                    continue
                for future in done:
                    finished += 1
                    yield future

        def parallel(items: list, action: Callable, workers: int):
            output = []
            with _DeadlineThreadPoolExecutor(max_workers=workers, deadline=work_deadline, cancel=cancel) as pool:
                futures = {pool.submit(action, item): i for i, item in enumerate(items)}
                for future in completed(futures, run["stage"]):
                    try:
                        output.append((futures[future], future.result()))
                    except (TimeoutError, SearchCancelled):
                        raise
                    except Exception as exc:
                        logger.warning("Stage item failed: %s", type(exc).__name__)
                        run["warnings"].append(f"{run['stage']}: часть операции не выполнена ({type(exc).__name__}).")
                    save()
            return [item for _, item in sorted(output)]

        def search(tasks, open_tasks=()):
            tasks = [{
                **task,
                "query": clean_search_query(task.get("query", "")),
                "queries": [{**query, "query": clean_search_query(query.get("query", ""), query.get("language", "und"))}
                            for query in task.get("queries", [])],
            } for task in tasks]
            if self.settings.web_search == "service":  # сервис поиска: SearXNG, Tavily, Exa, Brave, Яндекс…
                results = parallel(tasks, lambda t: search_task(t, run["as_of"]), self.settings.search_workers)
            elif self.settings.web_search == "llm" and getattr(provider, "web_search", True):
                results = parallel(tasks, lambda t: provider.search(t["query"], run["as_of"], t["branch"]),
                                   self.settings.search_workers)
            else:
                results = []
            if open_tasks and self.settings.open_sources:
                opened = parallel(list(open_tasks), lambda t: run_task(*t), 8)
                run["counters"]["open_sources"] = run["counters"].get("open_sources", 0) + sum(
                    len(r["references"]) for r in opened)
                results = results + opened  # пустые и ошибочные — тоже в аудит
            run["searches"].extend(results)
            refs, seen = [], set()
            # Round robin prevents the first branch from consuming the entire budget.
            for index in range(max((len(result["references"]) for result in results), default=0)):
                for result in results:
                    if index < len(result["references"]):
                        reference = result["references"][index]
                        try:
                            key = canonical_url(reference["url"])
                        except ValueError:
                            continue
                        if is_youtube_url(key):
                            reference["media_kind"] = "spoken"
                        if key not in seen:
                            refs.append(reference)
                            seen.add(key)
            # Reserve the first read slots for a root-topic result from each searched locale.
            root_topic = (run.get("plan") or {}).get("branches", [{}])[0].get("topic")
            root_by_language = {}
            for reference in refs:
                language = reference.get("query_language")
                branch_name = reference.get("branch", "")
                if (branch_name == root_topic or branch_name.startswith(f"{root_topic} ·")) and language:
                    root_by_language.setdefault(language, reference)
            reserved = list(root_by_language.values())
            reserved_ids = {reference["url"] for reference in reserved}
            refs = reserved + [reference for reference in refs if reference["url"] not in reserved_ids]
            run["counters"]["found"] = len({r["url"] for s in run["searches"] for r in s["references"]})
            save()
            return refs

        def fetch(refs, budget):
            known = {s["url"] for s in sources}
            refs = [r for r in refs if canonical_url(r["url"]) not in known][:budget]
            with _DeadlineThreadPoolExecutor(max_workers=6, deadline=work_deadline, cancel=cancel) as pool:
                futures = {pool.submit(self.fetcher.load, ref, as_of): i for i, ref in enumerate(refs)}
                for future in completed(futures, run["stage"]):
                    try:
                        source = future.result()
                    except Exception as exc:
                        run["warnings"].append(f"Источник не обработан ({type(exc).__name__}).")
                        continue
                    sources.append(source)
                    self.store.put_source(run["id"], source)
                    update_sources()
                    save()
            mark_duplicates(sources)
            for source in sources:
                self.store.put_source(run["id"], source)
            update_sources()
            save()

        def update_sources():
            run["sources"] = [public_source(s) for s in sources]
            run["counters"].update(fetched=len(sources), read=sum(s["status"] == "read" for s in sources),
                                   duplicates=sum(bool(s.get("duplicate_of")) for s in sources))

        def readable():
            return [s for s in sources if s["status"] == "read" and not s.get("duplicate_of")]

        def extract(documents):
            # Keep each structured response comfortably below the model token cap.
            batches = [documents[i:i+4] for i in range(0, len(documents), 4)]
            results = parallel(batches, lambda docs: provider.candidates(run["query"], run["as_of"], docs), self.settings.search_workers)
            allowed = {s["id"] for s in documents}
            candidates = []
            for batch in results:
                for c in batch.candidates:
                    item = c.model_dump()
                    item["source_ids"] = [sid for sid in item["source_ids"] if sid in allowed]
                    if item["source_ids"]:
                        candidates.append(item)
            return candidates

        def assess(candidates, verification_docs):
            source_map = {s["id"]: s for s in readable()}
            batches = [candidates[i:i+2] for i in range(0, len(candidates), 2)]
            pending_model: dict[str, tuple[dict, list[dict]]] = {}

            def publish_assessments():
                run["signals"], run["rejected"] = rank_results(run["assessments"], run["limit"])
                run["top_candidates"] = rank_candidates(run["assessments"], run["limit"])
                run["counters"]["confident"] = sum(
                    1 for a in run["signals"] if a.get("score_kind") == "model" and a.get("score", 0) > 75)
                run["counters"]["assessed"] = len(run["assessments"])
                save()

            def score_if_ready(item):
                if not scorer:
                    return apply_model(item, None, supporting_sources(item, source_map)) if self.settings.use_model else item
                source_items = supporting_sources(item, source_map)
                model_result = scorer.result(item["id"], 0.1)
                future = scorer.futures.get(item["id"])
                if model_result is None and future is not None:
                    if not future.done():
                        pending_model[item["id"]] = (item, source_items)
                        return apply_model(item, None, source_items)
                    model_result = scorer.result(item["id"], 0.1)
                if model_result is None:
                    model_result = scorer.fallback_result(
                        item["id"], "Полный сбор признаков завершился ошибкой; применена медианная подстановка.")
                return apply_model(item, model_result, source_items)

            def replace_with_model_result(candidate_id: str, item: dict, model_result: dict | None,
                                          source_items: list[dict]):
                if model_result is None:
                    model_result = scorer.fallback_result(
                        candidate_id, "Полный сбор признаков не уложился в бюджет запуска; применена медианная подстановка.")
                final = apply_model(item, model_result, source_items)
                final["id"] = f"{run['id']}_{candidate_id}"
                for index, existing in enumerate(run["assessments"]):
                    if existing["id"] == final["id"]:
                        run["assessments"][index] = final
                        return

            def one_batch(batch):
                # Relevant retrieval for audit context; always include each original source.
                terms = set(" ".join(c["title"] + " " + c["verification_query"] for c in batch).lower().split())
                extra = sorted(verification_docs, key=lambda s: sum(t in s["text"].lower() for t in terms if len(t) > 3), reverse=True)
                docs, doc_ids = [], set()
                trust_order = {"high": 3, "medium": 2, "low": 1, "unverified": 0}
                for candidate in batch:
                    own = [source_map[sid] for sid in candidate["source_ids"] if sid in source_map]
                    own.sort(key=lambda source: (trust_order.get(source.get("trust", "unverified"), 0),
                                                 source.get("published_at") or ""), reverse=True)
                    for source in own[:4]:
                        if source["id"] not in doc_ids:
                            docs.append(source)
                            doc_ids.add(source["id"])
                for source in extra:
                    if len(docs) >= 10:
                        break
                    if source["id"] not in doc_ids:
                        docs.append(source)
                        doc_ids.add(source["id"])
                return batch, provider.assess(run["query"], run["as_of"], batch, docs)

            # Publish actual checked batches as they finish.
            with _DeadlineThreadPoolExecutor(max_workers=self.settings.search_workers, deadline=work_deadline,
                                            cancel=cancel) as pool:
                futures = {pool.submit(one_batch, batch): batch for batch in batches}
                for future in completed(futures, run["stage"]):
                    batch = futures[future]
                    try:
                        _, result = future.result()
                        by_id = match_assessments(batch, result.assessments)
                        checked = [checked_assessment(c, by_id[c["id"]], source_map, as_of)
                                   if c["id"] in by_id else unassessed(c, "Модель не вернула оценку кандидата.") for c in batch]
                    except Exception as exc:
                        checked = [unassessed(c, f"Проверка не завершена ({type(exc).__name__}).") for c in batch]
                    if scorer:
                        checked = [score_if_ready(item) for item in checked]
                    for item in checked:
                        item["id"] = f"{run['id']}_{item['id']}"
                    run["assessments"].extend(checked)
                    publish_assessments()

            if pending_model:
                total_pending = len(pending_model)
                # Feature extraction involves several independent APIs and two LLM
                # calls. Wait for ready scores as a group, with heartbeats, rather
                # than blocking on one candidate for the rest of the run budget.
                # Every candidate must receive a model score before the run is
                # complete. Poll together so one slow feature job cannot block
                # LLM assessments or stop heartbeat updates; reserve time to save.
                scoring_deadline = work_deadline
                while pending_model and time.monotonic() < scoring_deadline:
                    check()
                    ready = [candidate_id for candidate_id in pending_model
                             if (future := scorer.futures.get(candidate_id)) is None or future.done()]
                    if not ready:
                        pending_futures = [scorer.futures[candidate_id] for candidate_id in pending_model
                                           if candidate_id in scorer.futures]
                        wait(pending_futures, timeout=min(5, max(0.1, scoring_deadline-time.monotonic())),
                             return_when=FIRST_COMPLETED)
                        ready = [candidate_id for candidate_id in pending_model
                                 if (future := scorer.futures.get(candidate_id)) is None or future.done()]
                    for candidate_id in ready:
                        item, source_items = pending_model.pop(candidate_id)
                        result = scorer.result(candidate_id, 0.1)
                        replace_with_model_result(candidate_id, item, result, source_items)
                    if run["events"]:
                        run["events"][-1].update(
                            at=now(),
                            message=(f"Сбор признаков и применение логистической регрессии: "
                                     f"{total_pending-len(pending_model)} из {total_pending}."),
                        )
                    publish_assessments()

                if pending_model:
                    fallback_count = len(pending_model)
                    for candidate_id, (item, source_items) in list(pending_model.items()):
                        replace_with_model_result(candidate_id, item, None, source_items)
                    pending_model.clear()
                    if run["events"]:
                        run["events"][-1].update(
                            at=now(),
                            message=(f"Логистическая регрессия рассчитана для всех кандидатов; "
                                     f"для {fallback_count} применена предварительная оценка "
                                     "с медианной подстановкой признаков."),
                        )
                    publish_assessments()

        try:
            provider = self.provider_factory(self.settings, work_deadline)
            stage(0)
            plan = provider.plan(run["query"], run["as_of"], run.get("directions", []))
            run["plan"] = plan.model_dump()
            branches = plan.branches[:self.settings.branches]
            if not branches:
                raise ValueError("Модель не сформировала план поиска")
            plan_warning = branch_coverage_warning(len(branches), self.settings.branches)
            if plan_warning:
                run["warnings"].append(plan_warning)
            paraphrase_fallbacks = getattr(provider, "plan_paraphrase_fallback_count", 0)
            if paraphrase_fallbacks:
                run["warnings"].append(
                    f"Для заполнения всех веток добавлено {paraphrase_fallbacks} поисковых вариантов "
                    "на основе исходной темы и выбранных направлений; новые подтемы не добавлялись.")
            stage(1)
            localized_by_branch = [
                normalize_queries(b.query_ru, b.query_en, [q.model_dump() for q in b.localized_queries])
                for b in branches
            ]
            search_locales_by_branch = [
                select_search_locales(localized, plan.query_language, broad=index == 0)
                for index, localized in enumerate(localized_by_branch)
            ]
            open_by_branch = [
                [(name, action, b.topic, query, language) for name, action, query, language in branch_tasks(
                    b.topic, b.query_ru, b.query_en, as_of, branch_locales)]
                for b, branch_locales in zip(branches, search_locales_by_branch)
            ]
            all_open_tasks = [task for tasks_for_branch in open_by_branch for task in tasks_for_branch]
            # These connectors use the unchanged top-level English query for every
            # branch. Running the same GDELT/arXiv request once per branch only
            # causes duplicate network load and public-API throttling.
            singleton_connectors = {"GDELT Global News", "arXiv (EN)"}
            seen_singletons = set()
            deduped_tasks = []
            for task in all_open_tasks:
                name, _, _, query, _ = task
                key = (name, query)
                if name in singleton_connectors and key in seen_singletons:
                    continue
                if name in singleton_connectors:
                    seen_singletons.add(key)
                deduped_tasks.append(task)
            all_open_tasks = deduped_tasks
            selected_open_ids = set()
            open_tasks = []
            # Reserve early fetch slots for each locale in both media and science,
            # instead of letting the first language fill the 64-document budget.
            for connector in ("Bing News", "Crossref"):
                for language_index, localized in enumerate(localized_by_branch[0]):
                    language = localized["language"]
                    choices = [task for task in all_open_tasks
                               if task[2] == branches[0].topic and task[4] == language
                               and task[0].startswith(connector) and id(task) not in selected_open_ids]
                    if not choices:
                        choices = [task for task in all_open_tasks
                                   if task[4] == language and task[0].startswith(connector)
                                   and id(task) not in selected_open_ids]
                    if choices:
                        picked = choices[language_index % len(choices)]
                        open_tasks.append(picked)
                        selected_open_ids.add(id(picked))
            # Structured research/public-sector evidence gets explicit fetch slots;
            # otherwise the much larger news and Crossref fan-out can crowd it out.
            structured_prefixes = ("Patents (", "OpenAlex", "NIH RePORTER", "CORDIS grants",
                                   "GitHub repositories", "Hugging Face", "TED procurement",
                                   "ClinicalTrials.gov", "FDA openFDA")
            for task in all_open_tasks:
                if task[0].startswith(structured_prefixes) and id(task) not in selected_open_ids:
                    open_tasks.append(task)
                    selected_open_ids.add(id(task))
            open_tasks.extend(task for task in all_open_tasks if id(task) not in selected_open_ids)
            branch_search_tasks = [
                {"query": "\n".join(f"{q['language_name']}: {q['query']}" for q in localized_queries),
                 "queries": [{"language": q["language"], "query": q["query"]} for q in localized_queries],
                 "branch": b.topic,
                 "max_queries": 14 if index == 0 else 3,
                 "priority_languages": [plan.query_language],
                 "per_query": 2 if index == 0 else 3,
                 "max_refs": 28 if index == 0 else 9}
                for index, (b, localized_queries) in enumerate(zip(branches, search_locales_by_branch))
            ]
            spoken_tasks = [spoken_media_search_task(b, localized, plan.query_language, broad=index == 0)
                            for index, (b, localized) in enumerate(zip(branches, search_locales_by_branch))]
            refs = search(branch_search_tasks + spoken_tasks, list(open_tasks))
            if not refs:
                raise ValueError("Поиск не вернул источников")
            stage(2)
            fetch(refs, max(8, int(self.settings.max_sources * .65)))
            if not readable():
                raise ValueError("Не удалось прочитать источники. Проверьте сеть или измените запрос.")
            stage(3)
            candidates = merge_candidates(extract(readable()), 48)
            if not candidates:
                raise ValueError("В прочитанных источниках не выделены кандидаты")
            try:
                consolidated = provider.consolidate(run["query"], candidates)
                if getattr(provider, "consolidation_chunked", False):
                    run["warnings"].append(
                        "Смысловое объединение выполнено пакетами; между пакетами проверены дубли названий.")
                allowed_ids = {s["id"] for s in readable()}
                cleaned = []
                for c in consolidated.candidates:
                    item = c.model_dump()
                    item["source_ids"] = [sid for sid in c.source_ids if sid in allowed_ids]
                    if item["source_ids"]:
                        cleaned.append(item)
                if cleaned:
                    candidates = merge_candidates(cleaned, self.settings.max_candidates)
            except Exception:
                run["warnings"].append("Смысловое объединение не завершено; применена консервативная дедупликация названий.")
            candidates = candidates[:self.settings.max_candidates]
            run["candidates"] = candidates
            run["counters"]["candidates"] = len(candidates)
            if self.settings.use_model:
                try:
                    scorer = ModelScorer(self.settings, as_of)
                    scorer.submit(candidates)  # размечаются параллельно с проверочным поиском
                except Exception as exc:  # noqa: BLE001 — без модели работает проверка источников
                    logger.warning("Model unavailable: %s", type(exc).__name__)
                    run["warnings"].append("Модель недоступна; решения приняты по критериям проверки источников.")
            stage(4)
            verify_tasks = [{"branch": "Проверка стадии и внедрений", "query": "\n".join(
                c["verification_query"] for c in candidates[i:i+6])} for i in range(0, len(candidates), 6)]
            previous_ids = {s["id"] for s in sources}
            verify_refs = search(verify_tasks, [
                ("Bing News (EN)", (lambda q: lambda: bing_news(q, "en", 4))(
                    clean_search_query(c["verification_query"], "en")),
                 "Проверка стадии и внедрений", clean_search_query(c["verification_query"], "en"))
                for c in candidates])
            fetch(verify_refs, max(0, self.settings.max_sources - len(sources)))
            verification_docs = [s for s in readable() if s["id"] not in previous_ids]
            stage(5)
            assess(candidates, verification_docs)
            check()
            shortfall = run["limit"] - len(run["top_candidates"])
            if shortfall > 0 and work_deadline - time.monotonic() > 80 and len(sources) < self.settings.max_sources:
                run["events"].append({"at": now(), "stage": "Дополнительный поиск",
                                      "message": f"Нужно проверить ещё {shortfall} кандидатов для финального ТОП."})
                old_ids = {s["id"] for s in sources}
                refs = search([{"branch": "Дополнительное покрытие", "query":
                    f"{clean_search_query(run['query'])}: новые исследования и ранние пилоты в ещё не покрытых применениях. "
                    f"Уже рассмотрены: {', '.join(c['title'] for c in candidates)[:1800]}",
                    # поисковому API — короткие запросы, а не инструкция для LLM
                    "queries": [{"query": f"{clean_search_query(run['query'])} ранние пилоты новые исследования", "language": "ru"},
                                {"query": f"{clean_search_query(run['query'])} early pilots emerging research", "language": "en"}]}])
                fetch(refs, self.settings.max_sources - len(sources))
                new_docs = [s for s in readable() if s["id"] not in old_ids]
                if new_docs:
                    old_candidates = {c["id"] for c in candidates}
                    additions = [c for c in merge_candidates(candidates + extract(new_docs), 40) if c["id"] not in old_candidates][:8]
                    run["candidates"].extend(additions)
                    run["counters"]["candidates"] = len(run["candidates"])
                    if scorer:
                        scorer.submit(additions)
                    assess(additions, verification_docs)
            unassessed_count = max(0, len(run["candidates"]) - len(run["assessments"]))
            unscored_count = (sum(1 for item in run["assessments"] if item.get("model") is None)
                              if self.settings.use_model else 0)
            partial_score_count = sum(1 for item in run["assessments"]
                                      if (item.get("model") or {}).get("fallback"))
            checks_complete = unassessed_count == 0 and unscored_count == 0
            run["status"] = ("completed" if len(run["top_candidates"]) >= run["limit"] and checks_complete
                             else "partial")
            run.update(stage="Готово" if run["status"] == "completed" else "Проверка завершена не полностью",
                       stage_index=6)
            if unassessed_count:
                run["warnings"].append(
                    f"Полная проверка не завершилась для {unassessed_count} из {len(run['candidates'])} кандидатов.")
            if unscored_count:
                run["warnings"].append(
                    f"Логистическая оценка не завершилась для {unscored_count} кандидатов; "
                    "они не засчитаны как слабые сигналы.")
            if partial_score_count:
                run["warnings"].append(
                    f"Логистическая регрессия рассчитала оценки для всех кандидатов; "
                    f"у {partial_score_count} оценка предварительная из-за неполного сбора признаков. "
                    "Такие оценки не подтверждают статус слабого сигнала.")
            if run["status"] == "partial":
                if len(run["top_candidates"]) < run["limit"]:
                    run["warnings"].append(
                        f"Найдено и проверено {len(run['top_candidates'])} из {run['limit']} кандидатов. "
                        "Недостающие результаты не добавлены без доказательств.")
            elif len(run["signals"]) < len(run["top_candidates"]):
                run["events"].append({"at": now(), "stage": "Готово",
                    "message": f"В ТОП вошло {len(run['top_candidates'])} кандидатов; "
                               f"слабым сигналом подтверждены {len(run['signals'])}. У остальных сохранён свой статус."})
        except SearchCancelled:
            run.update(status="cancelled", stage="Остановлено пользователем")
        except TimeoutError:
            run.update(status="partial" if run["top_candidates"] else "failed", stage="Лимит времени")
            run["warnings"].append("Исчерпан лимит времени. Проверенные результаты сохранены.")
        except Exception as exc:
            logger.exception("Radar run %s failed", run["id"])
            run.update(status="failed", stage="Не удалось завершить поиск")
            safe_message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            run["warnings"].append(safe_message[:300])
        finally:
            run["finished_at"] = now()
            save()
            if provider:
                provider.close()
            if scorer:
                scorer.close()


class SearchManager:
    def __init__(self, store: RadarStore, settings: RadarSettings, pipeline_factory=SearchPipeline):
        self.store, self.settings = store, settings
        self.pipeline_factory = pipeline_factory
        self.executor = ThreadPoolExecutor(max_workers=settings.max_active_runs, thread_name_prefix="radar")
        self.slots = threading.BoundedSemaphore(settings.max_active_runs)
        self.lock = threading.Lock()
        self.active: dict[str, threading.Event] = {}

    def start(self, request: SearchRequest, settings: RadarSettings) -> dict:
        if not self.slots.acquire(blocking=False):
            raise RuntimeError("Поиск уже выполняется. Дождитесь завершения или остановите текущий запуск.")
        run = new_run(request, settings)
        event = threading.Event()
        try:
            self.store.save(run)
            with self.lock:
                self.active[run["id"]] = event
            self.executor.submit(self._execute, run, event, settings)
        except Exception:
            self.slots.release()
            with self.lock:
                self.active.pop(run["id"], None)
            raise
        return {"id": run["id"], "status": "queued"}

    def _execute(self, run, event, settings):
        try:
            self.pipeline_factory(self.store, settings).execute(run, event)
        finally:
            with self.lock:
                self.active.pop(run["id"], None)
            self.slots.release()

    def cancel(self, run_id):
        with self.lock:
            event = self.active.get(run_id)
            if event:
                event.set()
        return bool(event)

    def close(self):
        with self.lock:
            for event in self.active.values():
                event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
