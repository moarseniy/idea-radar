"""Сборка объединённого обучающего датасета: наши строки + строки Арсения, 63 признака.

Шаги:
  1. Приводит обе выборки к общей схеме (id, source, label, name_ru, domain, ...).
  2. Для каждой строки и для золота получает канонический поисковый термин
     (app.ml.features.extract.derive_query) — одинаково для всех источников строк.
  3. Дедупликация по нормализованному термину: одна технология — одна строка.
     Группы с противоречащими метками удаляются целиком. Строки, чей термин совпадает
     с термином из золотой валидации, удаляются из обучения (утечка в валидацию).
  4. Разметка 63 признаками (app.ml.features.extract.featurize) на срез SNAPSHOT.
  5. paper_volume_percentile_domain — перцентиль papers_log_3y внутри области по
     обучающей выборке; опорное распределение сохраняется для сервиса.

Выход:
  ml/datasets/training_merged.csv       — обучающая выборка с признаками;
  ml/datasets/gold_features.csv         — 100 строк золота с теми же признаками;
  ml/datasets/merge_removed.csv         — что удалено при объединении и почему;
  ml/datasets/training_merged.report.json;
  app/ml/features/paper_volume_reference.json.

Запуск: python -m ml.scripts.build_training_set [--workers 12] [--limit N]
Повторный запуск берёт всё из кэша (ml/.cache/features) и даёт тот же файл.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_dotenv() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, _, value = line.partition("=")
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


load_dotenv()
# Для датасета полнота важнее скорости: источник не отключается после серии отказов,
# а ждёт и повторяет (до ~10 минут на запрос).
os.environ.setdefault("FEATURE_BREAKER_THRESHOLD", "1000000")
os.environ.setdefault("FEATURE_FETCH_TRIES", "14")

from app.ml.features.extract import derive_query, featurize  # noqa: E402
from app.ml.features.schema import FEATURE_NAMES  # noqa: E402

SNAPSHOT = date(2026, 9, 22)  # тот же срез, что у выборки Арсения
DATASETS = ROOT / "ml" / "datasets"
ARSENIY = ROOT / "data" / "training"
ARSENIY_FILES = ["positive_weak_signals_global.csv", "negative_weak_signals_global.csv",
                 "negative_weak_signals_additional_300.csv"]
REFERENCE = ROOT / "app" / "ml" / "features" / "paper_volume_reference.json"

META = ["id", "source", "source_file", "label", "name_ru", "name_original", "domain",
        "negative_type", "stage_ref", "trend_ref", "technology_group_id", "evidence_urls"]
SERVICE = ["search_term", "search_aliases", "snapshot_date", "feature_schema_version",
           "extractor_version", "extraction_status", "llm_stage", "evidence_digest"]


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def load_rows() -> list[dict]:
    rows = []
    for r in read(DATASETS / "labeled.csv"):
        rows.append({"id": r["id"], "source": "ours", "source_file": "ml/datasets/labeled.csv",
                     "label": r["label"], "name_ru": r["name"], "name_original": r["name"],
                     "domain": r["domain"], "negative_type": r["negative_type"],
                     "stage_ref": r["stage"], "trend_ref": r["trend"], "technology_group_id": "",
                     "evidence_urls": r["evidence_urls"]})
    for name in ARSENIY_FILES:
        for r in read(ARSENIY / name):
            urls = " ".join(u for u in (r.get("primary_source_url"), r.get("media_source_url"),
                                        r.get("patent_source_url")) if u)
            rows.append({"id": r["id"], "source": "arseniy", "source_file": f"data/training/{name}",
                         "label": r["label"], "name_ru": r["name_ru"], "name_original": r["name_original"],
                         "domain": r["domain_ru"], "negative_type": r.get("negative_type", ""),
                         "stage_ref": r.get("stage", ""), "trend_ref": r.get("trend", ""),
                         "technology_group_id": r["technology_group_id"], "evidence_urls": urls})
    return rows


def norm_term(term: str) -> str:
    words = re.findall(r"[a-z0-9]+", term.lower())
    return " ".join(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w for w in words)


def parallel(fn, items, workers, label):
    out, done, lock = [None] * len(items), 0, threading.Lock()
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                out[i] = fut.result()
            except Exception as exc:  # noqa: BLE001 — строка уходит в отчёт, прогон продолжается
                out[i] = exc
            with lock:
                done += 1
                if done % 50 == 0 or done == len(items):
                    print(f"  {label}: {done}/{len(items)}", flush=True)
    return out


def dedupe(rows: list[dict], gold_keys: set[str]) -> tuple[list[dict], list[dict]]:
    removed, groups = [], defaultdict(list)
    for r in rows:
        groups[norm_term(r["search_term"])].append(r)
    # Термин одной строки совпадает с синонимом другой — это та же технология.
    alias_owner = {}
    for key, members in groups.items():
        for r in members:
            for alias in filter(None, r["search_aliases"].split("; ")):
                a = norm_term(alias)
                if len(a) >= 6 and a != key:
                    alias_owner.setdefault(a, key)
    merged = defaultdict(list)
    for key, members in groups.items():
        merged[alias_owner.get(key, key) if alias_owner.get(key) in groups else key].extend(members)

    kept = []
    for key, members in sorted(merged.items()):
        if key in gold_keys:
            removed += [{**m, "reason": "термин совпадает с золотой валидацией"} for m in members]
            continue
        labels = {m["label"] for m in members}
        if len(labels) > 1:
            removed += [{**m, "reason": f"противоречие меток в группе «{key}»"} for m in members]
            continue
        members.sort(key=lambda m: (m["source"] != "ours", m["id"]))  # наши строки первыми
        kept.append(members[0])
        removed += [{**m, "reason": f"дубль технологии «{key}» → {members[0]['id']}"} for m in members[1:]]
    return kept, removed


def add_percentiles(rows: list[dict], reference: dict | None = None) -> dict:
    if reference is None:
        by_domain = defaultdict(list)
        for r in rows:
            if r.get("papers_log_3y") not in ("", None):
                by_domain[r["domain"]].append(float(r["papers_log_3y"]))
        reference = {d: sorted(v) for d, v in by_domain.items()}
        reference["__all__"] = sorted(x for v in by_domain.values() for x in v)
    for r in rows:
        value = r.get("papers_log_3y")
        if value in ("", None):
            continue
        ref = reference.get(r["domain"]) if len(reference.get(r["domain"], [])) >= 20 else reference["__all__"]
        r["paper_volume_percentile_domain"] = round(bisect.bisect_right(ref, float(value)) / len(ref), 6)
    return reference


def write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="для отладки: первые N строк")
    args = parser.parse_args()

    rows = load_rows()
    gold = read(DATASETS / "gold.csv")
    for g in gold:
        g.update({"source": "gold", "source_file": "ml/datasets/gold.csv", "label": "1",
                  "name_ru": g["name"], "name_original": g["name"], "negative_type": "",
                  "stage_ref": g.get("stage", ""), "trend_ref": g.get("trend", ""),
                  "technology_group_id": "", "evidence_urls": ""})
    if args.limit:  # отладка: детерминированная смешанная подвыборка
        rows = sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[: args.limit]
        gold = gold[: min(args.limit, len(gold))]
    print(f"строк: {len(rows)} ({Counter(r['source'] + ':' + r['label'] for r in rows)}), золото: {len(gold)}")

    print("1. поисковые термины")
    queries = parallel(lambda r: derive_query(r["name_ru"], r["domain"]), rows + gold, args.workers, "термины")
    failed = []
    for r, q in zip(rows + gold, queries):
        r.setdefault("search_aliases", "")
        if isinstance(q, Exception) or not q["term"]:
            failed.append(r)
            r["search_term"] = ""
            continue
        r["search_term"], r["search_aliases"], r["_query"] = q["term"], "; ".join(q["aliases"]), q
    gold_keys = {norm_term(g["search_term"]) for g in gold if g.get("search_term")}

    rows_ok = [r for r in rows if r["search_term"]]
    kept, removed = dedupe(rows_ok, gold_keys)
    removed += [{**r, "reason": "не получен поисковый термин"} for r in rows if not r["search_term"]]
    print(f"2. дедупликация: осталось {len(kept)}, удалено {len(removed)} "
          f"({Counter(x['reason'].split(' «')[0].split(' →')[0] for x in removed)})")

    print("3. признаки")
    def one(r):
        return featurize(r["name_ru"], r["domain"], SNAPSHOT, query=r["_query"])
    # Порядок перемешан детерминированно: если источник откажет посреди прогона, пропуски
    # не совпадут с классом или происхождением строк (иначе пропуск стал бы утечкой метки).
    targets = sorted(kept + [g for g in gold if g.get("search_term")],
                     key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    results = parallel(one, targets, args.workers, "признаки")
    errors = 0
    for r, res in zip(targets, results):
        if isinstance(res, Exception):
            errors += 1
            r["extraction_status"] = f"error: {str(res)[:120]}"
            continue
        r.update(res)
    for r in kept:
        if not r["technology_group_id"]:
            r["technology_group_id"] = "tech_" + hashlib.sha256(norm_term(r["search_term"]).encode()).hexdigest()[:12]

    training = [r for r in kept if not str(r.get("extraction_status", "")).startswith("error")]
    removed += [{**r, "reason": "ошибка разметки: " + r["extraction_status"]} for r in kept
                if str(r.get("extraction_status", "")).startswith("error")]
    reference = add_percentiles(training)
    if not args.limit:
        REFERENCE.write_text(json.dumps(reference, ensure_ascii=False), encoding="utf-8")
    add_percentiles(gold, reference)

    suffix = "" if not args.limit else ".debug"
    write(DATASETS / f"training_merged{suffix}.csv", training, META + SERVICE + FEATURE_NAMES)
    write(DATASETS / f"gold_features{suffix}.csv", gold, META + SERVICE + FEATURE_NAMES)
    write(DATASETS / f"merge_removed{suffix}.csv", removed, ["id", "source", "label", "name_ru", "search_term", "reason"])

    fill = {name: sum(1 for r in training if r.get(name) not in ("", None)) for name in FEATURE_NAMES}
    report = {
        "snapshot": SNAPSHOT.isoformat(), "rows": len(training),
        "by_source_label": Counter(f"{r['source']}:{r['label']}" for r in training),
        "by_domain_label": Counter(f"{r['domain']}:{r['label']}" for r in training),
        "negative_types": Counter(r["negative_type"] for r in training if r["label"] == "0"),
        "removed": Counter(x["reason"].split(" «")[0].split(" →")[0].split(":")[0] for x in removed),
        "feature_errors": errors, "query_failures": len(failed),
        "query_failure_examples": [f"{r['id']}: {r['name_ru'][:60]}" for r in failed[:10]],
        "extraction_status": Counter(r.get("extraction_status", "") for r in training),
        "feature_fill": fill,
        "constant_features": [n for n in FEATURE_NAMES if len({r.get(n) for r in training}) <= 1],
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True, cwd=ROOT).stdout.strip(),
    }
    (DATASETS / f"training_merged{suffix}.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "feature_fill"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
