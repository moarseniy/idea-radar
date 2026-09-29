"""Сравнение LLM для разметки признаков на фиксированных данных.

Меняется только модель, которая «читает»: признак is_technology_flag (derive_query),
события, стадия и массовость (extract_events). Поисковый термин, статьи Crossref и
заголовки новостей берутся из базовой сборки (training_merged.csv и др.) и одинаковы
для всех моделей. Так разница в метриках — это разница в чтении, а не в том, что
поисковик нашёл по другому термину. Строки обучения и валидации тоже одинаковы.

Для каждой модели: признаки → обучение и валидация тем же кодом, что export_model.fit →
метрики; плюс совпадение стадии LLM со стадией заказчика в золоте и доля сбоев.
Базовая модель (gemini) должна воспроизвести базовые признаки из кэша — это проверка
самого сравнения.

Выход: ml/datasets/model_compare/<модель>/*.csv, ml/datasets/model_compare/extractions.jsonl
(вход для judge_extractions.py), ml/reports/model_compare.{json,md}.
Запуск: python ml/scripts/compare_feature_models.py [--models a,b] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_training_set as bts  # noqa: E402  (грузит .env и настройки источников)

import app.ml.features.evidence as ev  # noqa: E402
import app.ml.features.extract as ex  # noqa: E402
from app.ml.features.schema import FEATURE_NAMES  # noqa: E402
from export_model import fit  # noqa: E402

BASELINE = "google/gemini-3.8-flash"
# Облачные из списка ТЗ, затем открытые модели для локального запуска (#nothink — без рассуждений).
MODELS = [BASELINE, "openai/gpt-5.6-luna", "openai/gpt-4.1", "qwen/qwen3.6-35b-a3b",
          "qwen/qwen3-235b-a22b-2507", "qwen/qwen3.8-27b#nothink", "qwen/qwen3.6-35b-a3b#nothink",
          "nvidia/nemotron-3.5-lightning#nothink", "google/gemma-4-26b-a4b-it", "google/gemma-4-31b-it"]
OUT = bts.DATASETS / "model_compare"
REPORTS = bts.ROOT / "ml" / "reports"


NOTHINK = "#nothink"  # суффикс модели: режим без рассуждений (как для запуска на ноутбуке)


def slug(model: str) -> str:
    return model.split("/")[-1].replace(NOTHINK, "-nothink")


def use(model: str) -> None:
    """Переключает разметчик на модель; "имя#nothink" — без рассуждений."""
    ex.MODEL = model.removesuffix(NOTHINK)
    ex.LLM_EXTRA = {"reasoning": {"enabled": False}} if model.endswith(NOTHINK) else {}


def load() -> dict[str, list[dict]]:
    gold = bts.read(bts.DATASETS / "gold_features.csv")
    for g in gold:
        g["label"] = "1"
    return {"training": bts.read(bts.DATASETS / "training_merged.csv"), "gold": gold,
            "validation": bts.read(bts.DATASETS / "validation_negatives.csv")}


def base_query(row: dict) -> dict:
    return {"term": row["search_term"], "aliases": [a for a in row["search_aliases"].split("; ") if a],
            "is_technology": row["is_technology_flag"] == "1"}


def evidence(row: dict) -> dict:
    """Статьи и новости по базовому термину — из кэша базовой сборки."""
    q = base_query(row)
    terms = [q["term"], *q["aliases"]]
    sci, pat, med = ev.science(terms, bts.SNAPSHOT), ev.patents(terms, bts.SNAPSHOT), ev.media(terms, bts.SNAPSHOT)
    return {"sci": sci, "pat": pat, "med": med, "headlines": ex.select_headlines(med) if med.ok else []}


def read_row(row: dict, evd: dict) -> dict:
    """Всё, что делает LLM для строки, — под текущей ex.MODEL."""
    started = time.monotonic()
    own = ex.derive_query(row["name_ru"], row["domain"])
    query = {**base_query(row), "is_technology": own["is_technology"]}
    try:
        ext = ex.extract_events(query, row["name_ru"], bts.SNAPSHOT, evd["headlines"], evd["sci"], evd["pat"])
    except RuntimeError:
        ext = None
    features = ex.compute(bts.SNAPSHOT, query, evd["sci"], evd["pat"], evd["med"], ext)
    features["paper_volume_percentile_domain"] = row["paper_volume_percentile_domain"]  # наука та же
    return {"features": features, "ext": ext, "own_query": own, "seconds": time.monotonic() - started}


def same(a, b) -> bool:
    if a in ("", None) or b in ("", None):
        return a in ("", None) and b in ("", None)
    return abs(float(a) - float(b)) < 1e-6


def stage_agreement(rows: list[dict], reads: list[dict]) -> dict:
    pairs = [(int(r["stage_ref"]), x["ext"]["stage"]) for r, x in zip(rows, reads)
             if r.get("stage_ref", "").isdigit() and x["ext"] and x["ext"]["stage"]]
    return {"n": len(pairs), "exact": round(sum(a == b for a, b in pairs) / max(1, len(pairs)), 3),
            "within_1": round(sum(abs(a - b) <= 1 for a, b in pairs) / max(1, len(pairs)), 3)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="для отладки: первые N строк каждой части")
    args = parser.parse_args()
    models = args.models.split(",")

    parts = load()
    if args.limit:
        parts = {k: v[: args.limit] for k, v in parts.items()}
    rows = [r for part in parts.values() for r in part]
    print(f"строк: {len(rows)}; данные по базовым терминам из кэша")
    evidences = bts.parallel(evidence, rows, args.workers, "данные")
    bad = [r["id"] for r, e in zip(rows, evidences) if isinstance(e, Exception)]
    if bad:
        raise RuntimeError(f"нет данных в кэше для {len(bad)} строк: {bad[:5]}")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "headlines.jsonl").open("w", encoding="utf-8") as fh:
        for r, e in zip(rows, evidences):
            fh.write(json.dumps({"id": r["id"], "name": r["name_ru"], "domain": r["domain"], "label": r["label"], "source": r["source"],
                                 "stage_ref": r.get("stage_ref", ""), "term": r["search_term"],
                                 "headlines": e["headlines"], "papers": e["sci"].titles[:15]},
                                ensure_ascii=False) + "\n")

    summary, extractions = {}, []
    for model in models:
        use(model)
        started = time.monotonic()
        reads = bts.parallel(lambda pair: read_row(*pair), list(zip(rows, evidences)), args.workers, slug(model))
        wall = time.monotonic() - started
        failed = [r["id"] for r, x in zip(rows, reads) if isinstance(x, Exception)]
        if failed:
            print(f"{model}: {len(failed)} строк без ответа модели — строки исключаются из сравнения у всех")
        for r, x in zip(rows, reads):
            if isinstance(x, Exception):
                x = {"features": None, "ext": None, "own_query": None, "seconds": 0}
            r.setdefault("_reads", {})[model] = x

        summary[model] = {"wall_seconds": round(wall), "failed_rows": len(failed)}

    # Одинаковый набор строк для всех моделей: исключаем строки, где хоть одна модель не ответила.
    usable = {id(r) for r in rows if all(r["_reads"][m]["features"] is not None for m in models)}
    for model in models:
        out_parts = {}
        (OUT / slug(model)).mkdir(parents=True, exist_ok=True)
        for name, part in parts.items():
            out_parts[name] = [{**{k: v for k, v in r.items() if not k.startswith("_")},
                                **r["_reads"][model]["features"]} for r in part if id(r) in usable]
            bts.write(OUT / slug(model) / f"{name}.csv", out_parts[name], bts.META + bts.SERVICE + FEATURE_NAMES)
        artifact = fit(out_parts["training"], out_parts["gold"] + out_parts["validation"],
                       f"ml/datasets/model_compare/{slug(model)}/training.csv")
        reads = [r["_reads"][model] for r in rows if id(r) in usable]
        kept = [r for r in rows if id(r) in usable]
        gold_rows = [(r, x) for r, x in zip(kept, reads) if r["source"] == "gold"]
        summary[model] |= {
            "rows": len(kept), "C": artifact["C"], "threshold": artifact["threshold"],
            "features": len(artifact["features"]),
            "cv": {k: artifact["cv"][k] for k in ("precision", "recall", "f1", "roc_auc")},
            "validation": {k: artifact["validation"][k] for k in ("precision", "recall", "f1", "roc_auc", "pr_auc", "brier")},
            "stage_vs_customer_gold": stage_agreement([r for r, _ in gold_rows], [x for _, x in gold_rows]),
            "no_extraction": sum(x["ext"] is None for x in reads),
            "events_per_row": round(sum(len(x["ext"]["events"]) for x in reads if x["ext"]) / len(reads), 2),
            "is_technology_share": {lab: round(sum(x["own_query"]["is_technology"] for r, x in zip(kept, reads) if r["label"] == lab)
                                               / max(1, sum(r["label"] == lab for r in kept)), 3) for lab in ("1", "0")},
            "stage_hist": dict(sorted(Counter(x["ext"]["stage"] for x in reads if x["ext"]).items(), key=str)),
        }
        if model == BASELINE:  # воспроизводимость: те же признаки, что в базовой сборке
            diff = Counter(f for r, x in zip(kept, reads) for f in FEATURE_NAMES if not same(r.get(f), x["features"].get(f)))
            summary[model]["baseline_feature_mismatches"] = dict(diff.most_common(8))
        for r, x in zip(kept, reads):
            extractions.append({"id": r["id"], "model": model, "own_query": x["own_query"],
                                "stage": x["ext"] and x["ext"]["stage"], "mass_market": x["ext"] and x["ext"]["mass_market"],
                                "events": x["ext"] and x["ext"]["events"]})
        print(model, json.dumps({k: summary[model][k] for k in ("validation", "stage_vs_customer_gold")}, ensure_ascii=False))

    with (OUT / "extractions.jsonl").open("w", encoding="utf-8") as fh:
        for e in extractions:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    (REPORTS / "model_compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Сравнение LLM для разметки признаков", "",
             "Данные одинаковы для всех моделей (термины, статьи, заголовки базовой сборки); меняется только LLM: "
             "«технология или нет», события, стадия, массовость. Обучение и валидация — `export_model.fit`.", "",
             "| Модель | Строк | CV F1 | CV AUC | Вал. P | Вал. R | Вал. F1 | Вал. AUC | Стадия = заказчик | ±1 | Без ответа |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, s in summary.items():
        v, c, st = s["validation"], s["cv"], s["stage_vs_customer_gold"]
        lines.append(f"| `{model}` | {s['rows']} | {c['f1']} | {c['roc_auc']} | {v['precision']} | {v['recall']} | "
                     f"{v['f1']} | {v['roc_auc']} | {st['exact']} | {st['within_1']} | {s['no_extraction']} |")
    (REPORTS / "model_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
