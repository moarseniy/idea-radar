"""Проверка локального запуска: та же открытая модель на ноутбуке (квантизация) против облака.

Сценарий развёртывания без внешнего API: модель слабых сигналов обучена на признаках,
которые размечены полными весами открытой LLM (compare_feature_models.py, облако), а
в сервисе признаки размечает квантизованная версия той же LLM на локальном сервере
(Ollama, vLLM, LM Studio — любой OpenAI-совместимый). Скрипт размечает локально только
валидацию (примеры заказчика + негативы) и сравнивает:
  метрики модели на облачных и на локальных признаках — сколько съедает квантизация;
  совпадение стадии, «технология или нет» и числа событий между облаком и ноутбуком;
  время на строку (два вызова LLM) — годится ли это для радара.

Выход: ml/reports/local_model_check_<модель>.{json,md}.
Запуск: python ml/scripts/local_model_check.py --local qwen3.6:35b-a3b \\
          --cloud "qwen/qwen3.6-35b-a3b#nothink" [--url http://localhost:11434/v1]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_feature_models as cmp
from export_model import fit, metrics

import app.ml.features.extract as ex
from app.ml.scorer import Scorer


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True, help="имя модели на локальном сервере")
    parser.add_argument("--cloud", required=True, help="та же модель в сравнении, например qwen/qwen3.6-35b-a3b#nothink")
    parser.add_argument("--url", default="http://localhost:11434/v1")
    parser.add_argument("--extra", default='{"reasoning_effort": "none"}', help="поля запроса: режим без рассуждений")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    folder = cmp.OUT / cmp.slug(args.cloud)
    name = args.local.replace(":", "-").replace("/", "-")
    training, gold, negatives = (read_csv(folder / f"{n}.csv") for n in ("training", "gold", "validation"))
    for g in gold:
        g["label"] = "1"
    cloud_rows = gold + negatives
    artifact = fit(training, cloud_rows, str(folder.relative_to(cmp.bts.ROOT) / "training.csv"))
    scorer = Scorer(artifact)

    rows = cloud_rows[: args.limit] if args.limit else cloud_rows
    ex.LLM_URL, ex.MODEL, ex.LLM_EXTRA = args.url.rstrip("/"), args.local, json.loads(args.extra)
    print(f"локально: {args.local} @ {args.url}; строк {len(rows)}; модель обучена на признаках {args.cloud}")
    evidences = cmp.bts.parallel(cmp.evidence, rows, 8, "данные")
    started = time.monotonic()
    reads = cmp.bts.parallel(lambda pair: cmp.read_row(*pair), list(zip(rows, evidences)), args.workers, "локально")
    wall = time.monotonic() - started
    ok = [(r, x) for r, x in zip(rows, reads) if not isinstance(x, Exception)]
    errors = [str(x)[:120] for x in reads if isinstance(x, Exception)]

    y = [int(r["label"]) for r, _ in ok]
    p_cloud = [scorer.score(r)["probability"] for r, _ in ok]
    local_rows = [{**r, **x["features"]} for r, x in ok]
    p_local = [scorer.score(r)["probability"] for r in local_rows]
    thr = artifact["threshold"]
    y_arr = np.array(y)
    stages = ["stage_research", "stage_poc", "stage_pilot", "stage_early_adoption", "stage_scaling_or_mature"]
    same_stage = [all(str(r[f]) == str(x["features"][f]) for f in stages) for r, x in ok]
    same_tech = [r["is_technology_flag"] == str(x["features"]["is_technology_flag"]) for r, x in ok]
    same_decision = [(a >= thr) == (b >= thr) for a, b in zip(p_cloud, p_local)]
    seconds = [x["seconds"] for _, x in ok]
    report = {
        "local": args.local, "cloud": args.cloud, "url": args.url, "extra": json.loads(args.extra),
        "rows": len(ok), "errors": len(errors), "error_examples": errors[:5],
        "threshold": thr,
        "validation_cloud_features": metrics(y_arr, np.array(p_cloud), thr),
        "validation_local_features": metrics(y_arr, np.array(p_local), thr),
        "agreement": {"decision": round(sum(same_decision) / len(ok), 3),
                      "stage": round(sum(same_stage) / len(ok), 3),
                      "is_technology": round(sum(same_tech) / len(ok), 3),
                      "mean_abs_probability_diff": round(float(np.mean(np.abs(np.array(p_cloud) - np.array(p_local)))), 3)},
        "seconds_per_row": {"median": round(statistics.median(seconds), 1),
                            "p90": round(sorted(seconds)[int(0.9 * (len(seconds) - 1))], 1)},
        "wall_seconds": round(wall), "workers": args.workers,
    }
    cmp.REPORTS.joinpath(f"local_model_check_{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    c, loc, a = report["validation_cloud_features"], report["validation_local_features"], report["agreement"]
    lines = [f"# Локальный запуск: `{args.local}` против `{args.cloud}`", "",
             f"Модель слабых сигналов обучена на признаках от `{args.cloud}` (облако, полные веса). "
             f"Валидация ({len(ok)} строк) размечена второй раз локально — `{args.local}` через {args.url}.", "",
             "| Признаки валидации | P | R | F1 | ROC-AUC |", "|---|---:|---:|---:|---:|",
             f"| облако | {c['precision']} | {c['recall']} | {c['f1']} | {c['roc_auc']} |",
             f"| ноутбук | {loc['precision']} | {loc['recall']} | {loc['f1']} | {loc['roc_auc']} |", "",
             f"Совпадение с облаком: решение модели — {a['decision']:.0%}, стадия — {a['stage']:.0%}, "
             f"«технология или нет» — {a['is_technology']:.0%}; средняя разница вероятностей {a['mean_abs_probability_diff']}.",
             f"Время на строку (2 вызова LLM, {args.workers} параллельно): медиана {report['seconds_per_row']['median']} с, "
             f"p90 {report['seconds_per_row']['p90']} с. Ошибок: {len(errors)}."]
    cmp.REPORTS.joinpath(f"local_model_check_{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
