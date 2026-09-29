"""Финальная модель: обучение на training_merged.csv и экспорт в JSON для сервиса.

Модель — логистическая регрессия на стандартизованных признаках с медианной импутацией.
C выбирается групповой CV (по technology_group_id) только на обучении: самый сильно
регуляризованный в пределах 0,005 AUC от лучшего. Затем
вероятность калибруется по Платту на out-of-fold предсказаниях — тоже только на обучении:
p = sigmoid(a · z + b), где z — логит модели. Калибровка монотонна и линейна по z, поэтому
вклады признаков остаются точными: вклад_i = a · w_i · (x_i − μ_i) / σ_i.

Вес классов сбалансирован, поэтому вероятность отвечает априорной доле 50/50 и
не зависит от того, что негативов в обучении в 2,4 раза больше.

Экспорт — JSON, без pickle: медианы, средние, масштабы, веса, калибровка, порог, метрики.
Сервис читает его чистым Python (app/ml/scorer.py), sklearn на рантайме не нужен.

Запуск: python ml/scripts/export_model.py [--data ml/datasets/model_compare/gpt-5.6-luna]
Выход:  app/ml/model/logreg_v1.json, ml/reports/model_card.md
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.ml.features.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION  # noqa: E402
from app.ml.scorer import Scorer  # noqa: E402
from train_eval import C_GRID, SEED, matrix, model, read  # noqa: E402

OUT = ROOT / "app" / "ml" / "model" / "logreg_v1.json"
CARD = ROOT / "ml" / "reports" / "model_card.md"

# Признаки, исключённые из модели по смыслу (в схеме и датасете остаются).
EXCLUDED = {
    # LLM ставит 0 самим зрелым технологиям («это оригинал, не ребрендинг») и умеренную
    # похожесть новым — вес выходит положительным, что противоречит названию признака.
    "rebranding_similarity_to_mature": "направление противоречит смыслу: у зрелых технологий LLM ставит 0",
    # Ускорение = рост сейчас − рост раньше; в паре с ростом получает отрицательный вес,
    # и в объяснении «замедление» выглядит доводом за сигнал. Без него CV AUC 0,9407 против 0,9422.
    "paper_acceleration": "в паре с ростом публикаций получает вес обратного знака; CV AUC без него −0,0015",
    "patent_acceleration": "то же, что paper_acceleration (патентный канал сейчас не измеряется)",
}


def metrics(y, p, thr) -> dict:
    pred = p >= thr
    return {"precision": round(precision_score(y, pred, zero_division=0), 3),
            "recall": round(recall_score(y, pred), 3), "f1": round(f1_score(y, pred), 3),
            "roc_auc": round(roc_auc_score(y, p), 3), "pr_auc": round(average_precision_score(y, p), 3),
            "brier": round(brier_score_loss(y, p), 3)}


def fit(training: list[dict], validation: list[dict], training_data: str) -> dict:
    """Обучение, калибровка, порог и метрики валидации. Возвращает артефакт модели."""
    features = [f for f in FEATURE_NAMES if f not in EXCLUDED and len({r[f] for r in training}) > 1]
    x = matrix(training, features)
    y = np.array([int(r["label"]) for r in training])
    groups = [r["technology_group_id"] for r in training]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)

    auc = {}
    for c in C_GRID:
        auc[c] = roc_auc_score(y, cross_val_predict(model(c), x, y, groups=groups, cv=cv, method="predict_proba")[:, 1])
    # Самый сильно регуляризованный C в пределах 0,005 AUC от лучшего: качество то же,
    # а веса коррелирующих признаков не расходятся в разные стороны и читаются по смыслу.
    best_c = min(c for c in C_GRID if auc[c] >= max(auc.values()) - 0.005)

    # Out-of-fold логиты → калибровка Платта (веса классов сбалансированы, как у модели).
    logits = cross_val_predict(model(best_c), x, y, groups=groups, cv=cv, method="decision_function")
    weights = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))
    platt = LogisticRegression(C=1e6).fit(logits.reshape(-1, 1), y, sample_weight=weights)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    p_oof = 1 / (1 + np.exp(-(a * logits + b)))
    grid = np.linspace(0.2, 0.8, 61)
    threshold = float(grid[np.argmax([f1_score(y, p_oof >= t, sample_weight=weights) for t in grid])])

    final = model(best_c).fit(x, y)
    imputer, scaler, logreg = final[0], final[1], final[2]
    artifact = {
        "name": "logreg_v1", "trained_at": date.today().isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "training_data": training_data,
        "training_rows": len(training), "training_labels": {"1": int(y.sum()), "0": int(len(y) - y.sum())},
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True, cwd=ROOT).stdout.strip(),
        "C": best_c, "class_weight": "balanced",
        "features": features, "excluded_features": EXCLUDED,
        "median": [float(v) for v in imputer.statistics_],
        "mean": [float(v) for v in scaler.mean_], "scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in logreg.coef_[0]], "intercept": float(logreg.intercept_[0]),
        "calibration": {"method": "platt_oof", "a": a, "b": b},
        "threshold": threshold,
        "cv": metrics(y, p_oof, threshold) | {"auc_by_C": {str(k): round(v, 4) for k, v in auc.items()}},
    }
    # Валидация — через тот же Scorer, что в сервисе: проверяет и модель, и экспорт.
    scorer = Scorer(json.loads(json.dumps(artifact)))
    yv = np.array([int(r["label"]) for r in validation])
    pv = np.array([scorer.score(r)["probability"] for r in validation])
    sk = 1 / (1 + np.exp(-(a * final.decision_function(matrix(validation, features)) + b)))
    assert np.allclose(pv, sk, atol=1e-9), "экспорт расходится со sklearn"
    artifact["validation"] = metrics(yv, pv, threshold) | {
        "positives": int(yv.sum()), "negatives": int(len(yv) - yv.sum()),
        "confident_positive_share_gt_0.75": round(float(np.mean(pv[yv == 1] > 0.75)), 3)}
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="папка сравнения моделей (training.csv, gold.csv, validation.csv) "
                                       "вместо базовой сборки, например ml/datasets/model_compare/gpt-5.6-luna")
    parser.add_argument("--out", help="путь артефакта вместо app/ml/model/logreg_v1.json (карточка модели "
                                      "тогда не перезаписывается), например app/ml/model/logreg_v1_qwen3.6-35b-a3b.json")
    args = parser.parse_args()
    out = ROOT / args.out if args.out else OUT
    if args.data:
        folder = ROOT / args.data
        training, gold, negatives = (list(csv.DictReader((folder / f"{n}.csv").open(encoding="utf-8")))
                                     for n in ("training", "gold", "validation"))
        source = f"{args.data}/training.csv"
    else:
        training, gold, negatives = read("training_merged.csv"), read("gold_features.csv"), read("validation_negatives.csv")
        source = "ml/datasets/training_merged.csv"
    for g in gold:
        g["label"] = "1"
    validation = gold + negatives
    artifact = fit(training, validation, source)
    features, threshold, best_c, y = artifact["features"], artifact["threshold"], artifact["C"], artifact["training_labels"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.out:
        print(f"{args.out}: валидация {artifact['validation']}")
        return 0

    order = np.argsort(-np.abs(artifact["coef"]))
    lines = ["# Карточка модели logreg_v1", "",
             f"Логистическая регрессия, {len(features)} признаков схемы `{FEATURE_SCHEMA_VERSION}`, "
             f"C={best_c}, веса классов сбалансированы, калибровка Платта на out-of-fold, порог {threshold:.2f}.",
             f"Обучение: `{source}`, {len(training)} строк ({y['1']} позитивов). "
             "C, калибровка и порог выбраны только на обучении (StratifiedGroupKFold по технологии).", "",
             "| | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, m in (("CV на обучении", artifact["cv"]), ("Валидация: золото + негативы", artifact["validation"])):
        lines.append(f"| {label} | {m['precision']} | {m['recall']} | {m['f1']} | {m['roc_auc']} | {m['pr_auc']} | {m['brier']} |")
    lines += ["", f"Валидация: {artifact['validation']['positives']} позитивов золота, "
              f"{artifact['validation']['negatives']} негативов (`validation_negatives.csv`). "
              f"Доля позитивов золота с уверенностью > 0,75: {artifact['validation']['confident_positive_share_gt_0.75']}.",
              "", "## Исключённые признаки", ""]
    lines += [f"- `{f}` — {why}" for f, why in EXCLUDED.items()]
    lines += ["- константные на обучении (не измеряются): " +
              ", ".join(f"`{f}`" for f in FEATURE_NAMES if f not in EXCLUDED and f not in features)]
    lines += ["", "## Веса (на стандартизованных признаках, до калибровки)", "", "| Признак | Вес |", "|---|---:|"]
    lines += [f"| `{features[i]}` | {artifact['coef'][i]:+.3f} |" for i in order if abs(artifact["coef"][i]) >= 1e-4]
    CARD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
