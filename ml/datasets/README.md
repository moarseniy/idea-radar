# Датасеты

Разметка автоматическая: аннотаторов у команды нет. Как собраны строки и почему им можно
доверять — [../../docs/08-dataset.md](../../docs/08-dataset.md).

## Итоговые файлы

| Файл | Что это |
|---|---|
| `training_merged.csv` | **Обучающая выборка**: наши строки + строки Арсения после дедупликации, 63 признака схемы `weak-signals-63-v2` |
| `training_merged.report.json` | Состав, заполненность признаков, константные признаки, причины удаления |
| `gold_features.csv` | 100 записей заказчика с теми же 63 признаками. **Только валидация** |
| `merge_removed.csv` | Что удалено при объединении и почему (дубль технологии, противоречие меток, совпадение с золотом) |

Колонки `training_merged.csv`:
- служебные: `id, source (ours/arseniy), source_file, label, name_ru, name_original, domain, negative_type, stage_ref, trend_ref, technology_group_id, evidence_urls`;
- разметка: `search_term, search_aliases, snapshot_date, feature_schema_version, extractor_version, extraction_status, llm_stage, evidence_digest`;
- 63 признака в порядке `app/ml/features/schema.py`.

`stage_ref`/`trend_ref` — метки из сбора, **в признаки не входят**. `technology_group_id` — группа для
`StratifiedGroupKFold`.

## Наша часть (до объединения)

| Файл | Что это |
|---|---|
| `gold.csv` | 100 записей заказчика, нормализованные (стадия 1–5, тренд 1–3). Не редактировать |
| `gold_sources_reference.csv` | Колонки «Источники» и «Почему это слабый сигнал». **В признаки не включать** |
| `labeled.csv` | 400 позитивов + 395 негативов, зеркальный баланс по областям |
| `reserve.csv` | Строки, не вошедшие в квоты (перебор по области) |
| `dedupe_removed.csv`, `stage_rejected.csv`, `collect_rejected.csv` | Отсеянное и причины |
| `exclusions/` | Названия и термины Арсения и золота — сбор их не повторяет |
| `manifest.md` | Журнал решений и находок по ходу сбора |

## Воспроизведение

```bash
python ml/scripts/build_training_set.py --workers 24   # термины → дедупликация → признаки
python -m pytest tests/test_features_extract.py
```

Сетевые и LLM-ответы кэшируются в `ml/.cache/features/` (в git не попадает); повторный
запуск без сети даёт тот же файл.
