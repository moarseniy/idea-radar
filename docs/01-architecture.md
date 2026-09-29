# Архитектура решения «Слабые сигналы»

Сервис автоматизированного сбора и анализа зарождающихся технологических трендов.
Кодовое имя: **Weak Signal Radar (WSR)**.

---

## 1. Принципы, заданные ТЗ

| Принцип | Как выражен в архитектуре |
|---|---|
| Чёткое разделение парсинга, инференса и интерфейса | Три независимых слоя: `ingest/` → `pipeline/ml/` → `frontend/`. Связь только через БД и HTTP API |
| Выдача не может опираться только на знания LLM | Каждое утверждение в отчёте привязано к `raw_document.id` из фактически загруженного корпуса. Валидатор цитат отклоняет ответ модели с URL вне корпуса |
| Прозрачность выбора модели | Слой `llm/` с реестром провайдеров; на каждый вызов пишется запись в `llm_calls` (провайдер, модель, стадия, токены, латентность) |
| Отказоустойчивость парсеров | Коннектор = изолированный компонент с таймаутом, ретраями, circuit breaker; падение одного не валит прогон |
| Хранение сырых данных | PostgreSQL: `raw_documents` хранит исходный текст и метаданные до всякой обработки |
| Воспроизводимость | Всё привязано к `run_id`; прогон можно переиграть из кэша сырых данных без обращения к сети |

---

## 2. Контекст системы

```
                 ┌──────────────────────────────────────────────┐
   Аналитик ────►│  React SPA (ru)                              │
   / жюри        │  Поиск · ТОП-15 · Отчёт · Источники ·        │
                 │  Карточка модели · Оценка на датасете        │
                 └───────────────┬──────────────────────────────┘
                                 │ REST + SSE
                 ┌───────────────▼──────────────────────────────┐
                 │  FastAPI backend                             │
                 │  api/ · pipeline/ · ml/ · llm/ · ingest/     │
                 └───┬───────────────┬──────────────┬───────────┘
                     │               │              │
          ┌──────────▼────┐   ┌──────▼──────┐  ┌────▼─────────┐
          │ PostgreSQL 16 │   │ Открытые    │  │ LLM-провайдер│
          │ сырьё+резуль- │   │ источники   │  │ GigaChat /   │
          │ таты+метрики  │   │ (коннекторы)│  │ Qwen / GPT   │
          └───────────────┘   └─────────────┘  └──────────────┘
```

Диаграммы в исходниках: `docs/diagrams/*.puml`.

---

## 3. Структура репозитория (целевая)

```
backend/
  app/
    main.py                    # сборка FastAPI, роутеры, middleware
    core/
      config.py                # Settings (pydantic-settings), .env
      logging.py               # структурные JSON-логи с run_id
      db.py                    # SQLAlchemy engine/session
    api/
      routes_search.py         # открытый запрос + SSE-прогресс
      routes_candidates.py     # карточки, отчёты, источники
      routes_datasets.py       # импорт датасета жюри, оценка, предсказания
      routes_model.py          # карточка модели, веса признаков, метрики
      routes_feedback.py       # human-in-the-loop
      routes_health.py
    models/                    # SQLAlchemy ORM
    schemas/                   # pydantic DTO
    ingest/
      base.py                  # Connector ABC, RawDoc, Retry, CircuitBreaker
      connectors/
        openalex.py  arxiv.py  crossref.py  europepmc.py
        patents_epo.py  patentsview.py
        cyberleninka.py  rospatent.py
        github.py  hackernews.py
        rss_media.py           # конфиг-лента: ru + en отраслевые медиа
        regulators.py          # NIST/ENISA/EC/ЦБ/Минцифры
      files/                   # перенос текущего document_parser.py (PDF/DOCX/XLSX/OCR)
      normalize.py             # дедуп, язык, даты, чистка
    pipeline/
      orchestrator.py          # оркестрация стадий, run-трекинг, SSE-события
      stages/
        s1_expand_query.py  s2_harvest.py  s3_normalize.py
        s4_extract_candidates.py  s5_cluster.py  s6_enrich.py
        s7_featurize.py  s8_classify.py  s9_gate.py
        s10_rank.py  s11_report.py  s12_translate.py
    ml/
      features/
        schema.py              # единый реестр признаков (имя, блок, ru-подпись, направление)
        from_row.py            # признаки из строки датасета (режим «датасет»)
        from_evidence.py       # признаки из собранного корпуса (режим «открытый поиск»)
        lexicons.py            # стадийная, трендовая, маркетинговая лексика
      model.py                 # обучение/инференс/калибровка
      explain.py               # вклады признаков → «ключевые предикторы»
      gates.py                 # жёсткие правила исключения + причины
      train.py  evaluate.py    # CLI: обучение и отчёт о метриках
      artifacts/               # app/ml/model/logreg_v1.json, ml/reports/model_card.md
    llm/
      base.py                  # LLMProvider ABC, строгий JSON-режим, ретраи
      providers/               # gigachat.py yandex.py qwen_local.py openai.py
      router.py                # выбор провайдера по стадии + логирование вызовов
      citations.py             # валидатор: все URL из ответа ∈ корпус прогона
    services/
      trust.py                 # реестр доверенности доменов A/B/C/D
      translate.py             # ru-резюме зарубежных источников + пометка «авто»
      report.py                # сборка документа-отчёта
frontend/                      # React 18 + TypeScript + Vite
  src/
    pages/  components/  api/  hooks/  lib/
    components/graph/          # обёртка над текущим graph-engine.js
ml/
  datasets/                    # positives.csv, negatives.csv, manifest.md
  reports/                     # metrics.md, confusion.png, calibration.png, ablation.md
docs/
  01-architecture.md  02-methodology.md  03-plan.md  04-tz-traceability.md
  diagrams/*.puml
docker/
  Dockerfile.backend  Dockerfile.frontend  nginx.conf
docker-compose.yml             # db + backend + frontend
```

### Что переносится из текущего кода

| Сейчас | Куда | Комментарий |
|---|---|---|
| `app/document_parser.py` (614 стр.) | `backend/app/ingest/files/` | Почти без изменений. Парсинг PDF/DOCX/XLSX + OCR + vision нужен для аналитических отчётов и импорта датасета жюри |
| `app/static/graph-engine.js` (747 стр.) | `frontend/src/components/graph/` | Обернуть в React-компонент, сменить типы узлов на technology/company/source |
| Экспорт MD/PDF из `main.py:500-690` | `backend/app/services/report.py` | Пригодится для выгрузки отчёта по инсайту |
| Ретраи и JSON-парсинг в `openrouter_service.py` | `backend/app/llm/base.py` | Логика `_extract_json`, backoff — переиспользуем |
| Docker/compose/healthcheck | `docker/` | Расширить сервисом Postgres и фронтом |

### Что удаляется

`knowledge.py` (доменные термины обогащения), `schemas.py` (проектные DTO), доменные промпты `openrouter_service.py`, таблицы `hypotheses`/`projects`, весь UI гипотез. Это другой кейс.

---

## 4. Пайплайн открытого запроса

```
Запрос пользователя («слабые сигналы в кибербезопасности»)
  │
  ├─ S1  Расширение запроса (LLM): ru+en термины, синонимы, смежные области,
  │      отрицательные термины. Логируется модель и результат.
  │
  ├─ S2  Сбор (параллельно, asyncio.gather(return_exceptions=True)):
  │      научные базы · патенты · репозитории · регуляторы · отраслевые медиа · агрегаторы
  │      → raw_documents (сырой текст + метаданные, до обработки)
  │
  ├─ S3  Нормализация: дедуп (DOI → канонический URL → simhash заголовка),
  │      определение языка, парсинг дат, оценка доверенности домена (services/trust)
  │
  ├─ S4  Извлечение кандидатов: LLM извлекает технологии из каждого документа
  │      + терминологический майнинг (n-граммы, частотный контраст к фоновому корпусу).
  │      Каждый кандидат несёт ссылки на документы-источники.
  │
  ├─ S5  Кластеризация: эмбеддинги названий+контекста, агломеративная кластеризация
  │      по косинусу (порог 0.82), слияние ru/en-вариантов и синонимов
  │
  ├─ S6  Обогащение кластера: ряд упоминаний по кварталам (OpenAlex/патенты),
  │      список независимых акторов, профиль доверенности, география
  │
  ├─ S7  Признаки (ml/features/from_evidence.py) — 63 кандидата в 7 блоках, в модели 18–25
  ├─ S8  Классификация + калибровка → вероятность «слабый сигнал» (уверенность, %)
  ├─ S9  Ворота исключения (G1–G5) → отсев зрелых/хайпа с текстовой причиной
  ├─ S10 Ранжирование → ТОП-15
  ├─ S11 Отчёт (LLM, строго по корпусу): описание, преимущество, кейс, объяснение статуса
  │      → валидатор цитат: каждый URL обязан быть в корпусе прогона, иначе перегенерация
  └─ S12 Русские резюме зарубежных источников с пометкой «авто-перевод»
```

Каждая стадия пишет прогресс в SSE-канал `/api/runs/{run_id}/stream`, поэтому UI показывает
живые счётчики: обработано источников, найдено кандидатов, отсеяно зрелых.

---

## 5. Модель данных (PostgreSQL)

```sql
-- прогоны и трассировка
runs(id uuid pk, query text, params jsonb, status text, started_at, finished_at,
     stats jsonb)                       -- счётчики для UI
run_events(id bigserial, run_id, stage text, level text, message text, payload jsonb, at)
llm_calls(id bigserial, run_id, stage text, provider text, model text,
          prompt_tokens int, completion_tokens int, latency_ms int, at)

-- сырые данные (требование ТЗ)
source_domains(domain text pk, name text, type text, trust_tier char(1),
               trust_reason text, country text)
raw_documents(id uuid pk, run_id, connector text, url text unique,
              canonical_url text, title text, authors jsonb, published_at date,
              language char(2), source_type text, trust_tier char(1),
              raw_text text, meta jsonb, fetched_at, content_hash text)

-- кандидаты и результаты
candidates(id uuid pk, run_id, name_ru text, name_en text, aliases jsonb,
           domain text, cluster_size int, created_at)
candidate_evidence(candidate_id, document_id, relevance real, quote text,
                   role text)           -- role: primary|confirming|indicator
candidate_features(candidate_id, features jsonb, timeseries jsonb)
predictions(candidate_id pk, raw_probability real, calibrated_probability real,
            threshold real, model_version text, feature_schema_version text,
            contributions jsonb)        -- вклад каждого признака и блока
exclusions(candidate_id pk, gate text, reason_ru text, evidence jsonb)
reports(candidate_id pk, description text, advantage text, case_example text,
        analyst_view text, explanation text, generated_by text, validated bool)
translations(document_id pk, summary_ru text, provider text, model text, is_auto bool)

-- датасет и метрики (этап 1)
datasets(id uuid pk, name text, kind text, uploaded_at)      -- kind: train|holdout|jury
dataset_rows(id uuid pk, dataset_id, payload jsonb, label int null, prediction real,
             contributions jsonb)
evaluations(id uuid pk, dataset_id, model_version text, metrics jsonb, created_at)

-- human-in-the-loop
feedback(id uuid pk, candidate_id, actor text, verdict text, comment text, at)
```

Индексы: `raw_documents(run_id)`, `raw_documents(canonical_url)`, GIN по `meta`,
`candidate_evidence(candidate_id)`, `predictions(calibrated desc)`.

---

## 6. HTTP API

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/search` | Открытый запрос → `run_id` |
| GET | `/api/runs/{id}/stream` | SSE-прогресс: стадия, счётчики, логи коннекторов |
| GET | `/api/runs/{id}` | Сводка: ТОП-15, статистика, исключённые |
| GET | `/api/candidates/{id}` | Документ-отчёт по инсайту |
| GET | `/api/candidates/{id}/sources` | Таблица источников с доверенностью |
| GET | `/api/runs/{id}/excluded` | Отсеянные кандидаты с причинами |
| POST | `/api/datasets/import` | Загрузка CSV/XLSX/JSON (в т.ч. датасета жюри) |
| POST | `/api/datasets/{id}/evaluate` | Метрики P/R/F1, confusion matrix — прямо на стенде |
| GET | `/api/datasets/{id}/predictions.csv` | Выгрузка предсказаний |
| GET | `/api/model/card` | Карточка модели: признаки, веса, метрики, версия |
| POST | `/api/feedback` | Экспертная оценка сигнала |
| GET | `/api/health` | Статус БД, коннекторов, LLM-провайдера |

---

## 7. Слой LLM и соответствие ТЗ

ТЗ (п.3.1) ограничивает список облачных API и требует «обязательного раскрытия формата выбора
модели для конкретного ответа и явного логирования».

```python
class LLMProvider(ABC):
    name: str; model: str; tier: Literal["allowed", "requires_approval"]
    def complete_json(self, *, instructions, prompt, schema, stage) -> dict: ...
```

Реестр провайдеров: `gigachat` (2 Lite/Pro/Max), `yandexgpt` (5/5.1), `qwen_local` (vLLM,
Qwen3-35B-A3B), `openai` (gpt-4.1). Маршрутизация по стадиям задаётся конфигом:

```yaml
stages:
  expand_query:       gigachat-2-lite      # дёшево, много вызовов
  extract_candidates: qwen3-35b-a3b        # локально, объёмно
  report:             gigachat-2-max       # качество текста
  translate:          gigachat-2-lite
```

Каждый вызов → запись в `llm_calls`. В UI на карточке модели — таблица «какая модель за что
отвечала в этом прогоне». Это закрывает требование ТЗ напрямую и снимает вопрос жюри.

**Классификатор слабых сигналов — не LLM.** Это scikit-learn поверх явных признаков; LLM
используется только для извлечения структуры из текста и генерации русских формулировок.
Отсюда и интерпретируемость, и воспроизводимость метрик.

---

## 8. Отказоустойчивость

- Коннектор: таймаут 20 с, 3 ретрая с экспоненциальной задержкой и джиттером, circuit breaker
  (5 отказов → пауза 5 мин), лимит RPS на домен, вежливый User-Agent, уважение robots.
- `asyncio.gather(return_exceptions=True)`: отказ источника → запись в `run_events` уровня WARN
  и пометка в UI «источник недоступен», прогон продолжается.
- Кэш сырых документов по `content_hash`: повторный прогон того же запроса не ходит в сеть.
- **Демо-режим**: `DEMO_CACHE=1` проигрывает сохранённый прогон из БД. Страховка от плохого
  интернета на защите — при этом честно помечается в UI как «кэшированный прогон от <дата>».
- Деградация LLM: при недоступности провайдера — фолбэк на следующий в цепочке, отметка в отчёте.
