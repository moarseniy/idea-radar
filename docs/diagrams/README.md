# Схемы архитектуры

| Файл | Что показывает |
|---|---|
| `01-context.puml` | Контекст системы и компоненты слоёв backend |
| `02-pipeline.puml` | Пайплайн обработки открытого запроса, стадии S1–S12 |
| `03-data-model.puml` | Модель данных PostgreSQL |

## Рендеринг

Локально, без отправки исходников на внешний сервер:

```bash
brew install plantuml          # или: apt-get install plantuml
plantuml -tpng docs/diagrams/*.puml
plantuml -tsvg docs/diagrams/*.puml   # для презентации
```

PNG/SVG кладутся рядом с исходниками и коммитятся — жюри смотрит репозиторий без
установленного PlantUML.
