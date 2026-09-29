"""Сервис веб-поиска: один интерфейс над разными поисковыми провайдерами.

Провайдеры (`SEARCH_PROVIDERS`, по умолчанию `searxng`):
  searxng    — свой инстанс SearXNG (метапоиск без ключей; поднимается в Docker Compose);
  tavily     — Tavily Search API (`TAVILY_API_KEY`);
  exa        — Exa (`EXA_API_KEY`);
  brave      — Brave Search API (`BRAVE_API_KEY`);
  yandex     — Yandex Search API v2 (`YANDEX_SEARCH_API_KEY`, `YANDEX_FOLDER_ID`);
  openserp   — свой инстанс OpenSERP (скрейпер Google/Яндекс/Bing, `OPENSERP_URL`);
  openrouter — веб-плагин OpenRouter: LLM ищет и цитирует ссылки (`OPENROUTER_API_KEY`).

Несколько провайдеров через запятую — цепочка: при отказе или пустой выдаче работает
следующий (`SEARCH_MODE=fallback`), либо опрашиваются все и выдача объединяется (`fanout`).

Запуск отдельным сервисом: `uvicorn app.search.server:app` (в Compose — контейнер `search`).
Клиенты (радар, скрипты датасета) ходят в него по `SEARCH_URL`; без `SEARCH_URL` тот же код
вызывается в процессе (`app.search.client.web_search`).
"""
from app.search.base import SearchQuery

__all__ = ["SearchQuery"]
