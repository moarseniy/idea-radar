# Локальный запуск IDEA Radar без OpenRouter

Эта инструкция поднимает основной радар на `http://localhost:8080`: приложение работает в Docker
Compose, а локальная LLM — в Ollama на компьютере. Ключ OpenRouter не нужен. Для поиска
источников всё равно требуется интернет.

## 1. Подготовьте Ollama

Установите Ollama и скачайте модель. Если Ollama ещё не запущена, запустите её в отдельном
терминале:

```bash
ollama serve
```

В другом терминале загрузите модель:

```bash
ollama pull qwen3.6:35b-a3b
```

## 2. Настройте `.env`

Из корня проекта создайте файл конфигурации:

```bash
cp .env.example .env
```

В `.env` выставьте эти значения:

```env
RADAR_PROVIDER=local
LOCAL_MODEL=qwen3.6:35b-a3b
LOCAL_LLM_URL=http://host.docker.internal:11434/v1
FEATURE_MODEL=qwen3.6:35b-a3b
FEATURE_LLM_URL=http://host.docker.internal:11434/v1
RADAR_WEB_SEARCH=service
SEARCH_PROVIDERS=searxng
```

Особенно важно заменить `FEATURE_MODEL`: в `.env.example` там указана облачная модель. Обе
настройки LLM должны указывать на локальную модель и Ollama. Если API запускается прямо на
компьютере, а не в Docker, замените `host.docker.internal` на `localhost`.

## 3. Запустите приложение

Из корня проекта выполните:

```bash
docker compose up -d --build
```

Compose запустит приложение, PostgreSQL, поисковый сервис и SearXNG. Откройте
[http://localhost:8080](http://localhost:8080). Проверить состояние API можно командой:

```bash
curl http://localhost:8080/api/health
```

Остановить контейнеры, сохранив данные:

```bash
docker compose down
```

## Что работает локально

Планирование запроса и извлечение признаков выполняются через Ollama. Для `qwen3.6:35b-a3b`
в проекте есть совместимый артефакт логистической регрессии. Веб-поиск через SearXNG и открытые
источники обращаются в интернет; у некоторых дополнительных источников могут быть собственные
ограничения или требования к ключу. Отдельный старый раздел `/lab` в эту конфигурацию не входит.

Если изменили `.env`, примените настройки повторно:

```bash
docker compose up -d --build
```
