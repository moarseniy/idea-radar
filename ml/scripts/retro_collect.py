"""Сбор ретро-датасета: исторические прогоны радара по темам и их разметка.

Для каждой темы из ml/datasets/retro/topics.csv (slug, domain, as_of, query):
  1. исторический прогон радара на as_of — локальный сервис на отдельном порту, по
     одной теме за раз (RADAR_MAX_ACTIVE_RUNS=1); прогон сохраняется в RUNS/<slug>;
  2. разметка retro_dataset.py — в фоне, пока радар считает следующую тему; одновременно
     не больше LABEL_WORKERS разметок (они делят темп внешних API с радаром).
Возобновляемо: тема с готовым run.json не прогоняется заново, с label.log «строк:» —
не размечается. Сбор останавливается, если расход OpenRouter с начала превысил --budget.

Конфигурация радара — гибрид, как в docker-compose.yml: luna, веб-поиск DeepSeek.
Запуск: python ml/scripts/retro_collect.py [--budget 30] [--only slug1,slug2]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOPICS = ROOT / "ml" / "datasets" / "retro" / "topics.csv"
RUNS = ROOT / "ml" / ".cache" / "retro_runs"
OUT = ROOT / "ml" / "datasets" / "retro" / "retro_rows.csv"
PORT = 8099
LABEL_WORKERS = 2
RADAR_ENV = {
    "OPENROUTER_MODEL": "openai/gpt-5.6-luna", "OPENROUTER_RESEARCH_MODEL": "deepseek/deepseek-v4.1-flash-1017",
    "FEATURE_MODEL": "openai/gpt-5.6-luna", "FEATURE_LLM_EXTRA": "", "RADAR_REASONING_EFFORT": "",
    "RADAR_REASONING_BY_ROLE": "web_search:none", "RADAR_PROVIDER": "openrouter", "RADAR_WEB_SEARCH": "llm",
    "RADAR_EXTRA_CONNECTORS": "all", "RADAR_SEARCH_BRANCHES": "7", "RADAR_MAX_SOURCES": "128",
    "RADAR_MAX_CANDIDATES": "64", "RADAR_MAX_ACTIVE_RUNS": "1", "RADAR_DATABASE_URL": "",
}


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def api(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def spent() -> float:
    """Накопленный расход ключа OpenRouter, $."""
    from app.ml.env import env

    request = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                     headers={"Authorization": "Bearer " + env("OPENROUTER_API_KEY")})
    for attempt in range(5):  # рукопожатие TLS с OpenRouter иногда не укладывается в таймаут
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return float(json.loads(response.read())["data"]["usage"])
        except OSError:
            if attempt == 4:
                raise
            time.sleep(15)


def radar_run(topic: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "uploads").mkdir(exist_ok=True)
    env = {**os.environ, **RADAR_ENV, "STORAGE_DIR": str(folder), "DATABASE_PATH": str(folder / "lab.db"),
           "UPLOADS_DIR": str(folder / "uploads")}
    with (folder / "server.log").open("w") as server_log:
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
                                  cwd=ROOT, env=env, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            time.sleep(1)
            if "Application startup complete" in (folder / "server.log").read_text(errors="ignore"):
                break
        run_id = api("/api/searches", {"query": topic["query"], "as_of": topic["as_of"], "directions": [],
                                       "limit": 15})["id"]
        while (run := api(f"/api/searches/{run_id}"))["status"] in ("queued", "running"):
            time.sleep(20)
        (folder / "run.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
        log(f"радар {topic['slug']}: {run['status']}, кандидатов {run['counters'].get('candidates')}, "
            f"{int(run.get('elapsed_seconds') or 0)} с")
    finally:
        server.terminate()
        server.wait(timeout=30)


def labeled(folder: Path) -> bool:
    path = folder / "label.log"
    return path.exists() and "строк:" in path.read_text(errors="ignore")


def start_label(topic: dict, folder: Path) -> subprocess.Popen:
    handle = (folder / "label.log").open("w")
    return subprocess.Popen([sys.executable, str(ROOT / "ml" / "scripts" / "retro_dataset.py"), "--run-dir",
                             str(folder), "--domain", topic["domain"], "--out", str(OUT)],
                            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=30.0, help="предел расхода OpenRouter, $")
    parser.add_argument("--only", default="", help="через запятую: только эти темы")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ml" / "scripts"))
    import build_training_set
    build_training_set.load_dotenv()

    topics = list(csv.DictReader(TOPICS.open(encoding="utf-8")))
    if args.only:
        topics = [t for t in topics if t["slug"] in set(args.only.split(","))]
    start = spent()
    log(f"тем: {len(topics)}, бюджет ${args.budget}")
    labels: dict[str, subprocess.Popen] = {}

    def reap() -> None:
        for slug, process in list(labels.items()):
            if process.poll() is not None:
                tail = (RUNS / slug / "label.log").read_text(errors="ignore").strip().splitlines()[-1:]
                log(f"разметка {slug}: {tail[0] if tail else 'нет вывода'} (код {process.returncode})")
                del labels[slug]

    for topic in topics:
        folder = RUNS / topic["slug"]
        used = spent() - start
        if used > args.budget:
            log(f"бюджет исчерпан: ${used:.2f}; остановка перед {topic['slug']}")
            break
        if not (folder / "run.json").exists():
            log(f"радар {topic['slug']} (потрачено ${used:.2f})")
            radar_run(topic, folder)
        reap()
        if not labeled(folder):
            while len(labels) >= LABEL_WORKERS:
                time.sleep(10)
                reap()
            labels[topic["slug"]] = start_label(topic, folder)
    while labels:
        time.sleep(10)
        reap()
    log(f"готово, потрачено ${spent() - start:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
