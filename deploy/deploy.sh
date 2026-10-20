#!/usr/bin/env bash
# Выкладка закоммиченного HEAD на сервер с записью app/VERSION (виден в /api/health).
#   DEPLOY_HOST=root@<ip> deploy/deploy.sh
# .env, storage/ и данные PostgreSQL на сервере не трогаются: архив содержит только
# файлы из git. Удалённые из репозитория файлы на сервере остаются — их чистят руками.
set -euo pipefail

host="${DEPLOY_HOST:?укажите DEPLOY_HOST, например root@1.2.3.4}"
dir="${DEPLOY_DIR:-/opt/idea-radar}"

if ! git diff --quiet HEAD --; then
  echo "Есть незакоммиченные изменения: выкладывается только HEAD, закоммитьте их" >&2
  exit 1
fi

version="$(git rev-parse --short HEAD) $(git log -1 --format=%cs) $(git rev-parse --abbrev-ref HEAD)"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
git archive HEAD | tar -x -C "$staging"
echo "$version" > "$staging/app/VERSION"

tar -C "$staging" -cf - . | ssh "$host" "tar -x -C '$dir'"
ssh "$host" "cd '$dir' && docker compose up -d --build"
echo "Выложено: $version"
