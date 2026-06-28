#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Creating .env from .env.example — fill in your API keys before using the app."
  cp .env.example .env
  echo ""
  echo "Edit .env now, then re-run: ./deploy.sh"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker, then re-run: ./deploy.sh"
  exit 1
fi

echo "Building and starting Memory Agent Chat on http://localhost:8501"
docker compose up --build -d

echo "Waiting for the app to become healthy…"
for _ in $(seq 1 30); do
  if docker compose ps --format json 2>/dev/null | grep -q '"Health":"healthy"'; then
    break
  fi
  if curl -sf http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo ""
echo "Ready: http://localhost:8501"
echo "Sample corpus: AUTO_INDEX_SAMPLE_CORPUS is enabled in docker-compose (indexes on first run)."
echo "Manual index:  docker compose exec app python scripts/index_sample_corpus.py"
echo "Logs:          docker compose logs -f app"
echo "Stop:          docker compose down"
