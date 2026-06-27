#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Creating .env from .env.example — fill in your API keys before using the app."
  cp .env.example .env
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker, then re-run: ./deploy.sh"
  exit 1
fi

echo "Building and starting Memory Agent Chat on http://localhost:8501"
docker compose up --build -d
echo "Done. View logs with: docker compose logs -f app"
