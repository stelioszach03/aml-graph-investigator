#!/usr/bin/env bash
set -euo pipefail
echo "🔎 Dev Doctor: checking files & env ..."
# Ensure .env exists
if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
  else
    echo "WARNING: .env.example not found; creating minimal .env"
    cat > .env <<EOF
APP_NAME=Graph AML Investigator
APP_ENV=dev
APP_VERSION=0.1.0
DATA_DIR=./data
MODEL_DIR=./models/baseline
SQLITE_URL=sqlite+aiosqlite:///./graph_aml.db
GRAPH_PAGE_RANK_ALPHA=0.85
EXPLAIN_MAX_PATH_LEN=6
EXPLAIN_K_PATHS=5
LOG_LEVEL=INFO
EOF
  fi
fi
mkdir -p data/raw data/interim data/processed models/baseline ui/web/dist
echo "✅ Dev Doctor OK."

