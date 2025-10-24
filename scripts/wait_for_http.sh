#!/usr/bin/env bash
# wait_for_http.sh <url> [timeout_seconds]
set -euo pipefail
URL="${1:-http://localhost:8000/api/v1/health}"
TIMEOUT="${2:-60}"
DEADLINE=$(( $(date +%s) + TIMEOUT ))
echo "Waiting for $URL up to ${TIMEOUT}s ..."
while true; do
  if curl -fsS "$URL" >/dev/null 2>&1; then
    echo "OK: $URL is reachable."
    exit 0
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "ERROR: Timeout waiting for $URL"
    exit 1
  fi
  sleep 1
done

