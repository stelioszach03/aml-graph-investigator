#!/usr/bin/env bash
set -euo pipefail

echo "Resetting Neo4j volumes and starting service..."
docker compose down -v || true
docker compose --profile dev up -d neo4j
scripts/wait_for_http.sh http://localhost:7474 120
echo "Neo4j is up at http://localhost:7474"
echo "Credentials: neo4j / testtest"

