#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"

if [ -f .env ]; then
  # shellcheck disable=SC2046
  export $(grep -E '^(API_AUTH_TOKEN)=' .env | xargs -d '\n' -I {} echo {}) || true
fi

AUTH_HEADERS=(-H "Content-Type: application/json")
if [ -n "${API_AUTH_TOKEN:-}" ]; then
  AUTH_HEADERS+=(-H "Authorization: Bearer ${API_AUTH_TOKEN}")
fi

scripts/wait_for_http.sh "${API_BASE}/api/v1/health" 120

# Prefer running Python inside the API container if available
PY_CMD="python"
if command -v docker >/dev/null 2>&1; then
  if docker compose ps -q api >/dev/null 2>&1 && [ -n "$(docker compose ps -q api)" ]; then
    PY_CMD="docker compose exec -T api python"
  fi
fi

echo "[A] Generate synthetic data"
${PY_CMD} scripts/generate_synth.py --n_accounts 1500 --fraud_rings 6 --seed 42

echo "[B] Ingest via API"
curl -fsS -X POST "${API_BASE}/api/v1/ingest" \
  "${AUTH_HEADERS[@]}" \
  -d '{"path":"data/raw/synth_edges.csv","push_neo4j":false}' | tee /tmp/ingest.json

feature_path=""
if command -v jq >/dev/null 2>&1; then
  feature_path=$(jq -r '.features_path' /tmp/ingest.json)
  n_features=$(jq -r '.n_features' /tmp/ingest.json)
else
  feature_path=$(python - <<'PY'
import json,sys
obj=json.load(open('/tmp/ingest.json'))
print(obj.get('features_path',''))
PY
)
  n_features=$(python - <<'PY'
import json,sys
obj=json.load(open('/tmp/ingest.json'))
print(obj.get('n_features',0))
PY
)
fi

if [ -z "$feature_path" ] || [ ! -f "$feature_path" ] || [ "${n_features:-0}" -le 0 ]; then
  echo "ERROR: Ingest failed or features not persisted correctly (path=$feature_path n_features=$n_features)" >&2
  exit 2
fi

echo "[C] Train via API"
curl -fsS -X POST "${API_BASE}/api/v1/train" \
  "${AUTH_HEADERS[@]}" \
  -d '{"labels_path":"data/processed/labels.csv"}' | tee /tmp/train.json

roc_auc_val=""
if command -v jq >/dev/null 2>&1; then
  roc_auc_val=$(jq -r '.metrics.roc_auc // 0' /tmp/train.json)
else
  roc_auc_val=$(python - <<'PY'
import json;print((json.load(open('/tmp/train.json')) or {}).get('metrics',{}).get('roc_auc',0) or 0)
PY
)
fi
if [ "${roc_auc_val}" != "null" ] && awk "BEGIN{exit !(${roc_auc_val:-0} > 0.5)}"; then
  echo "ROC-AUC looks ok: ${roc_auc_val}"
else
  echo "WARN: ROC-AUC low or undefined: ${roc_auc_val}"
fi

echo "[D] Score via API"
curl -fsS -X POST "${API_BASE}/api/v1/score" \
  "${AUTH_HEADERS[@]}" \
  -d '{"topk":50}' | tee /tmp/score.json

first_node=""
if command -v jq >/dev/null 2>&1; then
  topn=$(jq -r '.results | length' /tmp/score.json)
  first_node=$(jq -r '.results[0].node_id' /tmp/score.json)
else
  topn=$(python - <<'PY'
import json;print(len((json.load(open('/tmp/score.json')) or {}).get('results',[])))
PY
)
  first_node=$(python - <<'PY'
import json;res=(json.load(open('/tmp/score.json')) or {}).get('results',[]);print(res[0]['node_id'] if res else '')
PY
)
fi
if [ -z "$first_node" ] || [ "${topn:-0}" -le 0 ]; then
  echo "ERROR: Score returned no results" >&2
  exit 2
fi

echo "[E] Explain"
curl -fsS "${API_BASE}/api/v1/explain/${first_node}" "${AUTH_HEADERS[@]}" | tee /tmp/explain.json

has_paths=0
has_why=0
if command -v jq >/dev/null 2>&1; then
  jq -e '.paths and (.paths|type=="array")' /tmp/explain.json >/dev/null 2>&1 && has_paths=1 || true
  jq -e '.why and (.why|type=="string")' /tmp/explain.json >/dev/null 2>&1 && has_why=1 || true
else
  python - <<'PY'
import json,sys
obj=json.load(open('/tmp/explain.json'))
import os
ok=1
if not isinstance(obj.get('paths',None),list): ok=0
if not isinstance(obj.get('why',''),str): ok=0
sys.exit(0 if ok else 1)
PY
  [ $? -eq 0 ] && has_paths=1 && has_why=1 || true
fi

if [ $has_paths -ne 1 ] || [ $has_why -ne 1 ]; then
  echo "ERROR: Explain missing required keys (paths/why)" >&2
  exit 3
fi

echo "✅ E2E SMOKE PASS"
echo "Summary: features=${n_features}, roc_auc=${roc_auc_val}, first_node=${first_node}, paths_ok=${has_paths}"
