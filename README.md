# Graph AML Investigator

End-to-end graph-based AML/KYC pipeline with explainability and a clean UI.

## Why it matters
- Graph thinking surfaces rings (smurfing, fan-out, collusion) που δεν φαίνονται σε row-wise ML.
- Explainability (paths + feature contributions) → audit-friendly.
- One-command demo with Docker + CI to prove it works.

## Stack
- Backend: FastAPI, NetworkX, LightGBM, (optional) Neo4j
- Explainability: path explanations + local surrogate; what-if (local/global)
- UI: React/Vite (dark), metrics card, Scores, Case Explorer
- Ops: Docker Compose, pytest, smoke_e2e, GitHub Actions CI

## Quickstart (Docker)
```bash
make up            # build images, run dockerized tests, bring stack up
make demo-run      # generate → ingest → train → score (so Scores have data)
# visit UI:
http://localhost:5173
# API health:
http://localhost:8000/api/v1/health
```

### Open vs Protected
- Open demo: API_AUTH_TOKEN is empty → no token needed.
- Protected demo: set a token and auto-seed UI
```bash
make demo-protected TOKEN=demo123
```

### One-click demos
```bash
make demo-open                  # open API + pipeline + UI
make demo-protected TOKEN=xxx   # protected API, UI pre-seeded with token
```

## Endpoints (v1)
- GET /api/v1/health → {status, version, nodes, edges}
- POST /api/v1/ingest → build graph & features from CSV/JSONL
- POST /api/v1/train → LightGBM baseline; persists metrics + medians
- POST /api/v1/score → { topK:[{node_id,score}], topk, count, constant_scores }
- GET /api/v1/case/{node} → ego summary
- GET /api/v1/explain/{node} → { why, paths, top_contributors }
- POST /api/v1/what-if → local/global recompute & delta
- GET /api/v1/metrics/last → latest training metrics (ROC-AUC, PR-AUC, P@K, trained_at)
- GET /api/v1/auth/status → {protected: bool}
- GET /api/v1/debug/routes → registered routes (debug)

## Metrics
On every train we persist:
- ROC-AUC / PR-AUC
- Precision@100/500/1000
- trained_at timestamp
- important_features (gain)

UI shows metrics card and Range (min–max) on Scores, toggle Decimals (3/6), and a Mid-range slice to visualize distribution beyond top-K.

## What-if (local)
Local 2-hop ego recompute for instant simulation:
- Recomputes a compact feature subset (degrees, counts, pagerank on ego)
- Fills missing features with global medians
- Returns {baseline, new_score, delta, recompute:"local"}

## Dev (UI)
```bash
cd ui/web
npm i
npm run dev   # Vite proxy: /api -> :8000
```
`ui/web/.env.local`:
```ini
VITE_API_BASE=http://localhost:8000
# VITE_DEFAULT_TOKEN=demo123  # optional preseed
```

## Run tests
```bash
make docker-test     # dockerized pytest
make e2e             # smoke script (generate → ingest → train → score → explain)
```

## Troubleshooting
- All top scores look 1.000: top-K is very confident. Use Decimals: 6, check Range, or press Mid-range.
- 401 Unauthorized: set the token in UI header or clear API_AUTH_TOKEN for open demo.
- 404 on /auth/status or /metrics/last: rebuild API (`make docker-build && make up`) and verify `/api/v1/debug/routes`.

## License
MIT

---

Ensure `.env.example` contains:
- APP_NAME, APP_ENV, APP_VERSION, LOG_LEVEL
- DATA_DIR, MODEL_DIR, SQLITE_URL
- NEO4J_[URI|USER|PASS|DATABASE]
- API_AUTH_TOKEN (empty by default)
- GRAPH_PAGE_RANK_ALPHA, EXPLAIN_MAX_PATH_LEN, EXPLAIN_K_PATHS

