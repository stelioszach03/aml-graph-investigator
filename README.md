# AML Graph Investigator

Graph-native anti-money-laundering triage: 14 topology and flow features per node computed with NetworkX, scored by LightGBM, surfaced with k-shortest-path rationales and a what-if edge simulator in a React console.

[![CI](https://github.com/stelioszach03/aml-graph-investigator/actions/workflows/ci.yml/badge.svg)](https://github.com/stelioszach03/aml-graph-investigator/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-22d3ee?style=flat-square)](LICENSE)

## Results

**The graph is synthetic.** All 3,061 nodes and 51,758 edges — including the fraud-ring motifs that produce the labels — come from [`scripts/generate_synth.py`](scripts/generate_synth.py). No real AML data has been used.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8716 |
| PR-AUC | 0.6936 |
| Brier score | 0.0470 |
| Precision @ 100 | 0.200 |
| Precision @ 500 / @ 1000 | 0.0817 |

**Provenance caveat: these come from a training run, and the trained artifacts are not committed** — `models/baseline/` holds only a `.gitkeep`, so the table cannot be verified from a clean checkout. Reproduce it with the commands below, then read `GET /api/v1/metrics/last`. A model that separates a generator's own fraud motifs is evidence of a working pipeline, not of real-world AML performance.

The 14 features are pure NetworkX, so every signal is auditable: `degree_in`, `degree_out`, `txn_cnt_in`, `txn_cnt_out`, `txn_amt_sum_in`, `txn_amt_sum_out`, `pagerank`, `betweenness`, `clustering`, `avg_neighbor_degree`, `ego_density`, `triad_motifs`, `hub_score`, `authority_score`. Top LightGBM gains: `txn_cnt_out` (14,414), `degree_out` (6,729), `avg_neighbor_degree` (4,578).

## Run

```bash
cp .env.example .env
docker compose -f docker-compose.vps.yml up -d --build
# React UI :18730 · FastAPI :18300/docs · Neo4j Browser :7474
```

Local venv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_synth.py && python scripts/ingest_demo.py && python scripts/score_demo.py
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

make test                    # pytest: features, API, explanations
bash scripts/smoke_e2e.sh    # ingest → train → score → case → explain
```

Training is `POST /api/v1/train`; scoring, case detail, explanation and what-if are under `/api/v1/`.

## Limitations

- **Synthetic data end to end.** Graph, labels and fraud motifs all come from one generator. The metrics measure how well LightGBM recovers that generator's patterns; they do not transfer to real AML data.
- **No committed model artifact**, so the metrics table is reproducible but not independently verifiable from the repo.
- **`app/ml/gnn_optional.py` is a placeholder.** The shipped model is LightGBM on 14 precomputed node features — there is no graph neural network.
- **Features are recomputed on the full graph.** Betweenness and HITS are global, so ingest cost grows superlinearly. This is a batch pipeline, not a streaming one.
- **Explanations are local surrogates**, not exact attributions, and path rationales are capped by `EXPLAIN_MAX_PATH_LEN`.
- Auth is a single optional bearer token; an empty `API_AUTH_TOKEN` leaves the API open.

## License

MIT — see [LICENSE](LICENSE).
