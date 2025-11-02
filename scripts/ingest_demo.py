from __future__ import annotations

import os, sys
here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, ".."))
if repo not in sys.path:
    sys.path.insert(0, repo)

import argparse
import os
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.logging import configure_json_logger, get_logger
from app.graph.builder import load_edges, build_nx_graph, to_neo4j, save_graph
from app.graph.features import compute_node_features, persist_node_features


def call_api_or_fallback(path: Path, push_neo4j: bool) -> None:
    log = get_logger("scripts.ingest")
    base = os.getenv("API_BASE", "http://localhost:8000")
    token = os.getenv("API_AUTH_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(f"{base}/api/v1/ingest", json={"path": str(path), "push_neo4j": bool(push_neo4j)}, headers=headers)
            if resp.status_code == 200:
                log.info("Ingested via API: {}", resp.json())
                return
            else:
                log.warning("API ingest failed (status={}): {}", resp.status_code, resp.text)
    except Exception as e:
        log.warning("API not reachable, falling back to local ingest: {}", e)

    # Fallback: local ingest
    settings = get_settings()
    edges = load_edges(path)
    G = build_nx_graph(edges)
    save_graph(G, Path(settings.graph_path))
    if push_neo4j:
        to_neo4j(edges)
    df_feat, _ = compute_node_features(G)
    out_path = persist_node_features(df_feat)
    log.info("Local ingest: nodes={} edges={} features={}", G.number_of_nodes(), G.number_of_edges(), out_path)


def main():
    configure_json_logger()
    parser = argparse.ArgumentParser(description="Ingest edges and compute features")
    parser.add_argument("--path", type=Path, required=True, help="Path to edges CSV/JSONL")
    parser.add_argument("--neo4j", type=int, default=0, help="Set to 1 to push to Neo4j if configured")
    args = parser.parse_args()
    call_api_or_fallback(args.path, bool(args.neo4j))


if __name__ == "__main__":
    main()
