from __future__ import annotations

from _path_guard import *  # noqa: F401

import argparse
import os
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.logging import configure_json_logger, get_logger
from app.graph.builder import load_edges, build_nx_graph, to_neo4j, save_graph
from app.graph.features import compute_node_features, persist_node_features, load_node_features


def choose_api_base() -> str | None:
    """Prefer localhost (same container) then service DNS when inside Compose."""
    candidates = [
        os.getenv("API_BASE", "").strip(),
        "http://localhost:8000",
        "http://api:8000",
    ]
    for cand in [c for c in candidates if c]:
        try:
            r = httpx.get(f"{cand}/api/v1/health", timeout=3.0)
            if r.status_code == 200:
                return cand
        except Exception:
            continue
    return None


def call_api_or_fallback(path: Path, push_neo4j: bool) -> None:
    log = get_logger("scripts.ingest")
    base = choose_api_base()
    token = os.getenv("API_AUTH_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    ingested = False
    try:
        if base:
            with httpx.Client(timeout=8.0) as client:
                resp = client.post(f"{base}/api/v1/ingest", json={"path": str(path), "push_neo4j": bool(push_neo4j)}, headers=headers)
                if resp.status_code == 200:
                    log.info("Ingested via API: {}", resp.json())
                    ingested = True
                else:
                    log.warning("API ingest failed (status={}): {}", resp.status_code, resp.text)
    except Exception as e:
        log.warning("API not reachable, falling back to local ingest: {}", e)

    if not ingested:
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

    # After successful ingest (API or local), construct labels_all.csv to ensure negatives exist
    try:
        import pandas as pd
        import pathlib
        # Load features to get node ids
        df_feats = load_node_features()
        nodes = pd.DataFrame({"node_id": df_feats.index.astype(str)})
        lbl = pathlib.Path("data/processed/labels.csv")
        df_lbl = pd.read_csv(lbl) if lbl.exists() else pd.DataFrame(columns=["node_id", "y"])
        all_lbl = nodes.merge(df_lbl, how="left", on="node_id")
        all_lbl["y"] = all_lbl["y"].fillna(0).astype(int)
        outp = pathlib.Path("data/processed/labels_all.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        all_lbl.to_csv(outp, index=False)
        print(f"Saved {outp} (pos={int(all_lbl['y'].sum())} / total={len(all_lbl)})")
    except Exception as e:
        log.warning("Failed to build labels_all.csv: {}", e)


def main():
    configure_json_logger()
    parser = argparse.ArgumentParser(description="Ingest edges and compute features")
    parser.add_argument("--path", type=Path, required=True, help="Path to edges CSV/JSONL")
    parser.add_argument("--neo4j", type=int, default=0, help="Set to 1 to push to Neo4j if configured")
    args = parser.parse_args()
    call_api_or_fallback(args.path, bool(args.neo4j))


if __name__ == "__main__":
    main()
