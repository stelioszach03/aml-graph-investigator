from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx

from app.core.logging import get_logger


def degree_features(G: nx.Graph) -> dict:
    deg = dict(G.degree())
    in_deg = dict(G.in_degree()) if G.is_directed() else None
    out_deg = dict(G.out_degree()) if G.is_directed() else None
    features = {}
    for n in G.nodes:
        features[n] = {
            "degree": float(deg.get(n, 0)),
        }
        if in_deg is not None:
            features[n]["in_degree"] = float(in_deg.get(n, 0))
        if out_deg is not None:
            features[n]["out_degree"] = float(out_deg.get(n, 0))
    return features


def pagerank_features(G: nx.Graph, alpha: float = 0.85) -> dict:
    pr = nx.pagerank(G, alpha=alpha) if G.number_of_nodes() > 0 else {}
    return {n: {"pagerank": float(pr.get(n, 0.0))} for n in G.nodes}


def motif_counts(G: nx.Graph) -> dict:
    # Triangle counts as a basic motif
    tri = nx.triangles(G) if not G.is_directed() else {}
    return {n: {"triangles": float(tri.get(n, 0.0))} for n in G.nodes}


def merge_feature_maps(*feature_maps: dict) -> dict:
    out: dict = {}
    for fm in feature_maps:
        for k, v in fm.items():
            out.setdefault(k, {}).update(v)
    return out


def ego_features(G: nx.Graph, node_id, radius: int = 1) -> dict:
    if node_id not in G:
        raise KeyError("node not in graph")
    ego = nx.ego_graph(G, node_id, radius=radius)
    feats = merge_feature_maps(
        degree_features(ego),
        pagerank_features(ego),
        motif_counts(ego),
    )
    # Include simple ego stats
    stats = {
        "ego_nodes": float(ego.number_of_nodes()),
        "ego_edges": float(ego.number_of_edges()),
    }
    feats[node_id] = {**feats.get(node_id, {}), **stats}
    return feats[node_id]


# ------------------------------------------------------------
# Comprehensive node feature computation and persistence
# ------------------------------------------------------------


def _edge_amount(data: dict) -> float:
    v = data.get("amount", data.get("weight", 1.0))
    try:
        return float(v)
    except Exception:
        return 0.0


def _robust_scale(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    med = df.median(numeric_only=True)
    q1 = df.quantile(0.25, numeric_only=True)
    q3 = df.quantile(0.75, numeric_only=True)
    iqr = (q3 - q1)
    iqr = iqr.replace(0, 1.0)  # avoid division by zero
    scaled = df.copy()
    scaled[med.index] = (df[med.index] - med) / iqr
    scaled = scaled.fillna(0.0)
    return scaled, med.to_dict(), iqr.to_dict()


def _simplify_to_digraph(G: nx.Graph) -> nx.DiGraph:
    if isinstance(G, nx.MultiDiGraph):
        DG = nx.DiGraph()
        for u, v, data in G.edges(data=True):
            if DG.has_edge(u, v):
                # keep max weight
                DG[u][v]["amount"] = max(DG[u][v].get("amount", 0.0), _edge_amount(data))
            else:
                DG.add_edge(u, v, amount=_edge_amount(data))
        return DG
    if isinstance(G, nx.MultiGraph):
        return nx.DiGraph(nx.Graph(G))
    if G.is_directed():
        return nx.DiGraph(G)
    else:
        return nx.DiGraph(G)  # treat undirected as bidirectional


def _undirected_simple(G: nx.Graph) -> nx.Graph:
    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        return nx.Graph(G)
    return nx.Graph(G) if G.is_directed() else nx.Graph(G)


def compute_node_features(G: nx.Graph) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Compute a rich feature matrix for nodes.

    Returns (df_scaled, metadata) where df_scaled is indexed by node id and
    includes columns:
      degree_in, degree_out, pagerank, betweenness, clustering,
      ego_density (radius=2), avg_neighbor_degree, txn_amt_sum_in/out,
      txn_cnt_in/out, triad_motifs, hub_score, authority_score.
    Metadata contains: features (list), scaler medians/iqr.
    """
    log = get_logger("graph.features")
    nodes = list(G.nodes())
    if not nodes:
        df_empty = pd.DataFrame(index=pd.Index([], name="node_id"))
        return df_empty, {"features": []}

    DG = _simplify_to_digraph(G)
    UG = _undirected_simple(G)

    # Degrees
    deg_in = dict(DG.in_degree()) if DG.is_directed() else dict(DG.degree())
    deg_out = dict(DG.out_degree()) if DG.is_directed() else dict(DG.degree())

    # PageRank (weighted if amount available)
    try:
        pr = nx.pagerank(DG, alpha=0.85, weight="amount") if DG.number_of_nodes() > 0 else {}
    except Exception:
        pr = {n: 0.0 for n in nodes}

    # Betweenness centrality (can be expensive)
    try:
        btw = nx.betweenness_centrality(DG, normalized=True)
    except Exception:
        btw = {n: 0.0 for n in nodes}

    # Clustering on undirected projection
    try:
        clust = nx.clustering(UG)
    except Exception:
        clust = {n: 0.0 for n in nodes}

    # Ego density radius=2
    ego_density = {}
    for n in nodes:
        try:
            ego = nx.ego_graph(UG, n, radius=2)
            ego_density[n] = float(nx.density(ego))
        except Exception:
            ego_density[n] = 0.0

    # Average neighbor degree (undirected)
    try:
        andeg = nx.average_neighbor_degree(UG)
    except Exception:
        andeg = {n: 0.0 for n in nodes}

    # Transaction sums and counts
    amt_in = {n: 0.0 for n in nodes}
    amt_out = {n: 0.0 for n in nodes}
    cnt_in = {n: 0 for n in nodes}
    cnt_out = {n: 0 for n in nodes}

    if isinstance(G, nx.MultiDiGraph) or DG.is_directed():
        # Use original graph if it has parallel edges to count properly
        iter_graph = G if isinstance(G, nx.MultiDiGraph) else DG
        for v in nodes:
            for _, _, _, data in iter_graph.in_edges(v, data=True, keys=True):
                amt_in[v] += _edge_amount(data)
                cnt_in[v] += 1
            for _, _, _, data in iter_graph.out_edges(v, data=True, keys=True):
                amt_out[v] += _edge_amount(data)
                cnt_out[v] += 1
    else:
        # Undirected: count degree as both in/out
        for u, v, data in G.edges(data=True):
            w = _edge_amount(data)
            amt_in[u] += w
            amt_in[v] += w
            amt_out[u] += w
            amt_out[v] += w
            cnt_in[u] += 1
            cnt_in[v] += 1
            cnt_out[u] += 1
            cnt_out[v] += 1

    # Triangles (3-cycle in undirected sense)
    try:
        tri = nx.triangles(UG)
    except Exception:
        tri = {n: 0.0 for n in nodes}

    # HITS hubs/authorities on directed simplified graph
    try:
        hubs, auths = nx.hits(DG, max_iter=1000, normalized=True)
    except Exception:
        hubs = {n: 0.0 for n in nodes}
        auths = {n: 0.0 for n in nodes}

    data = {
        "degree_in": pd.Series({n: float(deg_in.get(n, 0.0)) for n in nodes}),
        "degree_out": pd.Series({n: float(deg_out.get(n, 0.0)) for n in nodes}),
        "pagerank": pd.Series({n: float(pr.get(n, 0.0)) for n in nodes}),
        "betweenness": pd.Series({n: float(btw.get(n, 0.0)) for n in nodes}),
        "clustering": pd.Series({n: float(clust.get(n, 0.0)) for n in nodes}),
        "ego_density": pd.Series({n: float(ego_density.get(n, 0.0)) for n in nodes}),
        "avg_neighbor_degree": pd.Series({n: float(andeg.get(n, 0.0)) for n in nodes}),
        "txn_amt_sum_in": pd.Series({n: float(amt_in.get(n, 0.0)) for n in nodes}),
        "txn_amt_sum_out": pd.Series({n: float(amt_out.get(n, 0.0)) for n in nodes}),
        "txn_cnt_in": pd.Series({n: float(cnt_in.get(n, 0)) for n in nodes}),
        "txn_cnt_out": pd.Series({n: float(cnt_out.get(n, 0)) for n in nodes}),
        "triad_motifs": pd.Series({n: float(tri.get(n, 0.0)) for n in nodes}),
        "hub_score": pd.Series({n: float(hubs.get(n, 0.0)) for n in nodes}),
        "authority_score": pd.Series({n: float(auths.get(n, 0.0)) for n in nodes}),
    }

    df = pd.DataFrame(data)
    df.index = pd.Index(nodes, name="node_id")
    df = df.sort_index()

    df_scaled, med, iqr = _robust_scale(df)
    meta = {"features": list(df.columns), "scaler_median": med, "scaler_iqr": iqr}
    return df_scaled, meta


def compute_local_subset_features(G: nx.Graph, node_id) -> dict:
    """Compute a compact subset of features on a local subgraph.

    Includes: degree_in/out, txn_amt_sum_in/out, txn_cnt_in/out, avg_neighbor_degree (undirected),
    pagerank (on simplified directed graph) using alpha from settings.
    """
    if node_id not in G:
        raise KeyError("node not in subgraph")

    # Degrees and counts
    if G.is_directed():
        deg_in = float(G.in_degree(node_id))
        deg_out = float(G.out_degree(node_id))
    else:
        d = float(G.degree(node_id))
        deg_in = d
        deg_out = d

    # Amount sums and counts
    amt_in = 0.0
    amt_out = 0.0
    cnt_in = 0
    cnt_out = 0
    if isinstance(G, (nx.MultiDiGraph, nx.DiGraph)):
        iter_in = G.in_edges(node_id, data=True, keys=True) if isinstance(G, nx.MultiDiGraph) else G.in_edges(node_id, data=True)
        for e in iter_in:
            data = e[-1]
            amt_in += float(data.get("amount", data.get("weight", 1.0)))
            cnt_in += 1
        iter_out = G.out_edges(node_id, data=True, keys=True) if isinstance(G, nx.MultiDiGraph) else G.out_edges(node_id, data=True)
        for e in iter_out:
            data = e[-1]
            amt_out += float(data.get("amount", data.get("weight", 1.0)))
            cnt_out += 1
    else:
        for _, v, data in G.edges(node_id, data=True):
            w = float(data.get("amount", data.get("weight", 1.0)))
            amt_in += w
            amt_out += w
            cnt_in += 1
            cnt_out += 1

    # Avg neighbor degree on undirected projection
    try:
        UG = nx.Graph(G)
        andeg = nx.average_neighbor_degree(UG)
        avg_nd = float(andeg.get(node_id, 0.0))
    except Exception:
        avg_nd = 0.0

    # Pagerank on simplified directed graph
    try:
        from app.core.config import get_settings as _get_settings
        alpha = _get_settings().graph_page_rank_alpha
    except Exception:
        alpha = 0.85
    try:
        DG = _simplify_to_digraph(G)
        pr = nx.pagerank(DG, alpha=alpha, weight="amount")
        pr_val = float(pr.get(node_id, 0.0))
    except Exception:
        pr_val = 0.0

    return {
        "degree_in": deg_in,
        "degree_out": deg_out,
        "txn_amt_sum_in": float(amt_in),
        "txn_amt_sum_out": float(amt_out),
        "txn_cnt_in": float(cnt_in),
        "txn_cnt_out": float(cnt_out),
        "avg_neighbor_degree": avg_nd,
        "pagerank": pr_val,
    }


def _features_path() -> Path:
    # Import lazily to avoid hard dependency at module import time
    from app.core.config import get_settings
    settings = get_settings()
    return Path(settings.model_dir) / "features.parquet"


def persist_node_features(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Persist features to Parquet under MODEL_DIR/features.parquet by default.

    Falls back to CSV if a parquet engine is not available.
    Returns the written path.
    """
    log = get_logger("graph.features")
    p = Path(path) if path else _features_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p)
        return p
    except Exception as e:
        # Fallback to CSV
        alt = p.with_suffix(".csv")
        log.warning("Parquet write failed ({}). Falling back to CSV at {}", e, alt)
        df.to_csv(alt)
        return alt


def load_node_features(path: Optional[Path] = None) -> pd.DataFrame:
    p = Path(path) if path else _features_path()
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    # Try CSV fallback
    p_csv = p if p.suffix == ".csv" else p.with_suffix(".csv")
    if p_csv.exists():
        df = pd.read_csv(p_csv)
        if "node_id" in df.columns:
            df = df.set_index("node_id")
        return df
    raise FileNotFoundError(p)


# ------------------------------------------------------------
# Pair motif utility
# ------------------------------------------------------------


def compute_pair_motif_counts(G: nx.Graph, u, v) -> Dict[str, float]:
    """Compute simple pairwise motif counts for case explanations.

    Returns a dict with keys:
      common_neighbors, triangles, two_hop_paths, jaccard, adamic_adar,
      edge_exists (0/1), u_degree, v_degree.
    Uses undirected projection for motif counts.
    """
    UG = _undirected_simple(G)
    res: Dict[str, float] = {
        "common_neighbors": 0.0,
        "triangles": 0.0,
        "two_hop_paths": 0.0,
        "jaccard": 0.0,
        "adamic_adar": 0.0,
        "edge_exists": 0.0,
        "u_degree": 0.0,
        "v_degree": 0.0,
    }

    if u not in UG or v not in UG:
        return res

    Nu = set(UG.neighbors(u))
    Nv = set(UG.neighbors(v))
    common = Nu & Nv
    union = (Nu | Nv) - {u, v}
    res["common_neighbors"] = float(len(common))
    res["triangles"] = float(len(common))  # each common neighbor closes a triangle
    res["two_hop_paths"] = float(len(common))
    res["edge_exists"] = 1.0 if UG.has_edge(u, v) else 0.0
    res["u_degree"] = float(UG.degree(u))
    res["v_degree"] = float(UG.degree(v))
    res["jaccard"] = float(len(common) / len(union)) if len(union) > 0 else 0.0

    # Adamic-Adar index sum over common neighbors
    aa = 0.0
    for w in common:
        dw = UG.degree(w)
        if dw > 1:
            aa += 1.0 / np.log(dw)
    res["adamic_adar"] = float(aa)
    return res
