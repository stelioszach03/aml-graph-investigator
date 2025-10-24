from __future__ import annotations

import math
from heapq import nsmallest
from typing import Dict, List, Optional, Tuple

import networkx as nx


def k_shortest_paths(G: nx.Graph, source, target, k: int = 3, cutoff: Optional[int] = None) -> List[List[str]]:
    if source not in G or target not in G:
        return []
    try:
        gen = nx.shortest_simple_paths(G, source, target, weight=None)
    except nx.NetworkXNoPath:
        return []
    paths: List[List[str]] = []
    for path in gen:
        if cutoff is not None and len(path) - 1 > cutoff:
            continue
        paths.append([str(n) for n in path])
        if len(paths) >= k:
            break
    return paths


def surrogate_local_explain(instance_features: dict, training_features: list[dict], training_labels: list[int],
                            top_n: int = 5) -> dict:
    """
    Very lightweight surrogate using a DecisionTree to obtain feature importances for the local neighborhood of an
    instance. Not a true SHAP, but a fast proxy for MVP purposes.
    """
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    if not training_features or not training_labels:
        return {"feature_importance": {}, "note": "no training data provided"}

    # Build consistent feature matrix
    keys = sorted({k for row in training_features for k in row.keys()})
    X = np.array([[row.get(k, 0.0) for k in keys] for row in training_features], dtype=float)
    y = np.array(training_labels, dtype=int)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    x0 = np.array([[instance_features.get(k, 0.0) for k in keys]], dtype=float)
    prob = float(clf.predict_proba(x0)[0, 1]) if hasattr(clf, "predict_proba") else float(clf.predict(x0)[0])
    importances = clf.feature_importances_.tolist()
    ranked = sorted(zip(keys, importances), key=lambda t: t[1], reverse=True)[:top_n]
    return {
        "pred_proba": prob,
        "feature_importance": {k: float(v) for k, v in ranked},
    }


# ------------------------------------------------------------
# Path explanations
# ------------------------------------------------------------


def _edge_amount(data: dict) -> float:
    try:
        return float(data.get("amount", data.get("weight", 1.0)))
    except Exception:
        return 1.0


def _to_simple_digraph_with_costs(G: nx.Graph) -> Tuple[nx.DiGraph, Dict[Tuple[str, str], dict]]:
    """Aggregate parallel edges, compute normalized amount and path costs.

    Returns (DG, pair_info) where DG has edge attribute 'cost' and pair_info[(u,v)]
    contains {'amount_sum', 'freq'}.
    """
    DG = nx.DiGraph()
    totals: Dict[Tuple[str, str], dict] = {}

    if isinstance(G, nx.MultiDiGraph):
        for u, v, data in G.edges(data=True):
            a = _edge_amount(data)
            key = (str(u), str(v))
            t = totals.setdefault(key, {"amount_sum": 0.0, "freq": 0})
            t["amount_sum"] += a
            t["freq"] += 1
    elif isinstance(G, nx.MultiGraph):
        for u, v, data in G.edges(data=True):
            a = _edge_amount(data)
            key1 = (str(u), str(v))
            key2 = (str(v), str(u))
            t1 = totals.setdefault(key1, {"amount_sum": 0.0, "freq": 0})
            t2 = totals.setdefault(key2, {"amount_sum": 0.0, "freq": 0})
            t1["amount_sum"] += a
            t2["amount_sum"] += a
            t1["freq"] += 1
            t2["freq"] += 1
    else:
        if G.is_directed():
            for u, v, data in G.edges(data=True):
                a = _edge_amount(data)
                key = (str(u), str(v))
                t = totals.setdefault(key, {"amount_sum": 0.0, "freq": 0})
                t["amount_sum"] += a
                t["freq"] += 1
        else:
            for u, v, data in G.edges(data=True):
                a = _edge_amount(data)
                key1 = (str(u), str(v))
                key2 = (str(v), str(u))
                t1 = totals.setdefault(key1, {"amount_sum": 0.0, "freq": 0})
                t2 = totals.setdefault(key2, {"amount_sum": 0.0, "freq": 0})
                t1["amount_sum"] += a
                t2["amount_sum"] += a
                t1["freq"] += 1
                t2["freq"] += 1

    max_amt = max((v["amount_sum"] for v in totals.values()), default=1.0) or 1.0
    eps = 1e-9
    for (u, v), agg in totals.items():
        norm = float(agg["amount_sum"]) / float(max_amt)
        cost = -math.log(norm + eps)
        DG.add_edge(u, v, cost=cost, amount_sum=float(agg["amount_sum"]), freq=int(agg["freq"]))
    return DG, totals


def _infer_targets(G: nx.Graph, node_id) -> List[str]:
    # Prefer nodes with y==1 or label==1
    y_targets = []
    for n, data in G.nodes(data=True):
        y = data.get("y", data.get("label", 0))
        try:
            if int(y) == 1 and str(n) != str(node_id):
                y_targets.append(str(n))
        except Exception:
            continue
    if y_targets:
        return y_targets[:50]

    # Fallback: top-degree nodes
    deg = dict(G.degree())
    top = sorted(((n, d) for n, d in deg.items() if n != node_id), key=lambda t: t[1], reverse=True)
    return [str(n) for n, _ in top[:50]]


def path_explanations(
    G: nx.Graph,
    node_id,
    k_paths: int = 5,
    max_len: int = 6,
    targets: Optional[List[str]] = None,
) -> List[dict]:
    if node_id not in G:
        return []
    DG, _ = _to_simple_digraph_with_costs(G)
    if str(node_id) not in DG:
        return []

    tgt_list = targets or _infer_targets(G, node_id)
    if not tgt_list:
        return []

    candidates: List[Tuple[float, List[str], List[dict], str]] = []
    for tgt in tgt_list:
        if tgt == node_id or tgt not in DG:
            continue
        try:
            gen = nx.shortest_simple_paths(DG, str(node_id), str(tgt), weight="cost")
        except nx.NetworkXNoPath:
            continue
        count = 0
        for path in gen:
            if max_len is not None and len(path) - 1 > max_len:
                continue
            # path cost and edges
            cost = 0.0
            pedges: List[dict] = []
            ok = True
            for u, v in zip(path[:-1], path[1:]):
                if not DG.has_edge(u, v):
                    ok = False
                    break
                ed = DG[u][v]
                cost += float(ed.get("cost", 0.0))
                pedges.append({
                    "u": u,
                    "v": v,
                    "amount": float(ed.get("amount_sum", 0.0)),
                    "freq": int(ed.get("freq", 1)),
                    "cost": float(ed.get("cost", 0.0)),
                })
            if not ok:
                continue
            candidates.append((cost, [str(n) for n in path], pedges, str(tgt)))
            count += 1
            if count >= k_paths:
                break

    # Take best overall k candidates
    if not candidates:
        return []
    top = nsmallest(k_paths, candidates, key=lambda x: x[0])
    out: List[dict] = []
    for cost, nodes, pedges, tgt in top:
        # Craft rationale with safe fallbacks (avoid 'nan')
        target_txt = str(tgt) if tgt is not None else "target node"
        if target_txt.strip().lower() in {"nan", "none", ""}:
            target_txt = "target node"
        # Try to infer merchant from path nodes metadata (if any)
        mname = None
        try:
            if hasattr(G, "nodes"):
                for n in nodes:
                    nd = G.nodes.get(n, {}) if hasattr(G.nodes, "get") else {}
                    if str(nd.get("type", "")).lower() == "merchant":
                        mname = str(n)
                        break
        except Exception:
            mname = None
        mtxt = mname or "merchant"
        if isinstance(mtxt, str) and mtxt.strip().lower() in {"nan", "none", ""}:
            mtxt = "merchant"
        # Amount vs frequency phrasing
        try:
            amts = [float(e.get("amount", 0.0)) for e in pedges]
            freqs = [int(e.get("freq", 1)) for e in pedges]
            max_amt = max(amts) if amts else 0.0
            high_freq = any(f >= 5 for f in freqs)
            # simple threshold: consider strong if there is any positive amount and not primarily high frequency
            phr_amount = "strong amounts" if (max_amt > 0 and not high_freq) else "high txn freq"
        except Exception:
            phr_amount = "high txn freq"
        rationale = f"connects to {target_txt} via {mtxt} with {phr_amount}"
        out.append({
            "path_nodes": nodes,
            "path_edges": pedges,
            "rationale": rationale,
            "total_cost": float(cost),
        })
    return out


# ------------------------------------------------------------
# Local surrogate explanation using permutation-like importance
# ------------------------------------------------------------


def local_surrogate_explain(
    node_id,
    features_df,
    model,
    top_n: int = 8,
) -> Dict[str, float]:
    import numpy as np
    import pandas as pd

    if node_id not in features_df.index:
        return {}
    # Use numeric columns only
    X = features_df.select_dtypes(include=["number"]).copy()
    x0 = X.loc[[node_id]]

    # Build a local neighborhood: nearest by Euclidean distance in feature space
    v = x0.iloc[0].values.astype(float)
    M = X.values.astype(float)
    # Avoid NaNs
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    dists = np.linalg.norm(M - v, axis=1)
    k = min(500, len(X))
    idx_sorted = np.argsort(dists)[:k]
    Xn = X.iloc[idx_sorted]

    # Baseline prediction
    if hasattr(model, "predict_proba"):
        baseline = float(model.predict_proba(x0)[:, 1][0])
    else:
        baseline = float(model.predict(x0)[0])

    # Directional contribution approximation for ALL features: set each feature
    # to neighborhood median and measure delta. Rank by absolute delta.
    med = Xn.median()
    contrib_all: Dict[str, float] = {}
    for f in X.columns:
        x_alt = x0.copy()
        x_alt[f] = med.get(f, 0.0)
        if hasattr(model, "predict_proba"):
            alt = float(model.predict_proba(x_alt)[:, 1][0])
        else:
            alt = float(model.predict(x_alt)[0])
        contrib_all[f] = float(baseline - alt)

    # Select top_n by absolute contribution
    top = sorted(contrib_all.items(), key=lambda t: -abs(t[1]))[:top_n]
    return {k: float(v) for k, v in top}


# ------------------------------------------------------------
# Case summary
# ------------------------------------------------------------


def summarize_case(G: nx.Graph, node_id, score: float, features_row: dict | None) -> dict:
    why_parts: List[str] = []
    if features_row:
        # pick top absolute features
        items = [(k, float(v)) for k, v in features_row.items() if isinstance(v, (int, float))]
        items.sort(key=lambda t: -abs(t[1]))
        top = items[:3]
        for k, v in top:
            direction = "high" if v > 0 else "low"
            why_parts.append(f"{direction} {k}")
        top_contributors = [{"feature": k, "value": float(v)} for k, v in items[:8]]
    else:
        top_contributors = []

    why = "; ".join(why_parts) if why_parts else "Elevated risk indicators in node features"
    paths = path_explanations(G, node_id, k_paths=5, max_len=6)
    return {
        "why": why,
        "score": float(score),
        "top_contributors": top_contributors,
        "paths": paths,
    }
