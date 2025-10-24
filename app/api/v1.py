from pathlib import Path
import json
from typing import Optional, List, Iterable, Tuple, Dict

import networkx as nx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from loguru import logger

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.builder import (
    load_graph,
    save_graph,
    load_edges,
    build_nx_graph,
    to_neo4j,
    neo4j_healthcheck,
)
from app.graph.features import ego_features, compute_node_features, persist_node_features, load_node_features, compute_local_subset_features
from app.graph.explain import path_explanations, local_surrogate_explain, summarize_case
from app.ml.train_lgbm import train_lightgbm, persist_artifacts, _read_features, _read_labels
from app.ml.predict import load_model as load_model_artifacts, score_nodes, align_features
from app.storage.sqlite import init_db, create_run, create_metric_run, get_last_metric_run
import math


log = get_logger("api.v1")
_graph: Optional[nx.Graph] = None


def get_bearer_dep():
    sec = HTTPBearer(auto_error=False)

    async def _auth(credentials: HTTPAuthorizationCredentials = Depends(sec)):
        settings = get_settings()
        token = getattr(settings, "api_auth_token", None)
        if not token:
            return True  # open API
        if not credentials or credentials.credentials != token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return True

    return _auth


router = APIRouter(prefix="/api/v1", tags=["v1"], dependencies=[Depends(get_bearer_dep())])


def _ensure_graph() -> nx.Graph:
    global _graph
    if _graph is None:
        settings = get_settings()
        graph_path = Path(settings.graph_path)
        if not graph_path.exists():
            logger.warning("Graph not found at {}", graph_path)
            _graph = nx.MultiDiGraph()
        else:
            _graph = load_graph(graph_path)
    return _graph


@router.get("/health")
async def health():
    settings = get_settings()
    g = _ensure_graph()
    return {"status": "ok", "version": settings.app_version, "nodes": g.number_of_nodes(), "edges": g.number_of_edges()}


@router.get("/auth/status")
async def auth_status():
    settings = get_settings()
    return {"protected": bool(getattr(settings, "api_auth_token", ""))}


class IngestBody(BaseModel):
    path: str
    push_neo4j: bool = False


@router.post("/ingest")
async def ingest(body: IngestBody):
    settings = get_settings()
    edges = load_edges(body.path)
    G = build_nx_graph(edges)
    # Persist graph for later APIs
    save_graph(G, Path(settings.graph_path))
    # Optionally push to Neo4j
    if body.push_neo4j:
        to_neo4j(edges, ensure_indices=True)
    # Compute and persist features
    df_feat, meta = compute_node_features(G)
    out_path = persist_node_features(df_feat)

    # refresh global cache
    global _graph
    _graph = G

    return {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "features_path": str(out_path),
        "n_features": int(df_feat.shape[1]),
    }


@router.get("/neo4j/health")
async def api_neo4j_health():
    settings = get_settings()
    if not (settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password):
        return {"ok": False, "error": "Neo4j not configured"}
    try:
        payload = neo4j_healthcheck()
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    return payload


class TrainBody(BaseModel):
    labels_path: str


def _clean_json_numbers(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, dict):
        return {k: _clean_json_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json_numbers(v) for v in obj]
    return obj


@router.post("/train")
async def train(body: TrainBody):
    settings = get_settings()
    # Load features and labels
    feat_path = Path(settings.model_dir) / "features.parquet"
    X = _read_features(feat_path)
    y = _read_labels(Path(body.labels_path))
    # Align
    common = X.index.intersection(y.index)
    if len(common) == 0:
        raise HTTPException(status_code=400, detail="No overlapping nodes between features and labels")
    X = X.loc[common]
    y = y.loc[common]

    # Train/val split
    from app.ml.dataset import make_splits

    df_lbl = pd.DataFrame({"y": y})
    idx_train, idx_val, _ = make_splits(df_lbl, stratify=True, test_size=0.2, val_size=0.1)
    booster, metrics = train_lightgbm(X.loc[idx_train], y.loc[idx_train], X.loc[idx_val], y.loc[idx_val])
    metrics = _clean_json_numbers(metrics)
    artifacts = persist_artifacts(booster, list(X.columns), metrics, out_dir=Path(settings.model_dir))
    # Persist feature medians alongside artifacts
    try:
        med = X.loc[idx_train].median(numeric_only=True)
        med_dict = {str(k): float(v) for k, v in med.items()}
        med_path = Path(settings.model_dir) / "medians.json"
        with med_path.open("w") as f:
            json.dump({"feature_medians": med_dict}, f, indent=2)
    except Exception as e:
        log.warning("Failed to persist medians.json: {}", e)

    # Create a run row
    await init_db()
    run_id = await create_run(model_version=settings.app_version, notes={"artifacts": artifacts})
    # Persist metric run metadata
    try:
        await create_metric_run(dataset_name=str(body.labels_path), metrics=metrics)
    except Exception as e:
        log.warning("Failed to persist metric run: {}", e)
    return {"run_id": run_id, "metrics": metrics, "artifacts": artifacts}


@router.get("/metrics/last")
async def metrics_last():
    row = await get_last_metric_run()
    if not row:
        return None
    # row may be ORM or dict; normalize
    if isinstance(row, dict):
        return {
            "id": row.get("id"),
            "created_at": row.get("created_at"),
            "dataset_name": row.get("dataset_name"),
            "metrics": row.get("metrics"),
        }
    return {
        "id": getattr(row, "id", None),
        "created_at": getattr(row, "created_at", None),
        "dataset_name": getattr(row, "dataset_name", None),
        "metrics": getattr(row, "metrics", None),
    }

# Re-export for main app wiring
api_router = router


class ScoreBody(BaseModel):
    topk: int = 1000


@router.post("/score")
async def score(body: ScoreBody):
    settings = get_settings()
    # Load features
    feat_path = Path(settings.model_dir) / "features.parquet"
    try:
        X = _read_features(feat_path)
        # Load model + artifacts
        model, feature_list, medians = load_model_artifacts(Path(settings.model_dir))
    except Exception:
        raise HTTPException(status_code=409, detail="No features/model in cache. Run /ingest then /train.")

    df_scores, info = score_nodes(X, model, feature_list or list(X.columns), medians)
    topk = int(min(body.topk, len(df_scores)))
    # Build normalized response payload
    ser = df_scores["score"].sort_values(ascending=False).head(topk)
    items = [{"node_id": str(i), "score": float(s)} for i, s in ser.items()]
    payload = {
        "topK": items,
        "topk": int(topk),
        "count": int(len(df_scores)),
        "results": items,  # legacy alias
        "constant_scores": bool(info.get("constant_scores", False)),
    }
    return payload


@router.get("/case/{node_id}")
async def case(node_id: str):
    g = _ensure_graph()
    if node_id not in g:
        raise HTTPException(status_code=404, detail="node not found")
    # Score
    settings = get_settings()
    feat_path = Path(settings.model_dir) / "features.parquet"
    X = _read_features(feat_path)
    if node_id not in X.index:
        raise HTTPException(status_code=404, detail="features not found for node")
    model, feat_list, medians = load_model_artifacts(Path(settings.model_dir))
    row = X.loc[[node_id]]
    if feat_list:
        row = align_features(row, feat_list, medians or {})
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(row)[:, 1][0])
    else:
        score = float(model.predict(row)[0])

    # Neighbors summary
    neighbors = list(g.neighbors(node_id)) if node_id in g else []
    ego = ego_features(g, node_id, radius=1)
    return {
        "node_id": node_id,
        "score": score,
        "neighbors": {"count": len(neighbors), "nodes": [str(n) for n in neighbors[:50]]},
        "ego": ego,
    }


@router.get("/explain/{node_id}")
async def explain(node_id: str):
    g = _ensure_graph()
    if node_id not in g:
        raise HTTPException(status_code=404, detail="node not in graph")
    settings = get_settings()
    feat_path = Path(settings.model_dir) / "features.parquet"
    X = _read_features(feat_path)
    if node_id not in X.index:
        raise HTTPException(status_code=404, detail="features not found for node")
    model, feat_list, medians = load_model_artifacts(Path(settings.model_dir))
    # Local surrogate
    Xalign = X.copy()
    if feat_list:
        Xalign = align_features(Xalign, feat_list, medians or {})
    contrib = local_surrogate_explain(node_id, Xalign, model, top_n=8)
    # Paths and summary
    case = summarize_case(g, node_id, score=0.0, features_row=contrib)
    case["paths"] = path_explanations(g, node_id, k_paths=5, max_len=6)
    return case


class WhatIfOp(BaseModel):
    op: str
    src: Optional[str] = None
    dst: Optional[str] = None
    amount: Optional[float] = None


class WhatIfBody(BaseModel):
    node_id: str
    simulate: List[WhatIfOp]
    recompute: str = "local"  # local or global


@router.post("/what-if")
async def what_if(body: WhatIfBody):
    g = _ensure_graph().copy()
    # Baseline score
    settings = get_settings()
    feat_path = Path(settings.model_dir) / "features.parquet"
    X = _read_features(feat_path)
    if body.node_id not in X.index:
        raise HTTPException(status_code=404, detail="features not found for node")
    model, feat_list, medians = load_model_artifacts(Path(settings.model_dir))
    x0 = X.loc[[body.node_id]]
    if feat_list:
        x0 = align_features(x0, feat_list, medians or {})
    if hasattr(model, "predict_proba"):
        base_score = float(model.predict_proba(x0)[:, 1][0])
    else:
        base_score = float(model.predict(x0)[0])

    if body.recompute == "local":
        # Build a 2-hop ego subgraph capped to 1000 nodes
        try:
            ego_all = nx.ego_graph(g, body.node_id, radius=2)
        except Exception:
            raise HTTPException(status_code=404, detail="node not in graph")
        nodes = list(ego_all.nodes())
        if len(nodes) > 1000:
            nodes = nodes[:1000]
        Gs = g.subgraph(nodes).copy()
        # Apply ops on ego subgraph
        for op in body.simulate:
            if op.op == "add_edge" and op.src and op.dst:
                a = float(op.amount or 1.0)
                Gs.add_edge(str(op.src), str(op.dst), amount=a)
            elif op.op == "remove_edge" and op.src and op.dst:
                u, v = str(op.src), str(op.dst)
                try:
                    if isinstance(Gs, nx.MultiDiGraph) and Gs.has_edge(u, v):
                        # remove one occurrence
                        keys = list(Gs[u][v].keys())
                        if keys:
                            Gs.remove_edge(u, v, keys[0])
                    elif Gs.has_edge(u, v):
                        Gs.remove_edge(u, v)
                except Exception:
                    pass
        # Compute subset features on local graph
        subset = compute_local_subset_features(Gs, body.node_id)
        features = feat_list or list(X.columns)
        med = X.median(numeric_only=True)
        vec = {}
        n_used = 0
        for f in features:
            if f in subset:
                vec[f] = float(subset[f])
                n_used += 1
            else:
                # prefer persisted medians if available
                vec[f] = float((medians or {}).get(f, med.get(f, 0.0)))
        from pandas import DataFrame
        x1 = DataFrame([vec], index=[body.node_id])
        if hasattr(model, "predict_proba"):
            new_score = float(model.predict_proba(x1)[:, 1][0])
        else:
            new_score = float(model.predict(x1)[0])
        return {
            "node_id": body.node_id,
            "baseline": base_score,
            "new_score": new_score,
            "delta": float(new_score - base_score),
            "recompute": "local",
            "used_features": int(n_used),
            "filled_from_median": int(len(features) - n_used),
        }
    else:
        # Global recompute
        for op in body.simulate:
            if op.op == "add_edge" and op.src and op.dst:
                a = float(op.amount or 1.0)
                try:
                    g.add_edge(str(op.src), str(op.dst), amount=a)
                except Exception:
                    pass
            elif op.op == "remove_edge" and op.src and op.dst:
                try:
                    g.remove_edge(str(op.src), str(op.dst))
                except Exception:
                    pass
        df_feat_new, _ = compute_node_features(g)
        x1 = df_feat_new.loc[[body.node_id]] if body.node_id in df_feat_new.index else None
        if x1 is None:
            raise HTTPException(status_code=400, detail="node_id missing after simulation")
        if feat_list:
            x1 = align_features(x1, feat_list, medians or {})
        if hasattr(model, "predict_proba"):
            new_score = float(model.predict_proba(x1)[:, 1][0])
        else:
            new_score = float(model.predict(x1)[0])

        return {
            "node_id": body.node_id,
            "baseline": base_score,
            "new_score": new_score,
            "delta": float(new_score - base_score),
            "recompute": "global",
        }


@router.get("/debug/routes")
async def debug_routes():
    """List registered routes for the v1 router."""
    out: List[Dict[str, object]] = []
    try:
        for r in router.routes:
            path = getattr(r, "path", None)
            methods = list(getattr(r, "methods", set()))
            if path:
                out.append({"path": path, "methods": methods})
    except Exception:
        pass
    # Sort by path for readability
    out.sort(key=lambda x: str(x.get("path", "")))
    return out
