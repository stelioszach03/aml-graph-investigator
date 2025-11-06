from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.dataset import graph_to_node_dataframe
from app.storage.sqlite import (
    sqlite_path_from_url,
    write_score_run_summary,
)


log = get_logger("ml.predict")


def align_features(df_features: pd.DataFrame, feature_list: list[str], global_medians: Dict[str, float]) -> pd.DataFrame:
    """Align features to the given list and fill NaN/Inf with medians.

    - Keeps column order matching `feature_list`.
    - For missing columns, uses provided `global_medians` or 0.0 fallback.
    - Replaces inf/-inf with NaN, then fills with median per feature.
    """
    import numpy as _np
    X = pd.DataFrame(index=df_features.index)
    for f in feature_list:
        if f in df_features.columns:
            col = df_features[f]
            # compute per-feature median fallback if not provided
            med = global_medians.get(f, float(col.median()) if col.notna().any() else 0.0)
            X[f] = col.replace([_np.inf, -_np.inf], _np.nan).fillna(med).astype(float)
        else:
            X[f] = float(global_medians.get(f, 0.0))
    return X


def load_model(model_dir: Path | str) -> Tuple[object, List[str], Dict[str, float]]:
    """Load model and feature list from a directory containing artifacts.

    Expects files:
      - model.joblib
      - features.json with {"features": [...]}.

    For backward compatibility, if `model_dir` points directly to a .joblib file,
    loads it and returns an empty feature list (caller must align separately).
    """
    p = Path(model_dir)
    medians: Dict[str, float] = {}
    if p.is_file() and p.suffix in {".joblib", ".pkl"}:
        model = joblib.load(p)
        return model, [], medians
    model = joblib.load(p / "model.joblib")
    feats_path = p / "features.json"
    if feats_path.exists():
        with open(feats_path) as f:
            obj = json.load(f)
        features = list(obj.get("features", []))
    else:
        features = []
    # Try to load persisted feature medians
    med_path = p / "medians.json"
    if med_path.exists():
        try:
            with open(med_path) as f:
                mobj = json.load(f)
            # support either {"feature_medians": {...}} or a flat dict
            med_src = mobj.get("feature_medians") if isinstance(mobj, dict) else None
            if isinstance(med_src, dict):
                medians = {str(k): float(v) for k, v in med_src.items()}
            elif isinstance(mobj, dict):
                medians = {str(k): float(v) for k, v in mobj.items() if isinstance(v, (int, float))}
        except Exception as e:
            log.warning("Failed to read medians.json: {}", e)
            medians = {}
    return model, features, medians


def score_nodes(df_features: pd.DataFrame, model, feature_list: List[str] | None = None,
                medians: Dict[str, float] | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Score nodes, aligning to the model's feature list if provided.

    Returns a DataFrame indexed by node id with columns: score (0..1), rank (1=best)
    """
    # Align to feature list using medians (fallback to local medians if missing)
    if feature_list:
        med = medians or {}
        # If medians not provided, compute from df_features
        if not med:
            try:
                m = df_features[feature_list].median(numeric_only=True)
                med = {str(k): float(v) for k, v in m.items()}
            except Exception:
                med = {}
        X = align_features(df_features, list(feature_list), med)
    else:
        X = df_features.copy()
    # predict probabilities
    # Predict probabilities of positive class
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        try:
            prob = prob[:, 1]
        except Exception:
            prob = np.asarray(prob).reshape(-1)
    else:
        # LightGBM Booster returns probability for binary objective
        prob = model.predict(X)
        prob = np.asarray(prob).reshape(-1)
        prob = np.clip(prob, 0.0, 1.0)
    scores = pd.Series(prob, index=X.index, name="score").astype(float)
    # ranks: 1-based descending order, stable for ties
    order = (-scores.values).argsort(kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    df = pd.DataFrame({"score": scores, "rank": ranks}, index=X.index)
    df.index.name = X.index.name or "node_id"
    df = df.sort_values("rank")
    # Constant-score detection
    try:
        smin = float(np.min(scores)) if len(scores) > 0 else 0.0
        smax = float(np.max(scores)) if len(scores) > 0 else 0.0
        const = (smax - smin) < 1e-9
        if const:
            log.warning("All predicted scores are identical: min={} max={}", smin, smax)
    except Exception:
        const = False
    return df, {"constant_scores": bool(const)}


def _compute_topk_stats(df_scores: pd.DataFrame, ks: Iterable[int]) -> dict:
    res = {}
    if df_scores.empty:
        return {int(k): (0, 0.0, 0.0, 0.0) for k in ks}
    scores_sorted = df_scores.sort_values("score", ascending=False)["score"].to_numpy()
    n_total = len(scores_sorted)
    for k in ks:
        n = int(min(k, n_total))
        if n == 0:
            res[int(k)] = (0, 0.0, 0.0, 0.0)
        else:
            topk = scores_sorted[:n]
            res[int(k)] = (n, float(topk.mean()), float(topk.max()), float(topk.min()))
    return res


def write_score_summary(df_scores: pd.DataFrame, run_id: str | None = None, ks: Iterable[int] = (100, 500, 1000)) -> None:
    """Persist top-K score stats into SQLite.

    Uses settings.sqlite_url to locate DB file (sync engine). Creates a summary
    table if not present and writes one row per K.
    """
    settings = get_settings()
    db_path = sqlite_path_from_url(settings.sqlite_url)
    if run_id is None:
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    ts = int(datetime.now(timezone.utc).timestamp())
    stats = _compute_topk_stats(df_scores, ks)
    write_score_run_summary(db_path, run_id, ts, stats)


# Backward-compatible helper for existing script
def predict_graph_nodes(G: nx.Graph, model_path: Path) -> pd.DataFrame:
    model = joblib.load(model_path)
    X, _ = graph_to_node_dataframe(G)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X)
    df = pd.DataFrame({"node": X.index.astype(str), "score": proba})
    return df


def load_model_artifacts(model_dir: Path | str):
    """
    Back-compat wrapper: returns (model, feature_list).
    New code should use load_model(model_dir) which returns (model, feature_list, medians).
    """
    loaded = load_model(model_dir)
    if isinstance(loaded, tuple) and len(loaded) >= 2:
        return loaded[0], loaded[1]
    raise RuntimeError("Unexpected load_model() return")
