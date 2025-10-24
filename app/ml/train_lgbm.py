from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from app.core.config import get_settings
from app.core.logging import configure_json_logger, get_logger
from app.ml.dataset import graph_to_node_dataframe, train_val_split
from app.ml.dataset import make_splits
from app.storage.sqlite import create_metric_run, init_db


def _scale_pos_weight(y: pd.Series, class_weight: str | None) -> float | None:
    if class_weight != "balanced":
        return None
    y = y.astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        return None
    return max(1.0, neg / max(pos, 1))


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    n = min(k, y_true.shape[0])
    if n == 0:
        return 0.0
    idx = np.argsort(-y_score)[:n]
    return float(y_true[idx].sum() / n)


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    class_weight: str | None = "balanced",
    num_leaves: int = 63,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
) -> Tuple[lgb.Booster, Dict[str, object]]:
    """Train LightGBM binary classifier and compute validation metrics.

    Returns (booster, report) with metrics and important_features.
    """
    log = get_logger("ml.train")

    spw = _scale_pos_weight(y, class_weight)
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "feature_pre_filter": False,
        "verbose": -1,
    }
    if spw is not None:
        params["scale_pos_weight"] = float(spw)

    train_set = lgb.Dataset(X, label=y.astype(int))
    val_set = lgb.Dataset(X_val, label=y_val.astype(int), reference=train_set)

    booster = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        num_boost_round=n_estimators,
    )

    # Validation predictions and metrics
    y_val_true = y_val.astype(int).to_numpy()
    y_val_score = booster.predict(X_val)
    try:
        roc = roc_auc_score(y_val_true, y_val_score)
    except Exception:
        roc = float("nan")
    try:
        pr_auc = average_precision_score(y_val_true, y_val_score)
    except Exception:
        pr_auc = float("nan")
    try:
        brier = brier_score_loss(y_val_true, y_val_score)
    except Exception:
        brier = float("nan")

    prec_at = {k: _precision_at_k(y_val_true, y_val_score, k) for k in (100, 500, 1000)}

    # Feature importances (gain)
    gains = booster.feature_importance(importance_type="gain")
    fnames = booster.feature_name()
    feat_gain = sorted(zip(fnames, gains.tolist()), key=lambda t: t[1], reverse=True)
    important_features = [{"feature": f, "gain": float(g)} for f, g in feat_gain[:20]]

    import time as _time
    report: Dict[str, object] = {
        "roc_auc": None if (isinstance(roc, float) and (roc != roc)) else float(roc),
        "pr_auc": None if (isinstance(pr_auc, float) and (pr_auc != pr_auc)) else float(pr_auc),
        "brier": None if (isinstance(brier, float) and (brier != brier)) else float(brier),
        "precision_at": {str(k): float(v) for k, v in prec_at.items()},
        "precision_at_100": float(prec_at.get(100, 0.0)),
        "precision_at_500": float(prec_at.get(500, 0.0)),
        "precision_at_1000": float(prec_at.get(1000, 0.0)),
        "important_features": important_features,
        "trained_at": int(_time.time()),
    }
    log.info(
        "Validation ROC-AUC={} PR-AUC={} Brier={}",
        report.get("roc_auc"),
        report.get("pr_auc"),
        report.get("brier"),
    )
    return booster, report


def persist_artifacts(
    booster: lgb.Booster,
    features: List[str],
    metrics: Dict[str, object],
    out_dir: Path | None = None,
) -> Dict[str, str]:
    settings = get_settings()
    out = Path(out_dir) if out_dir else Path(settings.model_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "model.joblib"
    feats_path = out / "features.json"
    metrics_path = out / "metrics.json"

    joblib.dump(booster, model_path)
    with open(feats_path, "w") as f:
        json.dump({"features": list(features)}, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return {
        "model": str(model_path),
        "features": str(feats_path),
        "metrics": str(metrics_path),
    }


# Backward-compatible helper used by scripts/score_demo.py
def train_baseline(G: nx.Graph, out_path: Path) -> Path:
    X, y = graph_to_node_dataframe(G)
    ds = train_val_split(X, y)
    booster, report = train_lightgbm(ds.X_train, ds.y_train, ds.X_val, ds.y_val)
    joblib.dump(booster, out_path)
    return out_path


def _read_features(path: Path) -> pd.DataFrame:
    # Try Parquet, then CSV fallback
    if path.exists():
        try:
            df = pd.read_parquet(path)
            if df.index.name is None:
                df.index.name = "node_id"
            return df
        except Exception:
            pass
    p_csv = path if path.suffix == ".csv" else path.with_suffix(".csv")
    if p_csv.exists():
        df = pd.read_csv(p_csv)
        if "node_id" in df.columns:
            df = df.set_index("node_id")
        return df
    raise FileNotFoundError(path)


def _read_labels(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "node_id" in df.columns:
        df = df.set_index("node_id")
    elif "node" in df.columns:
        df = df.set_index("node")
    else:
        raise ValueError("Labels CSV must include a 'node_id' (or 'node') column")
    label_col = "y" if "y" in df.columns else ("label" if "label" in df.columns else None)
    if label_col is None:
        raise ValueError("Labels CSV must include a 'y' or 'label' column")
    y = df[label_col].astype(int)
    y.name = "y"
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True, help="Path to features.parquet")
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels CSV (node_id,y)")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (defaults to MODEL_DIR)")
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    configure_json_logger()
    log = get_logger("ml.train")

    X = _read_features(args.features)
    y = _read_labels(args.labels)
    # Align indices
    common = X.index.intersection(y.index)
    if len(common) == 0:
        raise ValueError("No overlapping node_ids between features and labels")
    X = X.loc[common]
    y = y.loc[common]

    # Split into train/val using dataset utility
    df_lbl = pd.DataFrame(index=common)
    df_lbl["y"] = y
    from app.ml.dataset import make_splits

    idx_train, idx_val, _ = make_splits(df_lbl, stratify=True, test_size=0.2, val_size=0.1)
    booster, report = train_lightgbm(
        X.loc[idx_train], y.loc[idx_train], X.loc[idx_val], y.loc[idx_val],
        num_leaves=args.num_leaves, n_estimators=args.n_estimators, learning_rate=args.learning_rate,
    )

    artifacts = persist_artifacts(booster, list(X.columns), report, out_dir=args.out)
    # Compute and persist feature medians from training fold
    try:
        med = X.loc[idx_train].median(numeric_only=True)
        med_dict = {str(k): float(v) for k, v in med.items()}
        out_dir = Path(args.out) if args.out else Path(get_settings().model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        med_path = out_dir / "medians.json"
        with med_path.open("w") as f:
            json.dump({"feature_medians": med_dict}, f, indent=2)
        log.info("Saved feature medians to {}", str(med_path))
    except Exception as e:
        log.warning("Failed to persist medians.json: {}", e)

    log.info("Saved model to {}", artifacts["model"])
    log.info("Saved features to {} and metrics to {}", artifacts["features"], artifacts["metrics"])

    # Persist MetricRun even when training via CLI
    try:
        import asyncio
        asyncio.run(init_db())
        asyncio.run(create_metric_run(dataset_name=str(args.labels), metrics=report))
    except Exception as e:
        log.warning("Failed to persist MetricRun via CLI: {}", e)


if __name__ == "__main__":
    main()
