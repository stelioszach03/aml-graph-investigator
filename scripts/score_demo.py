from __future__ import annotations

import os, sys
here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, ".."))
if repo not in sys.path:
    sys.path.insert(0, repo)

import argparse
from pathlib import Path

import pandas as pd

from app.core.logging import configure_json_logger, get_logger
from app.core.config import get_settings
from app.ml.train_lgbm import train_lightgbm, persist_artifacts, _read_features, _read_labels
from app.ml.predict import load_model as load_model_artifacts, score_nodes, write_score_summary


def ensure_model(model_dir: Path, labels_path: Path) -> None:
    log = get_logger("scripts.score")
    model_path = model_dir / "model.joblib"
    if model_path.exists():
        return

    feat_path = model_dir / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found at {feat_path}. Run scripts/ingest_demo.py first.")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found at {labels_path}. Run scripts/generate_synth.py or provide labels.")

    X = _read_features(feat_path)
    y = _read_labels(labels_path)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    # Split
    from app.ml.dataset import make_splits

    idx_train, idx_val, _ = make_splits(pd.DataFrame({"y": y}), stratify=True, test_size=0.2, val_size=0.1)
    booster, report = train_lightgbm(X.loc[idx_train], y.loc[idx_train], X.loc[idx_val], y.loc[idx_val])
    artifacts = persist_artifacts(booster, list(X.columns), report, out_dir=model_dir)
    log.info("Trained model and saved artifacts: {}", artifacts)


def main():
    configure_json_logger()
    log = get_logger("scripts.score")
    ap = argparse.ArgumentParser(description="Score nodes and print top suspicious")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--labels", type=Path, default=Path("data/processed/labels.csv"))
    args = ap.parse_args()

    settings = get_settings()
    model_dir = Path(settings.model_dir)

    # Ensure model exists (train if missing)
    ensure_model(model_dir, args.labels)

    # Load features and model
    feat_path = model_dir / "features.parquet"
    X = _read_features(feat_path)
    model, feat_list = load_model_artifacts(model_dir)
    df_scores = score_nodes(X, model, feature_list=feat_list or list(X.columns))
    df_scores = df_scores.sort_values("score", ascending=False)
    top = df_scores.head(args.topk)

    # Print top suspicious nodes
    print("Top suspicious nodes:")
    for i, (nid, row) in enumerate(top.iterrows(), start=1):
        print(f"{i:3d}. {nid}\t{row.score:.4f}")

    # persist a quick score summary to sqlite
    try:
        write_score_summary(df_scores, run_id="score_demo")
    except Exception:
        pass


if __name__ == "__main__":
    main()
