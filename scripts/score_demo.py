from __future__ import annotations

from _path_guard import *  # noqa: F401

import os
import argparse
from pathlib import Path

import pandas as pd

from app.core.logging import configure_json_logger, get_logger
from app.core.config import get_settings
from app.graph.features import load_node_features
from app.ml.predict import load_model, score_nodes, write_score_summary


def ensure_model(model_dir: Path, labels_path: Path) -> None:
    # Deprecated: training handled separately in pipeline; keep for compatibility
    return


def main():
    configure_json_logger()
    log = get_logger("scripts.score")
    ap = argparse.ArgumentParser(description="Score nodes and print top suspicious")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()
    settings = get_settings()
    model_dir = Path(settings.model_dir)

    # Load model (supports new and old signatures)
    loaded = load_model(model_dir)
    if isinstance(loaded, tuple) and len(loaded) >= 2:
        model = loaded[0]
        feature_list = loaded[1]
        medians = loaded[2] if len(loaded) >= 3 else None
    else:
        raise RuntimeError("Unexpected load_model() return")

    # Load features matrix from disk
    df = load_node_features()

    # Score
    df_scores, info = score_nodes(df, model, feature_list or list(df.columns), medians)
    df_scores = df_scores.sort_values("score", ascending=False)
    topk = int(os.getenv("TOPK", str(args.topk)))
    top = df_scores.head(topk)

    # Print compact topK summary
    print(top.head(10).to_string())
    print({"constant_scores": bool(info.get("constant_scores", False))})


if __name__ == "__main__":
    main()
