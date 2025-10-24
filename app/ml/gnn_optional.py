from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from app.core.logging import configure_json_logger, get_logger
from app.ml.train_lgbm import train_lightgbm, persist_artifacts
from app.ml.dataset import make_splits


log = get_logger("ml.gnn")


try:
    import torch
    from torch.optim import SparseAdam
    from torch_geometric.nn.models import Node2Vec

    PYG_AVAILABLE = True
except Exception:  # pragma: no cover - environment optional
    torch = None  # type: ignore
    Node2Vec = None  # type: ignore
    PYG_AVAILABLE = False


def _require_pyg() -> None:
    if not PYG_AVAILABLE:
        raise ImportError("PyG not installed")


def _pick(colmap: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in colmap}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _load_edges_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    src_col = _pick(df.columns, ["src", "source", "from"]) or "src"
    dst_col = _pick(df.columns, ["dst", "dest", "to", "target"]) or "dst"
    src = df[src_col].astype(str).tolist()
    dst = df[dst_col].astype(str).tolist()
    nodes = sorted(set(src) | set(dst))
    idx = {n: i for i, n in enumerate(nodes)}
    edges = np.array([[idx[s], idx[d]] for s, d in zip(src, dst)], dtype=np.int64)
    return nodes, edges


def compute_node2vec_embeddings(
    edges_path: Path,
    dim: int = 64,
    walk_length: int = 20,
    context_size: int = 10,
    walks_per_node: int = 20,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.01,
    p: float = 1.0,
    q: float = 1.0,
) -> Tuple[List[str], np.ndarray]:
    _require_pyg()
    assert Node2Vec is not None and torch is not None
    nodes, edges = _load_edges_csv(edges_path)
    num_nodes = len(nodes)
    if edges.size == 0 or num_nodes == 0:
        return nodes, np.zeros((0, dim), dtype=np.float32)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Node2Vec(
        edge_index,
        embedding_dim=dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=p,
        q=q,
        num_negative_samples=1,
        sparse=True,
        num_nodes=num_nodes,
    ).to(device)

    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)
    opt = SparseAdam(list(model.parameters()), lr=lr)
    for epoch in range(epochs):
        total = 0.0
        for pos_rw, neg_rw in loader:
            opt.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            opt.step()
            total += float(loss.item())
        log.info("Node2Vec epoch {} loss {:.4f}", epoch + 1, total)

    emb = model.embedding.weight.data.detach().cpu().numpy().astype(np.float32)
    return nodes, emb


def _save_embeddings(out_path: Path, nodes: List[str], emb: np.ndarray) -> Tuple[Path, Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)
    nodes_path = out_path.with_suffix(".nodes.csv")
    pd.DataFrame({"node_id": nodes}).to_csv(nodes_path, index=False)
    return out_path, nodes_path


def _maybe_train_lgbm_from_embeddings(nodes: List[str], emb: np.ndarray, labels_path: Path, out_dir: Path) -> None:
    dfX = pd.DataFrame(emb, index=pd.Index(nodes, name="node_id"), columns=[f"emb_{i}" for i in range(emb.shape[1])])
    dfy = pd.read_csv(labels_path)
    if "node_id" in dfy.columns:
        dfy = dfy.set_index("node_id")
    elif "node" in dfy.columns:
        dfy = dfy.set_index("node")
    label_col = "y" if "y" in dfy.columns else ("label" if "label" in dfy.columns else None)
    if label_col is None:
        raise ValueError("labels CSV must include a 'y' or 'label' column")
    y = dfy[label_col].astype(int)
    common = dfX.index.intersection(y.index)
    if len(common) == 0:
        raise ValueError("No overlapping node_ids between embeddings and labels")
    dfX = dfX.loc[common]
    y = y.loc[common]
    df_lbl = pd.DataFrame({"y": y})
    idx_train, idx_val, _ = make_splits(df_lbl, stratify=True, test_size=0.2, val_size=0.1)
    booster, report = train_lightgbm(dfX.loc[idx_train], y.loc[idx_train], dfX.loc[idx_val], y.loc[idx_val])
    artifacts = persist_artifacts(booster, list(dfX.columns), report, out_dir=out_dir)
    log.info("Trained LightGBM on embeddings. ROC-AUC={:.4f}", report.get("roc_auc", float("nan")))
    log.info("Saved model={} features={} metrics={}", artifacts["model"], artifacts["features"], artifacts["metrics"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, required=True, help="Edges CSV with columns [src,dst]")
    parser.add_argument("--out", type=Path, required=True, help="Path to save embeddings .npy")
    parser.add_argument("--method", type=str, default="node2vec", choices=["node2vec", "sage"], help="Embedding method")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--context-size", type=int, default=10)
    parser.add_argument("--walks-per-node", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--labels", type=Path, default=None, help="Optional labels CSV to train LightGBM stage")
    args = parser.parse_args()

    configure_json_logger()

    if not PYG_AVAILABLE:
        raise ImportError("PyG not installed")

    if args.method == "node2vec":
        nodes, emb = compute_node2vec_embeddings(
            args.edges,
            dim=args.dim,
            walk_length=args.walk_length,
            context_size=args.context_size,
            walks_per_node=args.walks_per_node,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    else:
        # Minimal path: fall back to Node2Vec if SAGE requested but not implemented
        log.info("SAGE method not implemented; falling back to Node2Vec")
        nodes, emb = compute_node2vec_embeddings(
            args.edges,
            dim=args.dim,
            walk_length=args.walk_length,
            context_size=args.context_size,
            walks_per_node=args.walks_per_node,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    _save_embeddings(args.out, nodes, emb)
    log.info("Saved embeddings to {} (shape {}x{})", args.out, emb.shape[0], emb.shape[1] if emb.size else 0)

    if args.labels is not None:
        _maybe_train_lgbm_from_embeddings(nodes, emb, args.labels, out_dir=args.out.parent)


if __name__ == "__main__":
    main()
