from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set

import numpy as np
import pandas as pd
from tqdm import trange


CHANNELS = ["card", "wire", "ach", "cash", "crypto"]
CH_P = [0.55, 0.15, 0.15, 0.1, 0.05]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _pick(rng: np.random.Generator, items: List[str], p: List[float]) -> str:
    return str(rng.choice(items, p=p))


def _amount_normal(rng: np.random.Generator) -> float:
    # Log-normal base distribution for amounts
    return float(np.round(rng.lognormal(mean=3.2, sigma=0.7), 2))


def _amount_small(rng: np.random.Generator) -> float:
    return float(np.round(rng.lognormal(mean=2.2, sigma=0.5), 2))


def _amount_medium(rng: np.random.Generator) -> float:
    return float(np.round(rng.lognormal(mean=3.8, sigma=0.6), 2))


def generate_synthetic(
    n_accounts: int = 5000,
    fraud_rings: int = 10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic edges and labels with a few fraud motifs.

    Returns (df_edges, df_labels)
    - df_edges columns: src, dst, amount, ts, channel, merchant
    - df_labels columns: node_id, y
    """
    rng = _rng(seed)

    # Entities
    accounts = [f"A{i}" for i in range(n_accounts)]
    n_merchants = max(50, n_accounts // 50)
    merchants = [f"M{i}" for i in range(n_merchants)]

    # Time process (increasing seconds)
    t = int(1_700_000_000)
    def next_ts() -> int:
        nonlocal t
        # exponential inter-arrival, min 1 sec
        dt = max(1, int(rng.exponential(scale=5.0)))
        t += dt
        return t

    rows: List[Dict] = []
    labels: Set[str] = set()

    def add_tx(src: str, dst: str, amount: float, channel: str, merchant: str | None = None):
        rows.append({
            "src": src,
            "dst": dst,
            "amount": float(amount),
            "ts": next_ts(),
            "channel": channel,
            "merchant": merchant,
        })

    # Base normal traffic
    base_txs = n_accounts * 8
    for _ in trange(base_txs, desc="normal"):
        src = str(rng.choice(accounts))
        dst = str(rng.choice(accounts))
        if src == dst:
            continue
        amount = _amount_normal(rng)
        ch = _pick(rng, CHANNELS, CH_P)
        mer = str(rng.choice(merchants)) if rng.random() < 0.6 else None
        add_tx(src, dst, amount, ch, mer)

    # Fraud motifs
    for i in range(fraud_rings):
        motif = rng.choice(["smurfing", "fanout", "merchant_collusion"], p=[0.35, 0.35, 0.3])
        if motif == "smurfing":
            # many small incoming to one mule
            mule = str(rng.choice(accounts))
            donors = rng.choice(accounts, size=int(rng.integers(20, 80)), replace=False)
            for d in donors:
                for _ in range(int(rng.integers(2, 8))):
                    add_tx(str(d), mule, _amount_small(rng), _pick(rng, CHANNELS, CH_P), None)
            labels.add(mule)
            labels.update(str(x) for x in donors)
        elif motif == "fanout":
            # one node -> many recipients
            source = str(rng.choice(accounts))
            recips = rng.choice(accounts, size=int(rng.integers(30, 120)), replace=False)
            for r in recips:
                for _ in range(int(rng.integers(1, 4))):
                    add_tx(source, str(r), _amount_small(rng), _pick(rng, CHANNELS, CH_P), None)
            labels.add(source)
            labels.update(str(x) for x in recips[: int(len(recips) * 0.2)])
        else:  # merchant_collusion
            k_merch = int(rng.integers(3, 7))
            ring_merchants = rng.choice(merchants, size=k_merch, replace=False)
            ring_accounts = rng.choice(accounts, size=int(rng.integers(40, 150)), replace=False)
            for a in ring_accounts:
                # each account hits multiple ring merchants repeatedly
                k = int(rng.integers(3, 10))
                for _ in range(k):
                    m = str(rng.choice(ring_merchants))
                    dst = str(rng.choice(accounts))
                    add_tx(str(a), dst, _amount_medium(rng), "card", m)
            labels.update(str(m) for m in ring_merchants)
            labels.update(str(a) for a in ring_accounts[: int(len(ring_accounts) * 0.3)])

    df_edges = pd.DataFrame(rows)
    df_edges = df_edges.sort_values("ts").reset_index(drop=True)
    df_labels = pd.DataFrame({"node_id": sorted(labels), "y": 1}, columns=["node_id", "y"])
    return df_edges, df_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_accounts", type=int, default=5000)
    ap.add_argument("--fraud_rings", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_edges", type=Path, default=Path("data/raw/synth_edges.csv"))
    ap.add_argument("--out_labels", type=Path, default=Path("data/processed/labels.csv"))
    args = ap.parse_args()

    df_edges, df_labels = generate_synthetic(args.n_accounts, args.fraud_rings, args.seed)

    args.out_edges.parent.mkdir(parents=True, exist_ok=True)
    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    df_edges.to_csv(args.out_edges, index=False)
    df_labels.to_csv(args.out_labels, index=False)
    print(f"Wrote {args.out_edges} ({len(df_edges)} rows) and {args.out_labels} ({len(df_labels)} nodes)")


if __name__ == "__main__":
    main()
