from pathlib import Path
import csv
import os

from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings
import pytest


def _write_tiny_csv(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["src", "dst", "amount", "ts", "channel"])
        w.writeheader()
        w.writerow({"src": "A", "dst": "B", "amount": 10.0, "ts": 1, "channel": "card"})
        w.writerow({"src": "B", "dst": "C", "amount": 5.0, "ts": 2, "channel": "wire"})
        w.writerow({"src": "A", "dst": "C", "amount": 2.0, "ts": 3, "channel": "ach"})
    return p


def _write_labels(tmp_path: Path) -> Path:
    p = tmp_path / "labels.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["node_id", "y"])
        w.writeheader()
        for nid in ["A", "B", "C"]:
            w.writerow({"node_id": nid, "y": 1 if nid == "C" else 0})
    return p


def test_api_endpoints(tmp_path):
    client = TestClient(app)
    # Health
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and "version" in j

    # Ingest tiny CSV
    edges_path = _write_tiny_csv(tmp_path)
    r = client.post("/api/v1/ingest", json={"path": str(edges_path), "push_neo4j": False})
    assert r.status_code == 200
    feat_info = r.json()
    assert "features_path" in feat_info

    # Train with labels
    labels_path = _write_labels(tmp_path)
    r = client.post("/api/v1/train", json={"labels_path": str(labels_path)})
    assert r.status_code == 200
    train_out = r.json()
    assert "metrics" in train_out and "run_id" in train_out

    # Score
    r = client.post("/api/v1/score", json={"topk": 10})
    assert r.status_code == 200
    scores = r.json()
    assert "topK" in scores and isinstance(scores["topK"], list)
    assert "topk" in scores and isinstance(scores["topk"], int)
    # legacy alias still present
    assert "results" in scores and isinstance(scores["results"], list)
    node = scores["topK"][0]["node_id"]

    # Explain for that node
    r = client.get(f"/api/v1/explain/{node}")
    assert r.status_code == 200
    exp = r.json()
    assert "why" in exp and "paths" in exp

    # Metrics last should exist after training
    r = client.get("/api/v1/metrics/last")
    assert r.status_code == 200
    ml = r.json()
    assert "created_at" in ml and "metrics" in ml


def test_what_if_local_small(tmp_path):
    client = TestClient(app)
    # Ingest tiny CSV
    edges_path = _write_tiny_csv(tmp_path)
    r = client.post("/api/v1/ingest", json={"path": str(edges_path), "push_neo4j": False})
    assert r.status_code == 200
    # Train with labels
    labels_path = _write_labels(tmp_path)
    r = client.post("/api/v1/train", json={"labels_path": str(labels_path)})
    assert r.status_code == 200
    # What-if local: add incoming edge to B
    payload = {
        "node_id": "B",
        "simulate": [{"op": "add_edge", "src": "A", "dst": "B", "amount": 5}],
        "recompute": "local"
    }
    r = client.post("/api/v1/what-if", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "baseline" in j and "new_score" in j and "delta" in j
    assert j.get("recompute") == "local"
    # Allow small epsilon tolerance
    assert j["new_score"] >= j["baseline"] - 1e-9


def test_auth_enforced_when_token_set(monkeypatch):
    # Force token and clear cached settings
    monkeypatch.setenv("API_AUTH_TOKEN", "secret")
    try:
        get_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    client = TestClient(app)
    r = client.get("/api/v1/health")
    assert r.status_code == 401


def test_auth_ok_with_bearer(monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "secret")
    try:
        get_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    client = TestClient(app)
    r = client.get("/api/v1/health", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200


def test_neo4j_health(monkeypatch):
    # Ensure public for this test
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    try:
        get_settings.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    client = TestClient(app)
    r = client.get("/api/v1/neo4j/health")
    assert r.status_code == 200
    j = r.json()
    assert "ok" in j
    assert ("version" in j) or ("error" in j)


def test_score_constant_flag(monkeypatch):
    # Stub out features and model to produce constant probabilities
    import numpy as np
    import pandas as pd
    from fastapi.testclient import TestClient
    import app.api.v1 as v1
    from app.main import app

    df = pd.DataFrame({"f1": [0.0, 0.0], "f2": [0.0, 0.0]}, index=["A", "B"])  # all zeros

    class DummyModel:
        def predict_proba(self, X):
            n = len(X)
            return np.tile(np.array([[0.5, 0.5]]), (n, 1))

    monkeypatch.setattr(v1, "_read_features", lambda path: df)
    monkeypatch.setattr(v1, "load_model_artifacts", lambda p: (DummyModel(), ["f1", "f2"], {"f1": 0.0, "f2": 0.0}))

    client = TestClient(app)
    r = client.post("/api/v1/score", json={"topk": 10})
    assert r.status_code == 200
    j = r.json()
    assert j.get("constant_scores") is True
