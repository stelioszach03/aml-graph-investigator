from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, List, Tuple

import joblib
import networkx as nx
import pandas as pd
from loguru import logger

from app.core.config import get_settings


# -------------------------
# Domain model
# -------------------------


@dataclass(frozen=True)
class TxEdge:
    src: str
    dst: str
    amount: float
    ts: int
    channel: Optional[str] = None
    merchant: Optional[str] = None


def _normalize_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Remove all internal whitespace
    s = "".join(s.split())
    # Lowercase emails (basic heuristic: contains '@')
    if "@" in s:
        s = s.lower()
    return s


def _pick(obj: dict, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in obj and obj[c] is not None:
            return obj[c]
    # try case-insensitive
    lower = {k.lower(): v for k, v in obj.items()}
    for c in candidates:
        if c.lower() in lower and lower[c.lower()] is not None:
            return lower[c.lower()]
    return None


def _parse_amount(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _parse_ts(v) -> int:
    if v is None:
        return 0
    try:
        # numeric epoch detection (seconds or ms)
        x = float(v)
        if x > 1e12:  # ms
            x = x / 1000.0
        return int(x)
    except Exception:
        # try datetime parsing via pandas
        try:
            ts = pd.to_datetime(v, utc=True, errors="coerce")
            if pd.isna(ts):
                return 0
            # convert to seconds
            return int(ts.view("int64") // 1_000_000_000)
        except Exception:
            return 0


# -------------------------
# Loading
# -------------------------


def load_edges(path: str | Path) -> List[TxEdge]:
    """Load transaction edges from CSV or JSONL.

    Auto-detects by file extension. Deduplicates identical edges and normalizes IDs.
    Supported fields (case-insensitive, with fallbacks):
        - src: src, source, from, sender, account_src
        - dst: dst, dest, to, receiver, account_dst, beneficiary
        - amount: amount, amt, value, weight
        - ts: ts, timestamp, time, datetime, date
        - channel: channel, method, type, medium
        - merchant: merchant, merchant_id, merchant_name, mcc
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    edges: List[TxEdge] = []
    seen: set[Tuple[str, str, float, int, Optional[str], Optional[str]]] = set()

    if p.suffix.lower() == ".csv" or p.suffix.lower() == ".tsv":
        sep = "," if p.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(p, sep=sep)
        for _, row in df.iterrows():
            obj = {str(k): row[k] for k in df.columns}
            src = _normalize_id(_pick(obj, "src", "source", "from", "sender", "account_src"))
            dst = _normalize_id(_pick(obj, "dst", "dest", "to", "receiver", "account_dst", "beneficiary"))
            if not src or not dst:
                continue
            amount = _parse_amount(_pick(obj, "amount", "amt", "value", "weight", "transaction_amount"))
            ts = _parse_ts(_pick(obj, "ts", "timestamp", "time", "datetime", "date"))
            channel = _pick(obj, "channel", "method", "type", "medium")
            merchant = _normalize_id(_pick(obj, "merchant", "merchant_id", "merchant_name", "mcc"))
            key = (src, dst, round(amount, 6), ts, channel if channel is None else str(channel), merchant)
            if key in seen:
                continue
            seen.add(key)
            edges.append(TxEdge(src, dst, float(amount), int(ts), str(channel) if channel is not None else None, merchant))
    elif p.suffix.lower() in {".jsonl", ".ndjson"}:
        import json

        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                src = _normalize_id(_pick(obj, "src", "source", "from", "sender", "account_src"))
                dst = _normalize_id(_pick(obj, "dst", "dest", "to", "receiver", "account_dst", "beneficiary"))
                if not src or not dst:
                    continue
                amount = _parse_amount(_pick(obj, "amount", "amt", "value", "weight", "transaction_amount"))
                ts = _parse_ts(_pick(obj, "ts", "timestamp", "time", "datetime", "date"))
                channel = _pick(obj, "channel", "method", "type", "medium")
                merchant = _normalize_id(_pick(obj, "merchant", "merchant_id", "merchant_name", "mcc"))
                key = (src, dst, round(amount, 6), ts, channel if channel is None else str(channel), merchant)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(TxEdge(src, dst, float(amount), int(ts), str(channel) if channel is not None else None, merchant))
    else:
        raise ValueError(f"Unsupported file extension: {p.suffix}")

    logger.info("Loaded {} edges from {} (deduped={})", len(edges), p, len(seen))
    return edges


# -------------------------
# Graph building
# -------------------------


def build_nx_graph(edges: List[TxEdge]) -> nx.MultiDiGraph:
    """Build a MultiDiGraph with node attributes and totals.

    - Node 'type': 'account' or 'merchant' inferred from presence in merchant field.
    - Node 'node_degree_in' / 'node_degree_out'
    - Node 'total_in_amount' / 'total_out_amount'
    - Adds edges src->dst for every transaction.
      If merchant is present, also adds src->merchant edge.
    """
    G = nx.MultiDiGraph()

    merchant_ids = {e.merchant for e in edges if e.merchant}

    def ensure_node(node_id: str, is_merchant: bool = False):
        if not G.has_node(node_id):
            G.add_node(node_id, type=("merchant" if is_merchant else "account"))
        else:
            # If it was created as account but is merchant, upgrade type
            if is_merchant:
                G.nodes[node_id]["type"] = "merchant"

    for e in edges:
        ensure_node(e.src, False)
        ensure_node(e.dst, e.dst in merchant_ids)
        if e.merchant:
            ensure_node(e.merchant, True)
        # Primary transaction edge account -> account
        G.add_edge(e.src, e.dst, amount=float(e.amount), ts=int(e.ts), channel=e.channel)
        # Optional account -> merchant edge
        if e.merchant:
            G.add_edge(e.src, e.merchant, amount=float(e.amount), ts=int(e.ts), channel=e.channel)

    # Compute degrees and totals
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    for n in G.nodes:
        G.nodes[n]["node_degree_in"] = float(in_deg.get(n, 0))
        G.nodes[n]["node_degree_out"] = float(out_deg.get(n, 0))
        # running totals
        total_in = 0.0
        total_out = 0.0
        for _, _, _, data in G.in_edges(n, data=True, keys=True):
            total_in += float(data.get("amount", 0.0))
        for _, _, _, data in G.out_edges(n, data=True, keys=True):
            total_out += float(data.get("amount", 0.0))
        G.nodes[n]["total_in_amount"] = float(total_in)
        G.nodes[n]["total_out_amount"] = float(total_out)

    logger.info("Built MultiDiGraph: nodes={}, edges={}", G.number_of_nodes(), G.number_of_edges())
    return G


# -------------------------
# Neo4j integration
# -------------------------


def ensure_indices_neo4j(session) -> None:
    """Ensure uniqueness constraints exist for Account.id and Merchant.id using an active session."""
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Merchant) REQUIRE m.id IS UNIQUE")


def to_neo4j(edges: List[TxEdge], ensure_indices: bool = False) -> None:
    settings = get_settings()
    if not (settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password):
        logger.warning("Neo4j not configured; skipping push")
        return
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        logger.error("Neo4j driver not available: {}", e)
        return

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    db = settings.neo4j_database or "neo4j"

    accounts = sorted({e.src for e in edges} | {e.dst for e in edges})
    merchants = sorted({e.merchant for e in edges if e.merchant})

    # Prepare edge payloads
    edge_payload = [
        {"src": e.src, "dst": e.dst, "amount": float(e.amount), "ts": int(e.ts), "channel": e.channel}
        for e in edges
    ]
    edge_to_merchant_payload = [
        {"src": e.src, "merchant": e.merchant, "amount": float(e.amount), "ts": int(e.ts), "channel": e.channel}
        for e in edges if e.merchant
    ]

    with driver.session(database=db) as session:
        if ensure_indices:
            try:
                ensure_indices_neo4j(session)
            except Exception as e:
                logger.warning("Failed to ensure Neo4j indices: {}", e)

        # Create nodes in batches
        if accounts:
            session.run("UNWIND $ids as id MERGE (:Account {id: id})", ids=accounts)
        if merchants:
            session.run("UNWIND $ids as id MERGE (:Merchant {id: id})", ids=merchants)

        # Create relationships (allow multiple transactions)
        if edge_payload:
            session.run(
                """
                UNWIND $rows AS e
                MATCH (a:Account {id: e.src})
                MATCH (b:Account {id: e.dst})
                CREATE (a)-[:TX {amount: e.amount, ts: e.ts, channel: e.channel}]->(b)
                """,
                rows=edge_payload,
            )

        if edge_to_merchant_payload:
            session.run(
                """
                UNWIND $rows AS e
                MATCH (a:Account {id: e.src})
                MATCH (m:Merchant {id: e.merchant})
                CREATE (a)-[:TX {amount: e.amount, ts: e.ts, channel: e.channel}]->(m)
                """,
                rows=edge_to_merchant_payload,
            )

    driver.close()
    logger.info(
        "Pushed {} account nodes, {} merchant nodes and {}+{} TX edges to Neo4j",
        len(accounts), len(merchants), len(edge_payload), len(edge_to_merchant_payload),
    )


def neo4j_healthcheck() -> dict:
    """Return Neo4j server info and plugin availability.

    Returns a dict with keys: ok, name, version, apoc_or_gds (count), components (optional), error on failure.
    """
    settings = get_settings()
    if not (settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password):
        raise RuntimeError("Neo4j not configured")
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        return {"ok": False, "error": f"driver error: {e}"}

    try:
        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        db = settings.neo4j_database or "neo4j"
        payload: dict = {"ok": True}
        with driver.session(database=db) as session:
            # db.info
            try:
                rec = session.run("CALL db.info() YIELD name, version RETURN name, version LIMIT 1").single()
                if rec:
                    payload["name"] = rec.get("name")
                    payload["version"] = rec.get("version")
            except Exception:
                pass
            # components
            try:
                comps = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions").data()
                payload["components"] = comps
                if "version" not in payload and comps:
                    v = comps[0].get("versions")
                    if isinstance(v, list) and v:
                        payload["version"] = v[0]
            except Exception:
                pass
            # procedures with apoc or gds
            try:
                rec2 = session.run(
                    "CALL dbms.procedures() YIELD name WHERE name STARTS WITH 'apoc' OR name STARTS WITH 'gds' RETURN count(*) AS available"
                ).single()
                payload["apoc_or_gds"] = int(rec2["available"]) if rec2 else 0
            except Exception:
                payload["apoc_or_gds"] = 0
        if "version" not in payload:
            payload["version"] = "unknown"
        driver.close()
        return payload
    except Exception as e:
        return {"ok": False, "error": str(e)}


def build_graph_from_csv(path: Path, src: str = "src", dst: str = "dst", weight: Optional[str] = None,
                         directed: bool = False) -> nx.Graph:
    df = pd.read_csv(path)
    G = nx.DiGraph() if directed else nx.Graph()
    for _, row in df.iterrows():
        w = float(row[weight]) if weight and pd.notnull(row[weight]) else 1.0
        G.add_edge(str(row[src]), str(row[dst]), weight=w)
    logger.info("Built graph from CSV: nodes={}, edges={}", G.number_of_nodes(), G.number_of_edges())
    return G


def build_graph_from_jsonl(path: Path, src: str = "src", dst: str = "dst", weight: Optional[str] = None,
                           directed: bool = False) -> nx.Graph:
    import json

    G = nx.DiGraph() if directed else nx.Graph()
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            w = float(obj.get(weight, 1.0)) if weight else 1.0
            G.add_edge(str(obj[src]), str(obj[dst]), weight=w)
    logger.info("Built graph from JSONL: nodes={}, edges={}", G.number_of_nodes(), G.number_of_edges())
    return G


def save_graph(G: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(G, path)
    logger.info("Saved graph to {}", path)


def load_graph(path: Path) -> nx.Graph:
    G = joblib.load(path)
    logger.info("Loaded graph from {} (nodes={}, edges={})", path, G.number_of_nodes(), G.number_of_edges())
    return G


def push_to_neo4j(G: nx.Graph) -> None:
    settings = get_settings()
    if not (settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password):
        logger.warning("Neo4j not configured; skipping push")
        return
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        logger.error("Neo4j driver not available: {}", e)
        return

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    db = settings.neo4j_database or "neo4j"
    logger.info("Pushing graph to Neo4j database={} ...", db)
    with driver.session(database=db) as session:
        session.run("MATCH (n) DETACH DELETE n")
        # Create nodes
        for n in G.nodes:
            session.run("CREATE (n:Entity {id: $id})", id=str(n))
        # Create edges
        for u, v, data in G.edges(data=True):
            w = float(data.get("weight", 1.0))
            session.run(
                "MATCH (a:Entity {id: $u}), (b:Entity {id: $v}) CREATE (a)-[:TX {weight: $w}]->(b)",
                u=str(u), v=str(v), w=w,
            )
    driver.close()
    logger.info("Pushed graph to Neo4j")
