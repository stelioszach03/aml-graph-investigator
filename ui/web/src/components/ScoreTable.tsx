import React, { useEffect, useState } from "react";
import { api } from "../lib/api";

type Row = { node_id: string; score: number };

function scoreBand(s: number): "critical" | "high" | "watch" | "low" {
  if (s >= 0.85) return "critical";
  if (s >= 0.6) return "high";
  if (s >= 0.35) return "watch";
  return "low";
}

interface Props {
  onSelect: (id: string) => void;
  selected: string | null;
}

export default function ScoreTable({ onSelect, selected }: Props) {
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [needsBootstrap, setNeedsBootstrap] = useState(false);
  const [count, setCount] = useState<number>(0);
  const [constant, setConstant] = useState(false);
  const [view, setView] = useState<"top" | "mid">("top");

  const parseArray = (j: any): Row[] => {
    const arr = Array.isArray(j?.topK) ? j.topK : Array.isArray(j?.results) ? j.results : [];
    return arr as Row[];
  };

  const parseErr = (e: any): string => {
    if (e?.status === 409) {
      return "Model not initialized yet. Click 'Initialize demo' to run ingest + train.";
    }
    const body = typeof e?.body === "string" ? e.body : "";
    if (body) {
      try {
        const parsed = JSON.parse(body);
        if (parsed?.detail) return String(parsed.detail);
      } catch {}
      return body;
    }
    return e?.message || "Failed to load scores";
  };

  const loadTop = async () => {
    setLoading(true);
    setErr(null);
    setNeedsBootstrap(false);
    setView("top");
    try {
      const j = await api.score(100);
      const arr = parseArray(j);
      setRows(arr);
      setCount(Number(j?.count || arr.length));
      const scores = arr.map((x) => Number(x.score));
      const min = scores.length ? Math.min(...scores) : NaN;
      const max = scores.length ? Math.max(...scores) : NaN;
      const equalAll =
        Boolean(j?.constant_scores) ||
        (scores.length > 1 && isFinite(min) && isFinite(max) && max - min < 1e-9);
      setConstant(equalAll);
    } catch (e: any) {
      setErr(parseErr(e));
      setNeedsBootstrap(e?.status === 409);
    } finally {
      setLoading(false);
    }
  };

  const loadMid = async () => {
    setLoading(true);
    setErr(null);
    setNeedsBootstrap(false);
    setView("mid");
    try {
      const j = await api.score(1000);
      const arr = parseArray(j);
      setCount(Number(j?.count || arr.length));
      if (arr.length >= 100) {
        const start = Math.floor(arr.length * 0.4);
        const end = Math.floor(arr.length * 0.6);
        setRows(arr.slice(start, end));
      } else {
        const mid = Math.floor(arr.length / 2);
        setRows(arr.slice(Math.max(0, mid - 10), Math.min(arr.length, mid + 10)));
      }
      setConstant(false);
    } catch (e: any) {
      setErr(parseErr(e));
      setNeedsBootstrap(e?.status === 409);
    } finally {
      setLoading(false);
    }
  };

  const initializeDemo = async () => {
    setLoading(true);
    setErr(null);
    try {
      await api.ingest("data/raw/synth_edges.csv", false);
      await api.train("data/processed/labels_all.csv");
      await loadTop();
    } catch (e: any) {
      setErr(parseErr(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTop();
  }, []);

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">
          <span className="n">{view === "top" ? "TOP K" : "MID-RANGE"}</span>
          Ranked Suspects
        </h3>
        <div className="btn-row">
          <button className="btn small" onClick={loadTop} disabled={loading}>
            {loading && view === "top" ? "…" : "Top"}
          </button>
          <button className="btn small subtle" onClick={loadMid} disabled={loading}>
            Mid
          </button>
        </div>
      </div>
      <div className="card-body">
        {err && <div className="alert error">{err}</div>}

        {needsBootstrap && (
          <div style={{ marginBottom: 10 }}>
            <button className="btn primary" onClick={initializeDemo} disabled={loading}>
              {loading ? "Initializing…" : "Initialize demo"}
            </button>
          </div>
        )}

        {rows.length > 0 && (
          <div className="ranges">
            {view === "top" ? "Top-100" : "Mid 40-60%"} · <strong>{count.toLocaleString()}</strong>{" "}
            nodes scored · bands by p(fraud)
          </div>
        )}

        {constant && (
          <div className="alert info">
            All returned scores are nearly identical — likely because you're viewing only the
            most confident items. Try the Mid button to sample the decision boundary.
          </div>
        )}

        {!err && rows.length === 0 && !loading && (
          <div className="muted">No scores yet — train first.</div>
        )}

        {rows.length > 0 && (
          <div className="table">
            <div className="thead">
              <div>#</div>
              <div>Node</div>
              <div>Score</div>
              <div>Band</div>
            </div>
            <div className="tbody">
              {rows.map((r, i) => {
                const band = scoreBand(Number(r.score));
                const isActive = selected === r.node_id;
                return (
                  <button
                    key={`${view}-${r.node_id}`}
                    className={"row-item" + (isActive ? " active" : "")}
                    onClick={() => onSelect(r.node_id)}
                  >
                    <div className="row-rank">{String(i + 1).padStart(2, "0")}</div>
                    <div className="row-node">{r.node_id}</div>
                    <div className="row-score">{Number(r.score).toFixed(6)}</div>
                    <div className={"row-band " + band}>{band}</div>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
