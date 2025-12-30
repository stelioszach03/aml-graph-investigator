import React, { useEffect, useState } from "react";
import { api } from "../lib/api";

function since(ts: number | null | undefined): string {
  if (!ts) return "—";
  const d = Math.max(0, Date.now() / 1000 - ts);
  if (d < 60) return `${Math.floor(d)}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86_400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86_400)}d ago`;
}

function fmtNum(n: any, digits: number = 4): string {
  if (n == null || typeof n !== "number" || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export default function MetricsCard() {
  const [health, setHealth] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    try {
      const h = await api.health();
      setHealth(h);
    } catch (e: any) {
      setErr(e?.body || e?.message || "Health check failed");
    }
    try {
      const m = await api.getMetricsLast();
      setMetrics(m || null);
    } catch {
      /* no metrics yet */
    }
  };

  useEffect(() => {
    load();
    const interval = setInterval(load, 15_000);
    return () => clearInterval(interval);
  }, []);

  const m = metrics?.metrics || {};

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">
          <span className="n">BACKEND</span>
          Runtime
        </h3>
      </div>
      <div className="card-body">
        {err && <div className="alert error">{err}</div>}
        {!err && !health && <div className="muted">Loading runtime…</div>}

        {health && (
          <dl className="kv">
            <dt>Status</dt>
            <dd className="accent">{health.status === "ok" ? "OPERATIONAL" : health.status}</dd>
            <dt>Version</dt>
            <dd>{health.version || "—"}</dd>
            <dt>Graph Nodes</dt>
            <dd>{health.nodes?.toLocaleString?.() || health.nodes || "—"}</dd>
            <dt>Graph Edges</dt>
            <dd>{health.edges?.toLocaleString?.() || health.edges || "—"}</dd>
          </dl>
        )}

        {metrics && (
          <>
            <div className="section-label" style={{ marginTop: 18 }}>
              Last training
            </div>
            <div className="metric-grid">
              <div className="metric-tile">
                <span className="k">ROC-AUC</span>
                <span className="v">{fmtNum(m.roc_auc, 4)}</span>
              </div>
              <div className="metric-tile">
                <span className="k">PR-AUC</span>
                <span className="v">{fmtNum(m.pr_auc, 4)}</span>
              </div>
              <div className="metric-tile">
                <span className="k">Brier</span>
                <span className="v">{fmtNum(m.brier, 4)}</span>
              </div>
              <div className="metric-tile">
                <span className="k">P@100</span>
                <span className="v">{fmtNum(m.precision_at_100, 3)}</span>
              </div>
            </div>
            <div
              style={{
                marginTop: 12,
                fontFamily: "var(--mono)",
                fontSize: "0.62rem",
                color: "var(--ash)",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
              }}
            >
              Trained · {since(m.trained_at)}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
