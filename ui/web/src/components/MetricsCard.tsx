import React, { useEffect, useState } from "react";
import { api } from "../lib/api";

function since(ts: number | null | undefined) {
  if (!ts) return "–";
  const d = Math.max(0, Date.now()/1000 - ts);
  if (d < 60) return `${Math.floor(d)}s ago`;
  if (d < 3600) return `${Math.floor(d/60)}m ago`;
  return `${Math.floor(d/3600)}h ago`;
}

export default function MetricsCard() {
  const [health, setHealth] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const h = await api.health();
        setHealth(h);
        try {
          const m = await api.getMetricsLast();
          setMetrics(m || null);
        } catch (e:any) { /* metrics may not exist yet */ }
      } catch (e:any) {
        setErr(e?.body || e?.message || "Health check failed");
      }
    })();
  }, []);

  return (
    <div className="card">
      <div className="card-title">Backend</div>
      {err && <div className="error small">{err}</div>}
      {!err && !health && <div className="muted">Loading…</div>}
      {health && (
        <div className="kv">
          <div>status</div><div>{health.status}</div>
          {"version" in health && (<><div>version</div><div>{health.version}</div></>)}
          {"nodes" in health && (<><div>nodes</div><div>{health.nodes}</div></>)}
          {"edges" in health && (<><div>edges</div><div>{health.edges}</div></>)}
        </div>
      )}

      <div className="subtle" style={{marginTop:8}}>Last training metrics</div>
      {!metrics && <div className="muted">No metrics yet.</div>}
      {metrics?.metrics && (
        <div className="kv">
          <div>ROC-AUC</div><div>{metrics.metrics.roc_auc?.toFixed ? metrics.metrics.roc_auc.toFixed(3) : metrics.metrics.roc_auc}</div>
          <div>PR-AUC</div><div>{metrics.metrics.pr_auc?.toFixed ? metrics.metrics.pr_auc.toFixed(3) : metrics.metrics.pr_auc}</div>
          <div>P@100</div><div>{metrics.metrics.precision_at_100?.toFixed ? metrics.metrics.precision_at_100.toFixed(2) : metrics.metrics.precision_at_100}</div>
          <div>P@500</div><div>{metrics.metrics.precision_at_500?.toFixed ? metrics.metrics.precision_at_500.toFixed(2) : metrics.metrics.precision_at_500}</div>
          <div>P@1000</div><div>{metrics.metrics.precision_at_1000?.toFixed ? metrics.metrics.precision_at_1000.toFixed(2) : metrics.metrics.precision_at_1000}</div>
          <div>Trained</div><div>{since(metrics.metrics.trained_at)}</div>
        </div>
      )}
    </div>
  );
}

