import React, { useEffect, useState } from "react";
import { api } from "../lib/api";

interface Props {
  nodeId: string | null;
  onClear: () => void;
}

function scoreBand(s: number): "critical" | "high" | "watch" | "low" {
  if (s >= 0.85) return "critical";
  if (s >= 0.6) return "high";
  if (s >= 0.35) return "watch";
  return "low";
}

export default function CaseExplorer({ nodeId, onClear }: Props) {
  const [data, setData] = useState<any>(null);
  const [caseData, setCaseData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [manualId, setManualId] = useState("");

  const load = async (id: string) => {
    setLoading(true);
    setErr(null);
    setData(null);
    setCaseData(null);
    try {
      const [explainRes, caseRes] = await Promise.all([
        api.explain(id).catch(() => null),
        api.caseData(id).catch(() => null),
      ]);
      if (!explainRes && !caseRes) {
        throw new Error("No data for node");
      }
      setData(explainRes);
      setCaseData(caseRes);
    } catch (e: any) {
      setErr(e?.body || e?.message || "Explain failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (nodeId) {
      setManualId(nodeId);
      load(nodeId);
    } else {
      setData(null);
      setCaseData(null);
    }
  }, [nodeId]);

  const submitManual = (e: React.FormEvent) => {
    e.preventDefault();
    if (manualId.trim()) {
      load(manualId.trim());
    }
  };

  const caseScore =
    caseData?.score != null ? Number(caseData.score) : null;
  const displayScore = caseScore != null && !Number.isNaN(caseScore) ? caseScore : null;
  const band = displayScore != null ? scoreBand(displayScore) : null;

  // Find max absolute contribution for scaling bars
  const contributors: any[] = Array.isArray(data?.top_contributors) ? data.top_contributors : [];
  const maxContrib = contributors.length
    ? Math.max(
        ...contributors.map((c: any) =>
          Math.abs(Number(c?.contribution ?? c?.value ?? 0))
        ),
        1e-9
      )
    : 1;

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">
          <span className="n">02</span>
          Case Explorer
        </h3>
        {nodeId && (
          <button className="btn small subtle" onClick={onClear}>
            Clear
          </button>
        )}
      </div>
      <div className="card-body">
        <form className="row" onSubmit={submitManual} style={{ marginBottom: 14 }}>
          <input
            className="input"
            placeholder="Enter node id (e.g. A1415)"
            value={manualId}
            onChange={(e) => setManualId(e.target.value)}
          />
          <button className="btn primary" type="submit" disabled={loading || !manualId.trim()}>
            {loading ? "…" : "Explain"}
          </button>
        </form>

        {err && <div className="alert error">{err}</div>}
        {!err && loading && <div className="muted">Loading explanation…</div>}

        {!err && !loading && !data && !caseData && !nodeId && (
          <div
            className="muted"
            style={{
              padding: "28px 0",
              textAlign: "center",
              borderTop: "1px dashed var(--edge)",
              marginTop: 6,
            }}
          >
            Select a suspect from the ranked list or enter a node id above.
          </div>
        )}

        {(caseData || data) && (
          <>
            {caseData && (
              <div className="case-target">
                <div className="bullet">⊕</div>
                <div>
                  <div className="id">{caseData.node_id || nodeId}</div>
                  <div
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: "0.58rem",
                      letterSpacing: "0.14em",
                      textTransform: "uppercase",
                      color: "var(--ash)",
                      marginTop: 4,
                    }}
                  >
                    {caseData.neighbors?.count || 0} neighbors ·{" "}
                    {caseData.ego?.ego_nodes || 0} ego nodes ·{" "}
                    {caseData.ego?.ego_edges || 0} ego edges
                  </div>
                </div>
                {displayScore != null && band && (
                  <span className={"row-band " + band} style={{ marginLeft: "auto" }}>
                    {displayScore.toFixed(3)} · {band}
                  </span>
                )}
              </div>
            )}

            {data?.why && <div className="why">{data.why}</div>}

            {contributors.length > 0 && (
              <>
                <div className="section-label">Top Contributing Features</div>
                <ul className="contributors">
                  {contributors.slice(0, 8).map((c: any, i: number) => {
                    const v = Number(c?.contribution ?? c?.value ?? 0);
                    const pct = Math.max(2, (Math.abs(v) / maxContrib) * 100);
                    return (
                      <li key={i} className="contrib-row">
                        <span className="feat">{c?.feature || "—"}</span>
                        <span className="val">{v.toExponential(2)}</span>
                        <div className="contrib-bar">
                          <span className="fill" style={{ width: `${pct.toFixed(1)}%` }} />
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </>
            )}

            {Array.isArray(data?.paths) && data.paths.length > 0 && (
              <>
                <div className="section-label">Path Explanations</div>
                <ul className="chips">
                  {data.paths.slice(0, 5).map((p: any, i: number) => (
                    <li key={i} data-i={`P${String(i + 1).padStart(2, "0")}`}>
                      {p?.rationale ||
                        (Array.isArray(p?.path_nodes) ? p.path_nodes.join(" → ") : "—")}
                      {p?.total_cost != null && (
                        <span
                          style={{
                            marginLeft: 10,
                            color: "var(--cyan)",
                            fontSize: "0.66rem",
                          }}
                        >
                          cost {Number(p.total_cost).toFixed(2)}
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
