import React, { useState } from "react";
import { api } from "../lib/api";

export default function CaseExplorer() {
  const [id, setId] = useState("");
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const explain = async () => {
    if (!id.trim()) return;
    setLoading(true); setErr(null); setData(null);
    try {
      const j = await api.explain(id.trim());
      setData(j);
    } catch (e:any) {
      setErr(e?.body || e?.message || "Explain failed");
    } finally { setLoading(false); }
  };

  return (
    <div className="card">
      <div className="card-title">Case Explorer</div>
      <div className="row">
        <input className="input" placeholder="Enter node id" value={id} onChange={(e)=>setId(e.target.value)} />
        <button className="btn" onClick={explain} disabled={loading || !id.trim()}>{loading ? "…" : "Explain"}</button>
      </div>
      {err && <div className="error">{err}</div>}
      {!err && loading && <div className="muted">Loading explanation…</div>}
      {data && (
        <>
          {"why" in data && <div className="why">{data.why}</div>}
          {Array.isArray(data.paths) && data.paths.length>0 && (
            <div>
              <div className="subtle">Path explanations</div>
              <ul className="chips">
                {data.paths.slice(0,6).map((p:any,i:number)=>(
                  <li key={i} className="chip">{p?.rationale || p?.path_nodes?.join(" → ")}</li>
                ))}
              </ul>
            </div>
          )}
          {Array.isArray(data.top_contributors) && data.top_contributors.length>0 && (
            <div>
              <div className="subtle">Top contributors</div>
              <ul className="list">
                {data.top_contributors.slice(0,8).map((c:any,i:number)=>(
                  <li key={i} className="mono">{c?.feature || "-"}: {Number(c?.contribution||c?.value||0).toFixed(3)}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
}
