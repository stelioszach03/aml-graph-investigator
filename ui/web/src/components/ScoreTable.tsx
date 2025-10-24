import React, { useEffect, useState } from "react";
import { api } from "../lib/api";

type Row = { node_id: string; score: number };

export default function ScoreTable({ onSelect }: { onSelect: (id: string) => void }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [midRows, setMidRows] = useState<Row[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [decimals, setDecimals] = useState<3 | 6>(6);
  const [constant, setConstant] = useState(false);
  const [minMax, setMinMax] = useState<{min:number; max:number} | null>(null);

  const parseArray = (j:any): Row[] => {
    const arr = Array.isArray(j?.topK) ? j.topK : (Array.isArray(j?.results) ? j.results : []);
    return arr as Row[];
  };

  const loadTop = async () => {
    setLoading(true); setErr(null);
    try {
      const j = await api.score(100);         // top 100
      const arr = parseArray(j);
      setRows(arr);
      const scores = arr.map(x => Number(x.score));
      const min = scores.length ? Math.min(...scores) : NaN;
      const max = scores.length ? Math.max(...scores) : NaN;
      setMinMax(isFinite(min) && isFinite(max) ? {min, max} : null);
      const equalAll = Boolean(j?.constant_scores) || (scores.length>1 && (max - min < 1e-9));
      setConstant(equalAll);
    } catch (e:any) {
      setErr(e?.body || e?.message || "Failed to load scores");
    } finally { setLoading(false); }
  };

  const loadMid = async () => {
    setLoading(true); setErr(null); setMidRows(null);
    try {
      const j = await api.score(1000);        // request larger slice
      const arr = parseArray(j);
      if (arr.length >= 100) {
        const start = Math.floor(arr.length * 0.4);
        const end = Math.floor(arr.length * 0.6);
        setMidRows(arr.slice(start, end));
      } else {
        const mid = Math.floor(arr.length/2);
        setMidRows(arr.slice(Math.max(0, mid-10), Math.min(arr.length, mid+10)));
      }
    } catch (e:any) {
      setErr(e?.body || e?.message || "Failed to load mid-range");
    } finally { setLoading(false); }
  };

  useEffect(() => { loadTop(); }, []);

  const fmt = (x:number) => Number(x).toFixed(decimals);

  return (
    <div className="card">
      <div className="card-title" style={{gap:8}}>
        <span>Scores</span>
        <div style={{display:"flex", gap:8, alignItems:"center"}}>
          <button className="btn small" onClick={loadTop} disabled={loading}>{loading?"…":"Refresh"}</button>
          <button className="btn small" onClick={loadMid} disabled={loading}>Mid-range</button>
          <button className="btn small" onClick={()=>setDecimals(decimals===6?3:6)}>
            Decimals: {decimals}
          </button>
        </div>
      </div>

      {err && <div className="error">{err}</div>}

      {minMax && (
        <div className="subtle">
          Range (returned set): {fmt(minMax.min)} – {fmt(minMax.max)}
        </div>
      )}
      {constant && (
        <div className="error small">
          All returned scores are nearly identical. This often happens because you’re viewing only the top-K (very confident) items or due to extreme class imbalance. Try the “Mid-range” button or revisit training params.
        </div>
      )}

      {!err && rows.length === 0 && !loading && <div className="muted">No scores yet — train & score first.</div>}

      <div className="table" style={{marginTop:6}}>
        <div className="thead"><div>Node</div><div>Score</div></div>
        <div className="tbody">
          {rows.map(r => (
            <button key={r.node_id} className="row link" onClick={()=>onSelect(r.node_id)}>
              <div className="mono">{r.node_id}</div>
              <div>{fmt(r.score)}</div>
            </button>
          ))}
        </div>
      </div>

      {Array.isArray(midRows) && (
        <div style={{marginTop:12}}>
          <div className="subtle">Mid-range (center slice of larger set)</div>
          <div className="table">
            <div className="thead"><div>Node</div><div>Score</div></div>
            <div className="tbody">
              {midRows.map(r => (
                <button key={`mid-${r.node_id}`} className="row link" onClick={()=>onSelect(r.node_id)}>
                  <div className="mono">{r.node_id}</div>
                  <div>{fmt(r.score)}</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
