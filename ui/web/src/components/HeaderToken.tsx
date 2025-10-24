import React, { useEffect, useState } from "react";
import { api } from "../lib/api";

export default function HeaderToken() {
  const [val, setVal] = useState<string>(() => localStorage.getItem("API_TOKEN") || "");
  const [prot, setProt] = useState<boolean | null>(null);

  useEffect(() => {
    api.authStatus().then(j => {
      if (typeof j?.protected === "boolean") setProt(j.protected);
    }).catch(() => setProt(null));
  }, []);

  if (prot === false) return null; // Open API → hide box

  const save = () => { localStorage.setItem("API_TOKEN", val.trim()); window.location.reload(); };
  const clear = () => { localStorage.removeItem("API_TOKEN"); window.location.reload(); };

  return (
    <div className="token-box">
      <span className="muted">Auth: {prot === null ? "…" : prot ? "Protected" : "Open"}</span>
      {localStorage.getItem("API_TOKEN") ? (
        <>
          <span className="badge ok">Token set</span>
          <button className="btn subtle" onClick={clear}>Clear</button>
        </>
      ) : (
        <>
          <input className="input" placeholder="Paste Bearer token" value={val} onChange={(e)=>setVal(e.target.value)} />
          <button className="btn" onClick={save}>Save Token</button>
        </>
      )}
    </div>
  );
}
