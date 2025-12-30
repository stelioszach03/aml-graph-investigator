import React, { useEffect, useState } from "react";
import { api } from "../lib/api";

export default function HeaderToken() {
  const [val, setVal] = useState<string>(() => {
    try {
      return localStorage.getItem("API_TOKEN") || "";
    } catch {
      return "";
    }
  });
  const [prot, setProt] = useState<boolean | null>(null);

  useEffect(() => {
    api
      .authStatus()
      .then((j: any) => {
        if (typeof j?.protected === "boolean") setProt(j.protected);
      })
      .catch(() => setProt(null));
  }, []);

  // Open API → hide entirely
  if (prot === false) return null;

  const save = () => {
    try {
      localStorage.setItem("API_TOKEN", val.trim());
      window.location.reload();
    } catch {}
  };

  const clear = () => {
    try {
      localStorage.removeItem("API_TOKEN");
      window.location.reload();
    } catch {}
  };

  const hasToken = !!(val && val.trim());

  return (
    <div className="token-box">
      {hasToken ? (
        <>
          <span className="badge ok">AUTH OK</span>
          <button className="btn small subtle" onClick={clear}>
            Clear
          </button>
        </>
      ) : (
        <>
          <input
            className="input"
            style={{ width: 180, fontSize: "0.68rem", padding: "7px 10px" }}
            placeholder="Bearer token"
            value={val}
            onChange={(e) => setVal(e.target.value)}
          />
          <button className="btn small" onClick={save}>
            Save
          </button>
        </>
      )}
    </div>
  );
}
