import React, { useState } from "react";
import MetricsCard from "./components/MetricsCard";
import ScoreTable from "./components/ScoreTable";
import CaseExplorer from "./components/CaseExplorer";
import GraphMiniView from "./components/GraphMiniView";
import HeaderToken from "./components/HeaderToken";

export default function App() {
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}>
              <circle cx="6" cy="6" r="2" />
              <circle cx="18" cy="6" r="2" />
              <circle cx="12" cy="18" r="2" />
              <circle cx="12" cy="12" r="2" />
              <path d="M6 8L12 12M18 8L12 12M12 14L12 16" />
            </svg>
          </div>
          <div className="brand-text">
            <span className="brand-title">
              Aegis <em>/ graph-aml</em>
            </span>
            <span className="brand-sub">Investigator Console</span>
          </div>
        </div>
        <div className="top-right">
          <a className="top-link" href="/aml-graph-investigator/">
            ← Overview
          </a>
          <a
            className="top-link"
            href="/aegis-graph-aml/api/v1/health"
            target="_blank"
            rel="noopener noreferrer"
          >
            Health
          </a>
          <a
            className="top-link"
            href="/aegis-graph-aml/docs"
            target="_blank"
            rel="noopener noreferrer"
          >
            API Docs
          </a>
          <a
            className="top-link"
            href="https://github.com/stelioszach03/aml-graph-investigator"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
          <HeaderToken />
        </div>
      </header>

      <main className="grid">
        <aside className="col">
          <MetricsCard />
          <ScoreTable onSelect={(id) => setSelected(id)} selected={selected} />
        </aside>
        <section className="col">
          <CaseExplorer nodeId={selected} onClear={() => setSelected(null)} />
          <GraphMiniView nodeId={selected} />
        </section>
      </main>
    </div>
  );
}
