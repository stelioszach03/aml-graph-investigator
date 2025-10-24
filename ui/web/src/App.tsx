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
        <div className="brand">Graph AML Investigator</div>
        <HeaderToken />
      </header>
      <main className="grid">
        <aside className="col">
          <MetricsCard />
          <ScoreTable onSelect={(id)=>setSelected(id)} />
        </aside>
        <section className="col">
          <CaseExplorer />
          <GraphMiniView />
        </section>
      </main>
    </div>
  );
}
