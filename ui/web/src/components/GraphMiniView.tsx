import React, { useEffect, useState } from "react";
import { api } from "../lib/api";

interface Props {
  nodeId: string | null;
}

interface EgoNode {
  id: string;
  x: number;
  y: number;
  isFocus: boolean;
}

function seededRandom(seed: number): () => number {
  let x = seed;
  return () => {
    x = (x * 9301 + 49297) % 233280;
    return x / 233280;
  };
}

export default function GraphMiniView({ nodeId }: Props) {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!nodeId) {
      setData(null);
      setErr(null);
      return;
    }
    setLoading(true);
    setErr(null);
    api
      .caseData(nodeId)
      .then((j: any) => setData(j))
      .catch((e: any) => setErr(e?.body || e?.message || "Case lookup failed"))
      .finally(() => setLoading(false));
  }, [nodeId]);

  const ego = data?.ego || {};
  const neighbors: string[] = data?.neighbors?.nodes || [];
  const neighborCount: number = data?.neighbors?.count ?? neighbors.length;

  // Build layout for ego graph: focus at center, neighbors in a ring
  const renderGraph = () => {
    if (!neighbors.length || !nodeId) return null;

    const cx = 300;
    const cy = 175;
    const radius = 115;
    const shown = neighbors.slice(0, 16);
    const angleStep = (Math.PI * 2) / Math.max(shown.length, 1);

    // Seeded jitter for organic feel
    const rand = seededRandom(
      nodeId.split("").reduce((a: number, c: string) => a + c.charCodeAt(0), 0)
    );

    const nodes: EgoNode[] = [
      { id: nodeId, x: cx, y: cy, isFocus: true },
      ...shown.map((n, i) => {
        const angle = i * angleStep - Math.PI / 2;
        const r = radius + (rand() - 0.5) * 24;
        return {
          id: n,
          x: cx + Math.cos(angle) * r,
          y: cy + Math.sin(angle) * r,
          isFocus: false,
        };
      }),
    ];

    // A couple of cross-edges between neighbors for visual interest
    const crossEdges: [number, number][] = [];
    if (shown.length >= 4) {
      crossEdges.push([1, Math.min(3, shown.length)]);
      crossEdges.push([2, Math.min(5, shown.length)]);
    }

    return (
      <svg viewBox="0 0 600 350" preserveAspectRatio="xMidYMid meet">
        <defs>
          <radialGradient id="focusGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
          </radialGradient>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto"
          >
            <path d="M0 0L10 5L0 10z" fill="rgba(34, 211, 238, 0.55)" />
          </marker>
        </defs>

        {/* Focus-to-neighbor edges */}
        <g stroke="#0d9488" strokeOpacity="0.55" strokeWidth="1.2" fill="none">
          {nodes.slice(1).map((n, i) => (
            <line
              key={`e-${i}`}
              x1={cx}
              y1={cy}
              x2={n.x}
              y2={n.y}
              markerEnd="url(#arrow)"
            />
          ))}
        </g>

        {/* Cross edges for visual interest */}
        <g stroke="#7a7468" strokeOpacity="0.35" strokeWidth="0.8" fill="none" strokeDasharray="3 3">
          {crossEdges.map(([a, b], i) => (
            <line
              key={`c-${i}`}
              x1={nodes[a]?.x || cx}
              y1={nodes[a]?.y || cy}
              x2={nodes[b]?.x || cx}
              y2={nodes[b]?.y || cy}
            />
          ))}
        </g>

        {/* Focus glow */}
        <circle cx={cx} cy={cy} r="30" fill="url(#focusGlow)" />

        {/* Animated ring around focus */}
        <circle
          cx={cx}
          cy={cy}
          r="22"
          fill="none"
          stroke="#f59e0b"
          strokeOpacity="0.35"
          strokeWidth="1"
          strokeDasharray="3 4"
        >
          <animateTransform
            attributeName="transform"
            type="rotate"
            from={`0 ${cx} ${cy}`}
            to={`360 ${cx} ${cy}`}
            dur="18s"
            repeatCount="indefinite"
          />
        </circle>

        {/* Nodes */}
        {nodes.map((n, i) => {
          if (n.isFocus) {
            return (
              <g key={n.id}>
                <circle cx={n.x} cy={n.y} r="14" fill="#0a0f14" stroke="#f59e0b" strokeWidth="1.5" />
                <text
                  x={n.x}
                  y={n.y + 1}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="#f59e0b"
                  fontFamily="Fira Code, monospace"
                  fontSize="8"
                  fontWeight="700"
                  letterSpacing="0.5"
                >
                  TARGET
                </text>
                <text
                  x={n.x}
                  y={n.y + 32}
                  textAnchor="middle"
                  fill="#e4e1d7"
                  fontFamily="DM Serif Display, serif"
                  fontSize="13"
                  fontStyle="italic"
                >
                  {n.id}
                </text>
              </g>
            );
          }
          const isMerchant = n.id.startsWith("M");
          const color = isMerchant ? "#84cc16" : "#22d3ee";
          return (
            <g key={`${n.id}-${i}`}>
              <circle cx={n.x} cy={n.y} r="6" fill={color} stroke="#fff" strokeOpacity="0.35" strokeWidth="1" />
              <text
                x={n.x}
                y={n.y - 12}
                textAnchor="middle"
                fill="#c4bfb3"
                fontFamily="Fira Code, monospace"
                fontSize="8"
              >
                {n.id}
              </text>
            </g>
          );
        })}

        {neighbors.length > shown.length && (
          <text
            x="300"
            y="335"
            textAnchor="middle"
            fill="#7a7468"
            fontFamily="Fira Code, monospace"
            fontSize="9"
            letterSpacing="1"
          >
            + {neighbors.length - shown.length} more neighbors
          </text>
        )}
      </svg>
    );
  };

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">
          <span className="n">03</span>
          Ego Network
        </h3>
        {nodeId && (
          <span
            style={{
              fontFamily: "var(--mono)",
              fontSize: "0.62rem",
              color: "var(--ash)",
              letterSpacing: "0.1em",
              textTransform: "uppercase",
            }}
          >
            {nodeId} · 1-hop
          </span>
        )}
      </div>
      <div className="card-body">
        <div className="ego-wrap">
          {renderGraph()}
          {!nodeId && (
            <div className="ego-empty">
              Select a suspect to render their<br />1-hop ego subgraph
            </div>
          )}
          {nodeId && loading && <div className="ego-empty">Loading ego network…</div>}
          {nodeId && err && <div className="ego-empty">{err}</div>}
          {nodeId && !loading && !err && !neighbors.length && data && (
            <div className="ego-empty">No neighbors found</div>
          )}
        </div>

        {data && (
          <div className="ego-stats">
            <div>
              <div className="k">Neighbors</div>
              <div className="v">{neighborCount}</div>
            </div>
            <div>
              <div className="k">In-Deg</div>
              <div className="v">{Math.round(Number(ego.in_degree || 0))}</div>
            </div>
            <div>
              <div className="k">Out-Deg</div>
              <div className="v">{Math.round(Number(ego.out_degree || 0))}</div>
            </div>
            <div>
              <div className="k">PageRank</div>
              <div className="v">
                {Number(ego.pagerank || 0).toFixed(4)}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
