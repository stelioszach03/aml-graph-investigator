import networkx as nx

from app.graph.explain import path_explanations, summarize_case


def test_path_explanations_connected():
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", amount=5.0)
    G.add_edge("B", "C", amount=7.0)
    G.nodes["C"]["y"] = 1
    expl = path_explanations(G, "A", k_paths=3, max_len=5)
    assert isinstance(expl, list)
    assert len(expl) >= 1
    assert expl[0]["path_nodes"][0] == "A"


def test_summarize_case_keys():
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", amount=2.0)
    G.add_edge("B", "C", amount=3.0)
    G.nodes["C"]["y"] = 1
    summary = summarize_case(G, "A", score=0.8, features_row={"degree_in": 1.2, "pagerank": -0.3})
    assert set(["why", "top_contributors", "paths"]).issubset(set(summary.keys()))
