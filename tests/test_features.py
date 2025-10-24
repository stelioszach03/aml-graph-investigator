import networkx as nx

from app.graph.features import pagerank_features, motif_counts


def test_pagerank_monotonic_incoming():
    # Directed star: many edges into B -> B should have higher PR than leaves
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("C", "B")
    G.add_edge("D", "B")
    pr = pagerank_features(G)
    assert pr["B"]["pagerank"] > pr["A"]["pagerank"]
    assert pr["B"]["pagerank"] > pr["C"]["pagerank"]
    assert pr["B"]["pagerank"] > pr["D"]["pagerank"]


def test_motif_counts_triangle():
    G = nx.Graph()
    G.add_edge("X", "Y")
    G.add_edge("Y", "Z")
    G.add_edge("Z", "X")
    tri = motif_counts(G)
    assert tri["X"]["triangles"] > 0
    assert tri["Y"]["triangles"] > 0
    assert tri["Z"]["triangles"] > 0
