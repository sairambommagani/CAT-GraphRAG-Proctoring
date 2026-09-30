"""
Community detection.

Microsoft's reference GraphRAG implementation uses the Leiden algorithm
to cluster the knowledge graph into hierarchical communities. This
project uses the Louvain algorithm instead (via python-louvain) - Louvain
is Leiden's direct predecessor, optimizes the same modularity objective,
and needs no extra native dependencies, which keeps the demo easy to run
anywhere. Leiden improves on Louvain mainly by guaranteeing well-connected
communities (Louvain can occasionally produce disconnected ones); for a
100-document demo graph this distinction rarely matters in practice, but
swapping in `leidenalg` here would be a one-function change if the graph
grows large enough for it to matter.

This module also produces a *simple two-level hierarchy*: level 0 is the
full Louvain partition (fine-grained communities), and level 1 groups
those level-0 communities into larger clusters by re-running community
detection on a graph-of-communities. Microsoft's implementation goes
several levels deeper for very large corpora; two levels is enough to
demonstrate the "query at different granularities" idea at this scale.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx

try:
    import community as community_louvain  # python-louvain package
except ImportError:  # proctoring vendor change: fall back to NetworkX's built-in Louvain (same objective)
    community_louvain = None


def _best_partition(g: nx.Graph) -> dict:
    if community_louvain is not None:
        return community_louvain.best_partition(g, weight="weight", random_state=42)
    parts = nx.community.louvain_communities(g, weight="weight", seed=42)
    return {n: i for i, members in enumerate(parts) for n in members}


def detect_communities(graph: nx.MultiDiGraph) -> dict[str, int]:
    """Return {node: level-0 community id} using Louvain modularity
    optimization. Louvain requires an undirected simple graph."""
    undirected = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += 1
        else:
            undirected.add_edge(u, v, weight=1)
    for n in graph.nodes():
        if n not in undirected:
            undirected.add_node(n)

    if undirected.number_of_edges() == 0:
        # Degenerate case: no edges, every node its own community.
        return {n: i for i, n in enumerate(undirected.nodes())}

    return _best_partition(undirected)


def build_hierarchy(graph: nx.MultiDiGraph, level0: dict[str, int]) -> dict[str, int]:
    """Cluster level-0 communities into a coarser level-1 grouping, by
    building a graph where each level-0 community is a single node
    (weighted by inter-community edge count) and re-running Louvain."""
    meta = nx.Graph()
    for u, v in graph.edges():
        cu, cv = level0.get(u), level0.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        if meta.has_edge(cu, cv):
            meta[cu][cv]["weight"] += 1
        else:
            meta.add_edge(cu, cv, weight=1)
    for c in set(level0.values()):
        if c not in meta:
            meta.add_node(c)

    if meta.number_of_edges() == 0:
        level0_to_level1 = {c: i for i, c in enumerate(meta.nodes())}
    else:
        level0_to_level1 = _best_partition(meta)

    return {node: level0_to_level1[c] for node, c in level0.items()}


def group_nodes_by_community(community_map: dict[str, int]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for node, c in community_map.items():
        groups[c].append(node)
    return dict(groups)
