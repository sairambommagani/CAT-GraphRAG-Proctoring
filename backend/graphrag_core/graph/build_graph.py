"""
Knowledge graph construction.

Takes the entities/relationships extracted per text unit and assembles
them into a single NetworkX graph, running each entity name through
entity resolution so the same real-world entity mentioned across
multiple documents collapses into one node. This cross-document merging
is precisely what gives GraphRAG its multi-hop reasoning power over
plain vector RAG: a question that needs to connect facts from two
different source documents can be answered by walking a single node's
edges, rather than requiring both chunks to be independently retrieved
by similarity.
"""
from __future__ import annotations

import networkx as nx

from graphrag_core.extraction.base import ExtractionResult
from graphrag_core.graph.entity_resolution import EntityResolver


def build_graph(extraction_results: list[ExtractionResult]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    resolver = EntityResolver()

    # First pass: register all entities so resolution has full context
    for result in extraction_results:
        for ent in result.entities:
            canon = resolver.resolve(ent.name)
            if graph.has_node(canon):
                graph.nodes[canon]["mentions"] += 1
                graph.nodes[canon]["source_units"].add(ent.source_unit_id)
            else:
                graph.add_node(
                    canon,
                    type=ent.type,
                    mentions=1,
                    source_units={ent.source_unit_id},
                )

    # Second pass: add relationships between resolved entities
    for result in extraction_results:
        for rel in result.relationships:
            src = resolver.resolve(rel.source)
            tgt = resolver.resolve(rel.target)
            if src == tgt or not graph.has_node(src) or not graph.has_node(tgt):
                continue
            graph.add_edge(src, tgt, relation=rel.relation, source_unit_id=rel.source_unit_id)

    return graph


def graph_stats(graph: nx.MultiDiGraph) -> dict:
    return {
        "num_entities": graph.number_of_nodes(),
        "num_relationships": graph.number_of_edges(),
        "avg_degree": (2 * graph.number_of_edges() / graph.number_of_nodes()) if graph.number_of_nodes() else 0,
    }
