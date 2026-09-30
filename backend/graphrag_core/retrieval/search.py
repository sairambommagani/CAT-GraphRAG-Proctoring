"""
Query-time retrieval: local search and global search.

Local search: for narrow, entity-focused questions. We extract candidate
entity names from the query (by matching against known graph node names),
find that entity's immediate neighborhood, then RERANK those neighbors
by query relevance (app.retrieval.rerank.rerank_local_neighbors) instead
of returning them in whatever order NetworkX's edge iteration happens to
produce.

Global search: for broad, corpus-wide questions. A cheap first pass
(this file) filters every pre-computed community summary down to
candidates with any query-term overlap at all; a second, more careful
pass (app.retrieval.rerank.rerank_global_communities) then reranks those
candidates using IDF-weighted scoring, which corrects the original raw
word-overlap approach's bias toward large communities. The final answer
is produced by app.retrieval.generate.generate_global_answer, which
selects and orders the most query-relevant sentences from the reranked
communities instead of concatenating full summaries regardless of
relevance.

The key scaling property either mode is designed to demonstrate: local
search cost depends only on one entity's neighborhood (not corpus size),
and global search cost depends only on the number of communities (which
grows far slower than the number of documents, since Louvain/Leiden
compresses many entities into few communities) - never on raw document
count. That is what keeps query latency roughly flat from 100 documents
to lakhs. Reranking adds a second pass over an already-small candidate
set (one entity's neighbors, or a handful of filtered communities), so
it does not change this scaling property.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from graphrag_core.community.summarize import CommunitySummary
from graphrag_core.graph.entity_resolution import normalize
from graphrag_core.retrieval.generate import generate_global_answer
from graphrag_core.retrieval.rerank import RankedCommunity, RankedNeighbor, rerank_global_communities, rerank_local_neighbors


@dataclass
class LocalSearchResult:
    matched_entity: str | None
    neighbors: list[dict]  # kept for backward compatibility / raw access
    ranked_neighbors: list[RankedNeighbor]


@dataclass
class GlobalSearchResult:
    ranked_communities: list[RankedCommunity]


def local_search(graph: nx.MultiDiGraph, query: str) -> LocalSearchResult:
    query_norm = normalize(query)
    query_tokens = set(query_norm.split())

    best_match = None
    best_overlap = 0
    for node in graph.nodes():
        node_norm = normalize(node)
        node_tokens = set(node_norm.split())
        if node_norm in query_norm:
            overlap = len(node_tokens) + 1  # exact substring match wins
        else:
            overlap = len(node_tokens & query_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = node

    if best_match is None or best_overlap == 0:
        return LocalSearchResult(matched_entity=None, neighbors=[], ranked_neighbors=[])

    neighbors = []
    for _, v, data in graph.out_edges(best_match, data=True):
        neighbors.append({"direction": "out", "relation": data.get("relation"), "entity": v})
    for u, _, data in graph.in_edges(best_match, data=True):
        neighbors.append({"direction": "in", "relation": data.get("relation"), "entity": u})

    ranked_neighbors = rerank_local_neighbors(neighbors, query)

    return LocalSearchResult(matched_entity=best_match, neighbors=neighbors, ranked_neighbors=ranked_neighbors)


def global_search(
    community_summaries: dict[int, CommunitySummary], query: str, top_k: int = 3, candidate_pool: int = 10
) -> GlobalSearchResult:
    """Two-pass retrieval: a cheap first pass (raw word overlap) filters
    every community down to a candidate pool, then rerank_global_communities
    applies the more careful IDF-weighted second pass over just that
    smaller pool - the standard "retrieve cheaply, rerank carefully"
    pattern, kept cheap by never IDF-scoring communities with zero
    overlap in the first place."""
    query_tokens = set(normalize(query).split())

    first_pass = []
    for cid, summary in community_summaries.items():
        text_tokens = set(normalize(summary.summary).split())
        entity_tokens = set()
        for e in summary.entities:
            entity_tokens |= set(normalize(e).split())
        overlap = len(query_tokens & (text_tokens | entity_tokens))
        if overlap > 0:
            first_pass.append((cid, overlap))

    first_pass.sort(key=lambda t: t[1], reverse=True)
    candidate_ids = [cid for cid, _ in first_pass[:candidate_pool]]

    reranked = rerank_global_communities(community_summaries, query, candidate_ids)
    return GlobalSearchResult(ranked_communities=reranked[:top_k])


def synthesize_global_answer(
    result: GlobalSearchResult, query: str = "", community_summaries: dict[int, CommunitySummary] | None = None
) -> str:
    """Answer generation. When a query and the full community-summary
    map are provided, uses generate_global_answer to select and order
    the most query-relevant sentences across the reranked communities.
    Falls back to plain concatenation only if called without that
    context (kept for backward compatibility with any caller that
    doesn't have it), which is why passing query/community_summaries is
    strongly preferred."""
    if not result.ranked_communities:
        return "No relevant communities found for this query."

    if query and community_summaries is not None:
        all_summaries = {cid: s.summary for cid, s in community_summaries.items()}
        return generate_global_answer(result.ranked_communities, query, all_summaries)

    return " ".join(rc.summary for rc in result.ranked_communities)
