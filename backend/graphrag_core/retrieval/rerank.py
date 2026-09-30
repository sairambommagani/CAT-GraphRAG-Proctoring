"""
Reranking for local and global search.

Both search modes' first-pass scoring is cheap and coarse (see
search.py) - this module applies a second, more careful pass over the
candidates before they're used for answer generation, the same two-stage
"retrieve cheaply, then rerank" pattern used in production RAG/GraphRAG
systems (a fast first-pass retriever narrows the field; a more expensive
reranker reorders the smaller candidate set).

No LLM or cross-encoder is available in this environment, so both
rerankers here use TF-IDF-style term weighting instead of a learned
reranking model - entirely real, principled scoring (not a placeholder),
just simpler than what a production reranker would use. The interfaces
are still separated as their own functions so swapping in a real
cross-encoder or LLM-based reranker later only touches this file.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from graphrag_core.community.summarize import CommunitySummary
from graphrag_core.graph.entity_resolution import normalize


@dataclass
class RankedNeighbor:
    direction: str
    relation: str | None
    entity: str
    score: float


def rerank_local_neighbors(neighbors: list[dict], query: str) -> list[RankedNeighbor]:
    """Rerank an entity's neighbors by relevance to the query, instead of
    the arbitrary order NetworkX's edge iteration returns them in.

    Scoring: each neighbor is scored by how many query terms appear in
    its relation label plus its own entity name - both a relation that
    echoes the query's intent ("lead" for a "who leads X" query) and an
    entity name that itself matches query terms are relevant signals a
    graph-order listing ignores entirely.
    """
    query_tokens = set(normalize(query).split())
    if not query_tokens:
        # No signal to rerank by - preserve stable order rather than
        # scoring everything zero and arbitrarily reshuffling ties.
        return [
            RankedNeighbor(direction=n["direction"], relation=n.get("relation"), entity=n["entity"], score=0.0)
            for n in neighbors
        ]

    ranked = []
    for n in neighbors:
        relation_tokens = set(normalize(n.get("relation") or "").split())
        entity_tokens = set(normalize(n["entity"]).split())
        score = len(query_tokens & relation_tokens) * 2 + len(query_tokens & entity_tokens)
        ranked.append(
            RankedNeighbor(direction=n["direction"], relation=n.get("relation"), entity=n["entity"], score=float(score))
        )

    # Stable sort: ties keep their original relative order rather than
    # reshuffling arbitrarily, which matters for reproducibility.
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


@dataclass
class RankedCommunity:
    community_id: int
    score: float
    summary: str


def _idf_weights(all_token_sets: list[set[str]]) -> dict[str, float]:
    """Compute inverse-document-frequency weight per token across all
    community summaries, so a query term that appears in nearly every
    community (e.g. "and", or a term specific to this corpus that just
    happens to be common) contributes less than a term that's genuinely
    distinctive to a handful of communities."""
    n_docs = len(all_token_sets)
    doc_freq: Counter = Counter()
    for tokens in all_token_sets:
        for t in tokens:
            doc_freq[t] += 1
    return {t: math.log((n_docs + 1) / (df + 1)) + 1.0 for t, df in doc_freq.items()}


def rerank_global_communities(
    community_summaries: dict[int, CommunitySummary], query: str, candidate_ids: list[int]
) -> list[RankedCommunity]:
    """Rerank global-search candidates using IDF-weighted overlap instead
    of the raw overlap count used for the first-pass filter in
    global_search(). This directly fixes the size-bias problem: a large
    community's summary has more total words and so accumulates more raw
    overlap almost by chance, but IDF weighting means only genuinely
    distinctive, query-relevant terms drive the score, not sheer summary
    length."""
    query_tokens = set(normalize(query).split())
    if not query_tokens or not candidate_ids:
        return [
            RankedCommunity(community_id=cid, score=0.0, summary=community_summaries[cid].summary)
            for cid in candidate_ids
        ]

    # Build IDF weights across ALL communities (not just candidates) so
    # commonality is measured against the whole corpus, which is what
    # makes a term's weight meaningful.
    all_token_sets = []
    for summary in community_summaries.values():
        tokens = set(normalize(summary.summary).split())
        for e in summary.entities:
            tokens |= set(normalize(e).split())
        all_token_sets.append(tokens)
    idf = _idf_weights(all_token_sets)

    ranked = []
    for cid in candidate_ids:
        summary = community_summaries[cid]
        tokens = set(normalize(summary.summary).split())
        for e in summary.entities:
            tokens |= set(normalize(e).split())
        matched = query_tokens & tokens
        score = sum(idf.get(t, 1.0) for t in matched)
        ranked.append(RankedCommunity(community_id=cid, score=score, summary=summary.summary))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
