"""
Community summarization.

Microsoft's GraphRAG prompts an LLM to write a natural-language summary
of each community, in advance, at index time. That pre-computation is
what keeps query-time cost roughly flat as the corpus grows - global
search answers by reading cheap summaries, never by re-traversing the
raw graph per query (see README, "Scaling design").

This module follows the same interface (`Summarizer.summarize`) so an
LLM-backed implementation is a drop-in replacement, but the demo ships a
working *extractive* summarizer: it ranks entities in a community by
degree (how connected they are within the community) and renders the
highest-weight relationship triples into a templated summary. This is a
legitimate, well-understood approach - it is what several lightweight
GraphRAG implementations fall back to - though it produces a flatter,
more mechanical summary than an LLM would write.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx


@dataclass
class CommunitySummary:
    community_id: int
    level: int
    entities: list[str]
    summary: str


class Summarizer(ABC):
    @abstractmethod
    def summarize(self, community_id: int, level: int, subgraph: nx.MultiDiGraph) -> CommunitySummary: ...


class ExtractiveSummarizer(Summarizer):
    def summarize(self, community_id: int, level: int, subgraph: nx.MultiDiGraph) -> CommunitySummary:
        # Rank entities by degree within this community's subgraph.
        degrees = dict(subgraph.degree())
        ranked_entities = sorted(degrees, key=lambda n: degrees[n], reverse=True)

        # Collect relationship triples, deduped, favoring edges between
        # the highest-degree (most central) entities first.
        seen_triples = set()
        triples = []
        for u, v, data in subgraph.edges(data=True):
            key = (u, data.get("relation", "associated_with"), v)
            if key in seen_triples:
                continue
            seen_triples.add(key)
            triples.append(key)
        triples.sort(key=lambda t: degrees.get(t[0], 0) + degrees.get(t[2], 0), reverse=True)

        top_entities = ranked_entities[:8]
        top_triples = triples[:6]

        if top_triples:
            sentences = [f"{s} {r.replace('_', ' ')} {o}." for s, r, o in top_triples]
        else:
            sentences = [f"Entities in this community: {', '.join(top_entities)}."]

        summary = " ".join(sentences)
        return CommunitySummary(
            community_id=community_id,
            level=level,
            entities=ranked_entities,
            summary=summary,
        )


def summarize_all_communities(
    graph: nx.MultiDiGraph,
    community_map: dict[str, int],
    level: int,
    summarizer: Summarizer,
    show_progress: bool = False,
) -> dict[int, CommunitySummary]:
    groups: dict[int, list[str]] = {}
    for node, c in community_map.items():
        groups.setdefault(c, []).append(node)

    summaries = {}
    total = len(groups)
    for i, (community_id, nodes) in enumerate(groups.items(), start=1):
        subgraph = graph.subgraph(nodes)
        summaries[community_id] = summarizer.summarize(community_id, level, subgraph)
        if show_progress:
            # Community summarization can also be one API call each on
            # the LLM path - same rationale as the extraction progress
            # printing in pipeline.py: visible progress instead of
            # silence during a long-running real indexing pass.
            print(f"[summarization level {level}] {i}/{total} communities processed", flush=True)
    return summaries
