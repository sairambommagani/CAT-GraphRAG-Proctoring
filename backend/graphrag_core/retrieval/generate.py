"""
Answer generation.

The previous version of global search answered by literally concatenating
the top-ranked community summaries - not real synthesis, just
string-joining. This module replaces that with genuine (if still
extractive, not LLM-generated) synthesis:

  1. Split every candidate community's summary into individual sentences.
  2. Score each sentence by query-term overlap (using the same IDF
     weights as reranking, so the scoring logic is consistent end to
     end).
  3. Deduplicate near-identical sentences (two communities sometimes
     restate the same relationship, e.g. both mentioning "Henry Wotton
     answer Basil Hallward").
  4. Select and order the highest-scoring, non-duplicate sentences into
     one coherent short answer, capped at a target length.

This is still not an LLM - it can't paraphrase or infer beyond what's
literally stated in the summaries - but it directly fixes the previous
version's actual complaint (irrelevant communities' full text getting
dumped in regardless of relevance) by making generation query-aware
instead of query-blind.
"""
from __future__ import annotations

import re

from graphrag_core.graph.entity_resolution import normalize
from graphrag_core.retrieval.rerank import RankedCommunity, _idf_weights


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _looks_like_fragment(sentence: str) -> bool:
    """Reject sentences that are almost certainly dialogue-extraction
    noise rather than genuine relationship statements: containing
    em-dashes, or not starting with an uppercase letter/opening quote.
    This mirrors the entity-level filtering in spacy_extractor.py, since
    both are catching the same underlying issue (dialogue punctuation
    confusing extraction) at two different pipeline stages."""
    if "--" in sentence:
        return True
    first_alpha = next((c for c in sentence if c.isalpha()), None)
    if first_alpha is None or not first_alpha.isupper():
        return True
    return False


def _is_near_duplicate(sentence: str, seen_token_sets: list[set[str]], threshold: float = 0.7) -> bool:
    tokens = set(normalize(sentence).split())
    if not tokens:
        return True
    for seen in seen_token_sets:
        if not seen:
            continue
        overlap = len(tokens & seen) / min(len(tokens), len(seen))
        if overlap >= threshold:
            return True
    return False


def generate_global_answer(
    ranked_communities: list[RankedCommunity],
    query: str,
    all_community_summaries: dict[int, str],
    max_sentences: int = 5,
) -> str:
    """Select and order the most query-relevant, non-duplicate sentences
    across the reranked candidate communities into one answer."""
    if not ranked_communities:
        return "No relevant communities found for this query."

    query_tokens = set(normalize(query).split())

    # Reuse IDF weights computed over every community's summary, so a
    # sentence containing a distinctive term scores higher than one
    # padded with words common across the whole corpus.
    all_token_sets = [set(normalize(s).split()) for s in all_community_summaries.values()]
    idf = _idf_weights(all_token_sets) if query_tokens else {}

    candidate_sentences: list[tuple[float, str]] = []
    for rc in ranked_communities:
        for sentence in _split_sentences(rc.summary):
            if _looks_like_fragment(sentence):
                continue
            sent_tokens = set(normalize(sentence).split())
            overlap_score = sum(idf.get(t, 1.0) for t in (query_tokens & sent_tokens))
            # Weight by the community's own rerank score too, so a
            # highly relevant community's sentences are preferred over
            # an equally-matching sentence from a weaker community.
            combined_score = overlap_score + 0.1 * rc.score
            candidate_sentences.append((combined_score, sentence))

    candidate_sentences.sort(key=lambda t: t[0], reverse=True)

    selected: list[str] = []
    seen_token_sets: list[set[str]] = []
    for score, sentence in candidate_sentences:
        if score <= 0 and selected:
            break  # stop once we run out of query-relevant sentences
        if _is_near_duplicate(sentence, seen_token_sets):
            continue
        selected.append(sentence)
        seen_token_sets.append(set(normalize(sentence).split()))
        if len(selected) >= max_sentences:
            break

    if not selected:
        # Fall back to the single top-ranked community's summary if
        # nothing scored above zero (e.g. a query with no term overlap
        # at all, which shouldn't reach here given global_search's own
        # filtering, but keeps this function safe standalone).
        return ranked_communities[0].summary

    return " ".join(selected)
