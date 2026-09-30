"""
Entity resolution: merging references to the same real-world entity that
appear with different surface forms across documents (e.g. "OpenAI" vs
"Open AI", "Sam Altman" vs "Altman", "Dr. Demis Hassabis").

This is called out explicitly in the product's scaling design (see
README) as the step that gets harder - not the query path - as the
corpus grows from 100 to 1,000+ to lakhs of documents. At 100 documents,
simple normalization is enough to demo the pipeline end-to-end. At real
scale, this step is where the engineering investment goes: embedding-
similarity blocking followed by an LLM adjudicating likely-duplicate
pairs, rather than the exact-match/alias approach used here.

Two entities are merged if:
  1. their normalized (lowercased, punctuation-stripped) names match, or
  2. one name is a known alias/substring of the other (e.g. last-name-only
     mentions of a person already seen with a full name).
"""
from __future__ import annotations

import re


def normalize(name: str) -> str:
    n = name.strip().lower()
    n = re.sub(r"[^\w\s]", "", n)
    n = re.sub(r"\s+", " ", n)
    return n


class EntityResolver:
    """Tracks canonical entity names and resolves surface-form variants
    to a single canonical id as new mentions are seen."""

    def __init__(self):
        self._canonical_by_norm: dict[str, str] = {}  # normalized name -> canonical name
        self._canonical_names: list[str] = []  # canonical names, insertion order

    def resolve(self, name: str) -> str:
        norm = normalize(name)
        if norm in self._canonical_by_norm:
            return self._canonical_by_norm[norm]

        # Check if this is a substring/alias of an existing canonical name
        # (e.g. "Altman" seen after "Sam Altman" already registered), or
        # vice versa (a short mention seen first, fuller name seen later).
        for canon in self._canonical_names:
            canon_norm = normalize(canon)
            tokens = norm.split()
            canon_tokens = canon_norm.split()
            if len(tokens) == 1 and tokens[0] in canon_tokens and len(canon_tokens) > 1:
                self._canonical_by_norm[norm] = canon
                return canon
            if len(canon_tokens) == 1 and canon_tokens[0] in tokens and len(tokens) > 1:
                # New mention is fuller than the existing canonical form;
                # promote it and re-point the old alias too.
                self._canonical_by_norm[canon_norm] = name
                self._canonical_by_norm[norm] = name
                self._canonical_names.append(name)
                return name

        # New canonical entity
        self._canonical_by_norm[norm] = name
        self._canonical_names.append(name)
        return name
