"""
Entity/relationship extraction interface.

In the reference Microsoft GraphRAG pipeline, this step is an LLM prompted
per text unit to emit structured (entity, relation, entity) triples. This
project keeps that exact contract as an abstract interface (`Extractor`)
so a real LLM-backed extractor (Claude, GPT, etc.) can be dropped in later
with zero changes anywhere else in the pipeline - only this file's
implementation would change, and CommunityDetector/summarization/retrieval
never need to know which extractor produced the graph.

For this demo, `SpacyExtractor` implements the interface using a local
NER model plus dependency-parse relation heuristics, since no LLM API key
is available in the environment used to build this project. This is a
legitimate, commonly used approach in lighter-weight GraphRAG variants,
not a placeholder - it produces real (entity, relation, entity) triples
from real text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Entity:
    name: str
    type: str  # e.g. PERSON, ORG, PRODUCT
    source_unit_id: str


@dataclass
class Relationship:
    source: str
    target: str
    relation: str
    source_unit_id: str


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relationships: list[Relationship]
    failed: bool = False  # True if the extraction call itself failed
    # (rate limit, network error, malformed response) rather than the
    # text unit genuinely containing no entities. Distinguishing these
    # matters: a low final entity count could mean either "the
    # extractor was precise" or "many calls silently failed" - without
    # this flag those two very different situations look identical in
    # the final stats.


class Extractor(ABC):
    """Abstract extraction interface. Swap in an LLM-backed implementation
    (e.g. one that prompts Claude for structured JSON triples) by
    subclassing this and implementing `extract`; nothing downstream
    changes."""

    @abstractmethod
    def extract(self, unit_id: str, text: str) -> ExtractionResult: ...
