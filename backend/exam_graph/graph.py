"""Exam knowledge graph: GraphRAG over the assessment syllabus and the question bank.

    section: ML Fundamentals --contains--> topic: Overfitting and Regularization
    topic: Overfitting and Regularization --covers--> concept: l1 regularization
    topic: Deep Network Training --requires--> topic: Optimization
    concept: l1 regularization --contrasts_with--> concept: l2 regularization   (NIM relation extraction)
    question q039 --tests--> concept: l1 regularization

Built with the GraphRAG pipeline (graphrag_core/: Extractor interface -> build_graph with
entity resolution -> Louvain communities -> summaries -> local/global search). Used for:

* blueprint()           exam sections and weights for content-balanced CAT (cat_blueprint.py)
* topic_context()       grounding passages + prerequisite concepts for question generation
* tag_question()        section / topic / concepts of any question (seed or generated)
* coverage()            concepts and topics without enough questions (what to generate next)
* ask()                 examiner questions over syllabus + bank + candidate performance
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from typing import Optional

import networkx as nx

from graphrag_core.community.detect import build_hierarchy, detect_communities
from graphrag_core.community.summarize import ExtractiveSummarizer, summarize_all_communities
from graphrag_core.extraction.base import Entity, ExtractionResult, Extractor, Relationship
from graphrag_core.graph.build_graph import build_graph, graph_stats
from graphrag_core.graph.entity_resolution import normalize
from graphrag_core.retrieval.rerank import rerank_local_neighbors
from graphrag_core.retrieval.search import global_search, local_search, synthesize_global_answer

from .syllabus import Syllabus, Topic

RELATIONS = ("prerequisite_of", "part_of", "contrasts_with", "related_to", "example_of")
STOP = set("a an the of on in at to for and or is are was be what which who how why when where do does it its "
           "this that with by as from not no into than then there their them these those can will would "
           "following true false best describes used use using python create creates value values return "
           "returns print output result results evaluate main mean means called call typically primarily key "
           "difference between whether makes make one two several each many".split())


def section_node(name: str) -> str:
    return f"section: {name}"


def topic_node(name: str) -> str:
    return f"topic: {name}"


def concept_node(name: str) -> str:
    return f"concept: {name.lower()}"


def question_node(qid: str) -> str:
    return f"question {qid}"


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9_@]+", (text or "").lower()) if w not in STOP and len(w) > 1]


class SyllabusExtractor(Extractor):
    """GraphRAG Extractor for syllabus topics: the structure is authored (sections, topics,
    **bold** concepts, requires:), so extraction is exact and auditable."""

    def __init__(self, syllabus: Syllabus):
        self.s = syllabus

    def extract(self, unit_id: str, text: str) -> ExtractionResult:
        topic = next(t for t in self.s.topics if t.id == unit_id)
        ents, rels = [], []
        sec, top = section_node(topic.section), topic_node(topic.name)
        ents += [Entity(sec, "SECTION", unit_id), Entity(top, "TOPIC", unit_id)]
        rels.append(Relationship(sec, top, "contains", unit_id))
        for c in topic.concepts:
            ents.append(Entity(concept_node(c), "CONCEPT", unit_id))
            rels.append(Relationship(top, concept_node(c), "covers", unit_id))
        for r in topic.requires:
            ents.append(Entity(topic_node(r), "TOPIC", unit_id))
            rels.append(Relationship(top, topic_node(r), "requires", unit_id))
        return ExtractionResult(ents, rels)


class ExamGraph:
    def __init__(self, syllabus: Syllabus, llm=None):
        self.syllabus = syllabus
        self.llm = llm                      # nim.NIMText or None
        self.lock = threading.RLock()
        self.concept_relations: list[tuple[str, str, str]] = []     # (a, relation, b) from NIM
        self.questions: dict[str, dict] = {}                        # id -> tagged question
        self.performance: dict[str, dict] = {}                      # question id -> stats (analytics)
        self._comm = None
        self._rebuild()

    # ---- indexing ---------------------------------------------------------------------------
    def _rebuild(self) -> None:
        ex = SyllabusExtractor(self.syllabus)
        results = [ex.extract(t.id, t.text) for t in self.syllabus.topics]
        extra = ExtractionResult([], [])
        for a, rel, b in self.concept_relations:
            extra.entities += [Entity(concept_node(a), "CONCEPT", "llm"), Entity(concept_node(b), "CONCEPT", "llm")]
            extra.relationships.append(Relationship(concept_node(a), concept_node(b), rel, "llm"))
        for q in self.questions.values():
            qn = question_node(q["id"])
            extra.entities.append(Entity(qn, "QUESTION", q["id"]))
            for c in q.get("concepts", []):
                extra.relationships.append(Relationship(qn, concept_node(c), "tests", q["id"]))
            if q.get("topic_name"):
                extra.relationships.append(Relationship(qn, topic_node(q["topic_name"]), "in_topic", q["id"]))
        self.graph = build_graph(results + [extra])
        self._comm = None

    def set_syllabus(self, syllabus: Syllabus) -> None:
        with self.lock:
            self.syllabus = syllabus
            self.concept_relations = []
            self._rebuild()

    def extract_concept_relations(self) -> int:
        """NIM: typed relations between the syllabus' own concepts (constrained vocabulary,
        so the model can't invent entities). Adds edges like 'l1 regularization
        contrasts_with l2 regularization' that make multi-hop questions possible."""
        if self.llm is None:
            return 0
        found = []
        for t in self.syllabus.topics:
            if len(t.concepts) < 2:
                continue
            try:
                out = self.llm.chat_json(
                    "You extract relations between exam syllabus concepts. Use ONLY the given concept "
                    f"names and ONLY these relation types: {', '.join(RELATIONS)}. Reply with a JSON array of "
                    '{"source": "...", "relation": "...", "target": "..."}. No other text.',
                    f"Topic: {t.name}\nText: {t.text}\nConcepts: {t.concepts}", max_tokens=800)
            except Exception:
                continue
            names = {c.lower() for c in t.concepts}
            for r in out if isinstance(out, list) else []:
                a, rel, b = str(r.get("source", "")).lower(), str(r.get("relation", "")), str(r.get("target", "")).lower()
                if a in names and b in names and a != b and rel in RELATIONS:
                    found.append((a, rel, b))
        with self.lock:
            self.concept_relations = sorted(set(found))
            self._rebuild()
        return len(self.concept_relations)

    def communities(self) -> dict:
        with self.lock:
            if self._comm is None:
                l0 = detect_communities(self.graph)
                l1 = build_hierarchy(self.graph, l0)
                s = ExtractiveSummarizer()
                self._comm = {"level0": l0, "level1": l1,
                              "level0_summaries": summarize_all_communities(self.graph, l0, 0, s),
                              "level1_summaries": summarize_all_communities(self.graph, l1, 1, s)}
            return self._comm

    def stats(self) -> dict:
        c = self.communities()
        with self.lock:
            types = Counter(d.get("type") for _, d in self.graph.nodes(data=True))
            return {**graph_stats(self.graph), "title": self.syllabus.title, "source": self.syllabus.source,
                    "sections": len(self.syllabus.sections), "topics": len(self.syllabus.topics),
                    "concepts": types.get("CONCEPT", 0), "questions_linked": types.get("QUESTION", 0),
                    "concept_relations": len(self.concept_relations), "entity_types": dict(types),
                    "level0_communities": len(c["level0_summaries"]), "level1_communities": len(c["level1_summaries"])}

    # ---- blueprint ------------------------------------------------------------------------------
    def blueprint(self) -> dict:
        s = self.syllabus
        return {"title": s.title, "exam_length": s.exam_length,
                "sections": [{"name": sec.name, "weight": round(w, 3),
                              "topics": [t.name for t in sec.topics]}
                             for sec in s.sections for w in [s.weights()[sec.name]]]}

    # ---- retrieval for question generation --------------------------------------------------------
    def topic_context(self, topic: Topic) -> dict:
        """The topic's passage, its concepts, prerequisite topics (1 hop) and their concepts,
        and concept relations: everything a generator needs for grounded, multi-hop items."""
        with self.lock:
            g = self.graph
            prereq = [v.split(": ", 1)[1] for _, v, d in g.out_edges(topic_node(topic.name), data=True)
                      if d.get("relation") == "requires"]
            pre_topics = [t for t in (self.syllabus.topic(p) for p in prereq) if t]
            rels = [(a, r, b) for a, r, b in self.concept_relations
                    if a in {c.lower() for c in topic.concepts} or b in {c.lower() for c in topic.concepts}]
            return {"section": topic.section, "topic": topic.name, "passage": topic.text,
                    "concepts": topic.concepts,
                    "prerequisites": [{"topic": t.name, "concepts": t.concepts[:6]} for t in pre_topics],
                    "relations": rels[:12]}

    # ---- tagging --------------------------------------------------------------------------------
    def tag_question(self, q: dict) -> dict:
        """Section, topic and concepts of a question, from the graph (IDF-weighted overlap
        between the question and each topic's passage + concepts)."""
        text = " ".join([q.get("text", "")] + list(q.get("options", [])))
        qt = set(tokens(text))
        topics = self.syllabus.topics
        section = self.syllabus.section(q.get("section") or q.get("topic") or "")
        cands = [t for t in topics if section is None or t.section == section.name]
        df = Counter(tok for t in topics for tok in set(tokens(t.text + " " + " ".join(t.concepts))))
        n = len(topics)
        idf = lambda w: math.log((n + 1) / (df.get(w, 0) + 1)) + 1

        def score(t: Topic) -> float:
            tt = set(tokens(t.text + " " + " ".join(t.concepts)))
            ct = set(tokens(" ".join(t.concepts)))
            return sum(idf(w) for w in qt & tt) + sum(idf(w) for w in qt & ct)
        best = max(cands, key=score) if cands else None
        concepts = []
        if best is not None:
            low = text.lower()
            for c in best.concepts:
                ctoks = tokens(c)
                if c.lower() in low or (ctoks and all(w in qt for w in ctoks)):
                    concepts.append(c)
            if not concepts:
                concepts = sorted(best.concepts, key=lambda c: -len(qt & set(tokens(c))))[:1]
        return {"section": best.section if best else (section.name if section else None),
                "topic_name": best.name if best else None, "concepts": concepts}

    def link_questions(self, questions: list[dict]) -> None:
        with self.lock:
            self.questions = {q["id"]: q for q in questions}
            self._rebuild()

    def coverage(self, min_per_topic: int = 3) -> list[dict]:
        """Topics ordered by how badly they need questions (approved questions per topic,
        weighted by the section's blueprint share)."""
        weights = self.syllabus.weights()
        per_topic = Counter(q.get("topic_name") for q in self.questions.values() if q.get("status") == "approved")
        tested = {c.lower() for q in self.questions.values() if q.get("status") == "approved" for c in q.get("concepts", [])}
        out = []
        for t in self.syllabus.topics:
            have = per_topic.get(t.name, 0)
            untested = [c for c in t.concepts if c.lower() not in tested]
            need = max(0, min_per_topic - have)
            out.append({"section": t.section, "topic": t.name, "approved_questions": have, "need": need,
                        "untested_concepts": untested,
                        "priority": round(need * weights.get(t.section, 0) + 0.1 * len(untested), 3)})
        return sorted(out, key=lambda r: -r["priority"])

    # ---- examiner Q&A ------------------------------------------------------------------------------
    KEYWORDS = {"fail": "performance", "difficult": "performance", "hard": "performance", "weak": "performance",
                "coverage": "coverage", "untested": "coverage", "no question": "coverage", "missing": "coverage"}

    def ask(self, question: str) -> dict:
        c = self.communities()
        q = question.lower()
        with self.lock:
            content = " ".join(w for w in tokens(question))
            local = local_search(self.graph, content or question)
            anchor = self._anchor(q)
            if anchor:
                nbrs = [{"direction": "out", "relation": d.get("relation"), "entity": v}
                        for _, v, d in self.graph.out_edges(anchor, data=True)]
                nbrs += [{"direction": "in", "relation": d.get("relation"), "entity": u}
                         for u, _, d in self.graph.in_edges(anchor, data=True)]
                local.matched_entity, local.ranked_neighbors = anchor, rerank_local_neighbors(nbrs, question)
            glob = global_search(c["level0_summaries"], question, top_k=3)
            lines, method = [], "extractive (GraphRAG global search)"
            intent = next((v for k, v in self.KEYWORDS.items() if k in q), None)
            if intent == "performance":
                lines = self._performance_lines(anchor)
            elif intent == "coverage":
                lines = [f"{r['section']} / {r['topic']}: {r['approved_questions']} approved question(s)"
                         + (f", untested concepts: {', '.join(r['untested_concepts'][:5])}" if r["untested_concepts"] else "")
                         for r in self.coverage() if r["need"] or r["untested_concepts"]][:8]
            elif anchor:
                lines = self._describe(anchor)
            if lines:
                method = "graph traversal (GraphRAG local search)"
                answer = " ".join(lines)
            else:
                answer = synthesize_global_answer(glob, question, c["level0_summaries"])
        return {"question": question, "answer": answer, "method": method, "matched_entity": local.matched_entity,
                "neighbors": [{"relation": n.relation, "entity": n.entity} for n in local.ranked_neighbors[:12]],
                "communities": [{"id": rc.community_id, "summary": rc.summary[:300]} for rc in glob.ranked_communities]}

    def _anchor(self, q: str) -> Optional[str]:
        g = self.graph
        best, best_len = None, 0
        for node in g.nodes:
            name = str(node).split(": ", 1)[-1].lower()
            if str(node).startswith("question "):
                name = str(node).lower()
            if len(name) > 3 and re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", q) and len(name) > best_len:
                best, best_len = node, len(name)
        return best

    def _describe(self, node: str) -> list[str]:
        g = self.graph
        kind = g.nodes[node].get("type")
        name = str(node).split(": ", 1)[-1]
        if kind == "SECTION":
            topics = [v.split(": ", 1)[1] for _, v, d in g.out_edges(node, data=True) if d.get("relation") == "contains"]
            w = self.syllabus.weights().get(name, 0)
            return [f"Section '{name}' is {w:.0%} of the exam and covers: {', '.join(topics)}."]
        if kind == "TOPIC":
            t = self.syllabus.topic(name)
            req = [v.split(": ", 1)[1] for _, v, d in g.out_edges(node, data=True) if d.get("relation") == "requires"]
            needed_by = [u.split(": ", 1)[1] for u, _, d in g.in_edges(node, data=True) if d.get("relation") == "requires"]
            qs = [u for u, _, d in g.in_edges(node, data=True) if d.get("relation") == "in_topic"]
            out = [f"Topic '{name}' ({t.section if t else '?'}) covers {', '.join(t.concepts if t else [])}."]
            if req:
                out.append(f"It requires: {', '.join(req)}.")
            if needed_by:
                out.append(f"It is a prerequisite of: {', '.join(needed_by)}.")
            out.append(f"{len(qs)} question(s) in the bank test it.")
            return out
        if kind == "CONCEPT":
            topics = [u.split(": ", 1)[1] for u, _, d in g.in_edges(node, data=True) if d.get("relation") == "covers"]
            qs = [u.split(" ", 1)[1] for u, _, d in g.in_edges(node, data=True) if d.get("relation") == "tests"]
            rel = [f"{d.get('relation')} {v.split(': ', 1)[1]}" for _, v, d in g.out_edges(node, data=True)
                   if d.get("relation") in RELATIONS]
            out = [f"Concept '{name}' is taught in: {', '.join(topics) or '-'}; tested by {len(qs)} question(s)"
                   + (f" ({', '.join(qs[:6])})" if qs else "") + "."]
            if rel:
                out.append("Relations: " + "; ".join(rel) + ".")
            return out + self._performance_lines(node)[:2]
        if kind == "QUESTION":
            qd = self.questions.get(name.replace("question ", ""), {})
            return [f"{node}: {qd.get('text', '')[:160]} (topic {qd.get('topic_name')}, concepts {qd.get('concepts')})."]
        return []

    def _performance_lines(self, anchor: Optional[str]) -> list[str]:
        """Concept / topic failure rates from the candidate response log (analytics.py)."""
        if not self.performance:
            return ["No candidate responses recorded yet."]
        by_topic = defaultdict(lambda: [0, 0])
        for qid, st in self.performance.items():
            q = self.questions.get(qid)
            if not q:
                continue
            if anchor and anchor.startswith("section: ") and q.get("section") != anchor[9:]:
                continue
            by_topic[q.get("topic_name")][0] += st["n"] - st["correct"]
            by_topic[q.get("topic_name")][1] += st["n"]
        rows = sorted(((w / n, t, n) for t, (w, n) in by_topic.items() if n), reverse=True)
        return [f"{t}: {rate:.0%} wrong over {n} answer(s)." for rate, t, n in rows[:6]]
