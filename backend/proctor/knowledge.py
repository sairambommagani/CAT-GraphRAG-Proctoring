"""Proctoring knowledge graph: GraphRAG over exam rules and past incidents.

Why a graph
-----------
A verdict should follow the institution's written rules and stay consistent
with how examiners decided similar cases before. Both are relational:

    R-03 No phones --governs--> PROHIBITED_OBJECT --detected_by--> object detector
    R-03 --targets--> phone <--direction-- Incident 7f3a --verdict--> fraud
    Incident 7f3a --reviewed--> examiner confirmed ; --in_session--> Session 1a2b

Walking these edges answers "which rules apply to this flag, what are the
exceptions, and what did examiners decide for similar flags?" - a multi-hop
question that plain keyword or vector lookup over the rule text can't answer.
The same graph answers examiner questions such as "which sessions had
repeated phone use?".

Built on the GraphRAG pipeline from github.com/sairambommagani/GraphRAG
(vendored in graphrag_core/): Extractor interface -> build_graph with entity
resolution -> Louvain communities (2 levels) -> community summaries ->
local search (entity neighbourhood + rerank) and global search (IDF rerank +
answer generation). This module adds:

* ProctorExtractor: a domain extractor for the rule format in
  knowledge/policy/*.md (structured keys + a gazetteer over the rule text)
  and for incident records. Rules are authored text, so a deterministic
  extractor gives an exact, auditable graph; an LLM extractor (the GraphRAG
  repo's LLMExtractor) can be plugged in for free-form policy documents.
* context_for(evidence): the judge's retrieval step (rules + past cases).
* record_incident / record_review: every verdict and examiner decision
  becomes knowledge (the human-in-the-loop feedback loop).
* ask(question): examiner Q&A (local + global search, optional NIM LLM answer).

Retention: incidents hold metadata only (no images, audio or transcript) and
are deleted with the evidence metadata (90 days) or on an erasure request.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

from graphrag_core.community.detect import build_hierarchy, detect_communities
from graphrag_core.community.summarize import ExtractiveSummarizer, summarize_all_communities
from graphrag_core.extraction.base import Entity, ExtractionResult, Extractor, Relationship
from graphrag_core.graph.build_graph import build_graph, graph_stats
from graphrag_core.retrieval.rerank import rerank_local_neighbors
from graphrag_core.retrieval.search import global_search, local_search, synthesize_global_answer

POLICY_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "policy"

EVENT_SIGNALS = {
    "OFFSCREEN_SUSTAINED": ["gaze tracking", "head pose"],
    "REPEATED_GLANCES": ["gaze tracking", "head pose"],
    "NO_FACE": ["face tracking"],
    "MULTIPLE_FACES": ["face tracking"],
    "PROHIBITED_OBJECT": ["object detector"],
    "EXTRA_PERSON": ["object detector"],
    "FOREIGN_HAND": ["hand tracking"],
    "VOICE_DETECTED": ["microphone", "lip sync"],
}

# Gazetteer: surface forms in rule text -> canonical entity (type)
GAZETTEER = {
    "phone": ("phone", "OBJECT"), "mobile phone": ("phone", "OBJECT"), "smartwatch": ("smartwatch", "OBJECT"),
    "earphones": ("earphones", "OBJECT"), "book": ("book", "OBJECT"), "books": ("book", "OBJECT"),
    "notes": ("notes", "OBJECT"), "notebook": ("notes", "OBJECT"), "paper": ("paper", "OBJECT"),
    "laptop": ("laptop", "OBJECT"), "tablet": ("tablet", "OBJECT"), "monitor": ("second screen", "OBJECT"),
    "second screen": ("second screen", "OBJECT"), "television": ("tv", "OBJECT"), "tv": ("tv", "OBJECT"),
    "another person": ("another person", "PERSON"), "second face": ("another person", "PERSON"),
    "helper": ("another person", "PERSON"), "examiner": ("examiner", "PERSON"),
    "looking away": ("looking away", "BEHAVIOUR"), "looking down": ("looking down", "BEHAVIOUR"),
    "talking": ("talking", "BEHAVIOUR"), "whispering": ("whispering", "BEHAVIOUR"),
    "reading aloud": ("reading aloud", "BEHAVIOUR"), "reading questions aloud": ("reading aloud", "BEHAVIOUR"),
    "dictating": ("dictating answers", "BEHAVIOUR"), "signalling": ("signalling", "BEHAVIOUR"),
    "impersonation": ("impersonation", "BEHAVIOUR"), "call": ("phone call", "BEHAVIOUR"),
    "accommodation": ("accommodation", "CONCEPT"), "screen reader": ("screen reader", "CONCEPT"),
    "background": ("background", "CONCEPT"), "pattern": ("pattern", "CONCEPT"),
}

STOPWORDS = set("a an the of on in at to for and or is are was were be been what which who whom whose how "
                "why when where do does did it its this that these those with about from by as any all "
                "had has have there their them they i me my we our you your can could should would will "
                "tell show give list summarise summarize explain say says said".split())

# detector direction -> canonical target entity
DIRECTION_TARGETS = {"phone": "phone", "book": "book", "second device": "laptop", "second screen": "second screen",
                     "down": "down", "left": "left", "right": "right", "up": "up",
                     "candidate": "candidate", "another person": "another person", "unattributed": "unattributed"}


def target_node(t: str) -> str:
    """Targets/objects/people share one namespace, e.g. 'about: phone'. (GraphRAG's entity
    resolver merges a one-word name into any longer name containing it - 'phone' into
    'phone call' - so graph entity names here are always two or more words.)"""
    return f"about: {t.lower()}"


def rule_node(r) -> str:
    return f"{r.id} {r.title}"


def incident_targets(direction: Optional[str]) -> list[str]:
    """Detector direction -> target names ('book, phone' -> ['book', 'phone'])."""
    if not direction:
        return []
    return [DIRECTION_TARGETS.get(p.strip(), p.strip()) for p in direction.split(",") if p.strip()]


@dataclass
class Rule:
    id: str
    title: str
    applies_to: list[str]
    targets: list[str]
    severity: str
    allowed: list[str]
    text: str
    source: str

    def brief(self) -> str:
        allowed = "; ".join(a for a in self.allowed if a.lower() != "none")
        return f"{self.title}: {self.text.split('. ')[0].rstrip('.')}." + (f" Allowed: {allowed}." if allowed else "")

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "severity": self.severity, "applies_to": self.applies_to,
                "targets": self.targets, "allowed": self.allowed, "text": self.text, "source": self.source}


def parse_policy(text: str, source: str = "policy") -> list[Rule]:
    rules = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        lines = block.strip().splitlines()
        m = re.match(r"(R-\d+)\s+(.*)", lines[0].strip())
        if not m:
            continue
        fields, body = {}, []
        for ln in lines[1:]:
            kv = re.match(r"^(applies_to|targets|severity|allowed):\s*(.*)$", ln.strip())
            if kv:
                fields[kv.group(1)] = kv.group(2).strip()
            elif ln.strip():
                body.append(ln.strip())
        split = lambda s, sep=",": [x.strip() for x in (s or "").split(sep) if x.strip()]
        rules.append(Rule(m.group(1), m.group(2).strip(), split(fields.get("applies_to")),
                          [t.lower() for t in split(fields.get("targets"))], fields.get("severity", "major"),
                          split(fields.get("allowed"), ";"), " ".join(body), source))
    return rules


def load_rules(policy_dir: Path = POLICY_DIR) -> list[Rule]:
    rules: list[Rule] = []
    for p in sorted(Path(policy_dir).glob("*.md")):
        rules += parse_policy(p.read_text(encoding="utf-8"), p.name)
    return rules


@dataclass
class Incident:
    evidence_id: str
    session_id: str
    at: float
    type: str
    direction: Optional[str]
    verdict: str
    confidence: float
    rule: str = ""
    reason: str = ""
    model: str = ""
    review: Optional[str] = None       # examiner decision: confirm | dismiss

    @property
    def node(self) -> str:
        return f"Incident {self.evidence_id[:8]}"

    @property
    def outcome(self) -> str:
        """Ground truth when an examiner decided, else the AI verdict."""
        if self.review == "confirm":
            return "confirmed by examiner"
        if self.review == "dismiss":
            return "dismissed by examiner"
        return f"AI {self.verdict}"

    def summary(self) -> str:
        what = f"{self.type}" + (f" ({self.direction})" if self.direction else "")
        rule = f", {self.rule}" if self.rule else ""
        why = f": {self.reason}" if self.reason else ""
        return f"{what} -> {self.outcome}{rule}{why}"[:200]


class ProctorExtractor(Extractor):
    """GraphRAG Extractor for rule sections and incident records.

    Text units are prefixed with their kind ("RULE ..." / "INCIDENT ...") and
    carry structured JSON, so extraction is exact and repeatable."""

    def extract(self, unit_id: str, text: str) -> ExtractionResult:
        kind, _, payload = text.partition(" ")
        data = json.loads(payload)
        ents: list[Entity] = []
        rels: list[Relationship] = []

        def ent(name, typ):
            ents.append(Entity(name, typ, unit_id))
            return name

        def rel(a, r, b):
            rels.append(Relationship(a, b, r, unit_id))

        if kind == "RULE":
            r = Rule(**data)
            node = ent(rule_node(r), "RULE")
            rel(node, "severity", ent(f"severity {r.severity}", "SEVERITY"))
            for et in r.applies_to:
                rel(node, "governs", ent(et, "EVENT_TYPE"))
                for sig in EVENT_SIGNALS.get(et, []):
                    rel(et, "detected_by", ent(f"signal: {sig}", "SIGNAL"))
            for t in r.targets:
                rel(node, "targets", ent(target_node(t), "TARGET"))
            for a in r.allowed:
                if a.lower() != "none":
                    rel(node, "allows", ent(f"exception: {a[:80]}", "EXCEPTION"))
            low = r.text.lower()
            for surface, (canon, typ) in GAZETTEER.items():
                if re.search(rf"\b{re.escape(surface)}\b", low):
                    rel(node, "mentions", ent(target_node(canon), typ))
        elif kind == "INCIDENT":
            inc = Incident(**data)
            node = ent(inc.node, "INCIDENT")
            rel(node, "of_type", ent(inc.type, "EVENT_TYPE"))
            for t in incident_targets(inc.direction):
                rel(node, "about", ent(target_node(t), "TARGET"))
            rel(node, "verdict", ent(f"verdict {inc.verdict}", "VERDICT"))
            rel(node, "in_session", ent(f"session {inc.session_id[:8]}", "SESSION"))
            if inc.review:
                rel(node, "reviewed", ent(f"examiner {inc.review}ed", "REVIEW"))
            if inc.rule:
                rule_id = inc.rule.split()[0].upper()
                rel(node, "applied_rule", ent(f"{rule_id} (cited)", "RULE_REF"))
        return ExtractionResult(ents, rels)


class ProctorKnowledge:
    def __init__(self, root: str | os.PathLike, policy_dir: Path = POLICY_DIR,
                 metadata_ttl_s: float = 90 * 86400.0, clock=time.time, llm=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy_dir = Path(policy_dir)
        self.metadata_ttl_s = metadata_ttl_s
        self.clock = clock
        self.llm = llm                                  # optional NIMTextLLM for examiner answers
        self.lock = threading.RLock()
        self.rules: list[Rule] = []
        self.incidents: dict[str, Incident] = {}
        self.graph = nx.MultiDiGraph()
        self._dirty = True
        self._communities: Optional[dict] = None
        self._load()

    @classmethod
    def from_env(cls) -> "ProctorKnowledge":
        root = Path(os.environ.get("PROCTOR_DATA_DIR", "data/proctoring")) / "knowledge"
        llm = NIMTextLLM.from_env()
        return cls(root, llm=llm)

    # ---- persistence ---------------------------------------------------------------
    @property
    def _incident_file(self) -> Path:
        return self.root / "incidents.jsonl"

    def _load(self) -> None:
        self.rules = load_rules(self.policy_dir)
        if self._incident_file.exists():
            for line in self._incident_file.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.incidents[d["evidence_id"]] = Incident(**d)
        self._dirty = True

    def _persist(self) -> None:
        tmp = self._incident_file.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(i.__dict__) + "\n" for i in self.incidents.values()))
        tmp.replace(self._incident_file)

    def reload_rules(self) -> int:
        with self.lock:
            self.rules = load_rules(self.policy_dir)
            self._dirty = True
        return len(self.rules)

    # ---- indexing (GraphRAG) -----------------------------------------------------------
    def _units(self) -> list[tuple[str, str]]:
        units = [(f"rule:{r.id}", "RULE " + json.dumps(r.to_dict())) for r in self.rules]
        units += [(f"incident:{i.evidence_id}", "INCIDENT " + json.dumps(i.__dict__)) for i in self.incidents.values()]
        return units

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        extractor = ProctorExtractor()
        results = [extractor.extract(uid, text) for uid, text in self._units()]
        # RULE_REF "R-03 (cited)" -> link the incident to the actual rule node
        self.graph = build_graph(results)
        for r in self.rules:
            ref = f"{r.id} (cited)"
            if self.graph.has_node(ref):
                self.graph.add_edge(ref, rule_node(r), relation="is")
        self._communities = None
        self._dirty = False

    def communities(self) -> dict:
        with self.lock:
            self._rebuild()
            if self._communities is None:
                level0 = detect_communities(self.graph)
                level1 = build_hierarchy(self.graph, level0)
                summ = ExtractiveSummarizer()
                self._communities = {
                    "level0": level0, "level1": level1,
                    "level0_summaries": summarize_all_communities(self.graph, level0, 0, summ),
                    "level1_summaries": summarize_all_communities(self.graph, level1, 1, summ),
                }
            return self._communities

    def stats(self) -> dict:
        c = self.communities()
        with self.lock:
            types = Counter(d.get("type") for _, d in self.graph.nodes(data=True))
            return {**graph_stats(self.graph), "rules": len(self.rules), "incidents": len(self.incidents),
                    "reviewed_incidents": sum(1 for i in self.incidents.values() if i.review),
                    "entity_types": dict(types), "level0_communities": len(c["level0_summaries"]),
                    "level1_communities": len(c["level1_summaries"])}

    # ---- retrieval for the judge -----------------------------------------------------------
    def _rule(self, rid: str) -> Optional[Rule]:
        return next((r for r in self.rules if r.id == rid), None)

    def context_for(self, ev, session_id: Optional[str] = None, max_rules: int = 3, max_cases: int = 3) -> dict:
        """Rules governing this flag (ranked) + what happened in similar past cases."""
        e = ev.event
        targets = incident_targets(e.direction)
        with self.lock:
            self._rebuild()
            g = self.graph
            if not g.has_node(e.type):
                return {"rules": [], "cases": []}
            # 1) rules: in-neighbours of the event-type node via 'governs' edges
            target = targets[0] if targets else ""
            candidates = [{"direction": "in", "relation": "governs", "entity": u}
                          for u, _, d in g.in_edges(e.type, data=True) if d.get("relation") == "governs"]
            transcript = (getattr(ev, "transcript", None) or {}).get("text", "")
            query = " ".join(filter(None, [e.type.replace("_", " ").lower(), " ".join(targets),
                                           " ".join(str(v) for v in e.details.values() if isinstance(v, str)),
                                           transcript[:200]]))
            ranked = rerank_local_neighbors(candidates, query)       # GraphRAG local rerank
            scored = []
            for rn in ranked:
                rid = rn.entity.split(" ", 1)[0]
                rule = self._rule(rid)
                if rule is None:
                    continue
                score = rn.score
                if any(t in rule.targets for t in targets):
                    score += 5                                     # 2-hop: rule -targets-> this object/person
                if len(rule.applies_to) > 4:
                    score -= 2                                     # general rules after specific ones
                score += {"critical": 0.6, "major": 0.3}.get(rule.severity, 0.0)
                scored.append((score, rule))
            scored.sort(key=lambda x: -x[0])
            rules = [{"id": r.id, "title": r.title, "severity": r.severity, "text": r.brief()}
                     for _, r in scored[:max_rules]]
            # 2) past cases: incidents of the same type (and target), examiner decisions first
            cases, same = self._similar(e.type, target)
            outcome = Counter(i.outcome for i in same)
            out_cases = []
            if same:
                stats = ", ".join(f"{n} {o}" for o, n in outcome.most_common(4))
                label = f"{e.type}" + (f" ({e.direction})" if e.direction else "")
                out_cases.append({"summary": f"{len(same)} earlier {label} flags: {stats}"})
            out_cases += [{"summary": i.summary(), "evidence_id": i.evidence_id} for i in cases[:max_cases - 1]]
            # 3) this session's history (pattern / escalation, R-12)
            sid = session_id
            history = [i for i in self.incidents.values() if sid and i.session_id == sid and
                       (i.review == "confirm" or (i.review is None and i.verdict == "fraud"))]
            if history:
                out_cases.insert(0, {"summary": f"this session already has {len(history)} fraud incident(s): "
                                                + ", ".join(sorted({i.type for i in history}))})
            return {"rules": rules, "cases": out_cases[:max_cases]}

    def _similar(self, etype: str, target: str) -> tuple[list[Incident], list[Incident]]:
        g = self.graph
        same = []
        for u, _, d in g.in_edges(etype, data=True):
            if d.get("relation") == "of_type" and u.startswith("Incident "):
                inc = self._incident_by_node(u)
                if inc:
                    same.append(inc)
        if target:
            matching = [i for i in same if target in incident_targets(i.direction)]
            if matching:
                same = matching
        ranked = sorted(same, key=lambda i: (i.review is None, -i.at))
        return ranked, same

    def _incident_by_node(self, node: str) -> Optional[Incident]:
        short = node.split(" ", 1)[1]
        return next((i for eid, i in self.incidents.items() if eid.startswith(short)), None)

    # ---- learning from verdicts and examiners ----------------------------------------------------
    def record_incident(self, session_id: str, ev, verdict: dict, evidence_id: Optional[str] = None) -> None:
        eid = evidence_id or getattr(ev, "evidence_id", None) or ev.event.id
        # The model's reason can quote what was said; for clips with speech keep no free text.
        reason = "" if getattr(ev, "speech_s", 0) >= 0.5 else str(verdict.get("reason") or "")[:160]
        inc = Incident(eid, session_id, self.clock(), ev.event.type, ev.event.direction,
                       verdict.get("verdict", "error"), float(verdict.get("confidence") or 0.0),
                       rule=str(verdict.get("rule") or ""), reason=reason,
                       model=str(verdict.get("model") or ""))
        with self.lock:
            self.incidents[eid] = inc
            self._persist()
            self._dirty = True

    def record_review(self, evidence_id: str, decision: str) -> bool:
        with self.lock:
            inc = self.incidents.get(evidence_id)
            if inc is None:
                return False
            inc.review = decision
            self._persist()
            self._dirty = True
            return True

    def forget_session(self, session_id: str) -> int:
        with self.lock:
            gone = [eid for eid, i in self.incidents.items() if i.session_id == session_id]
            for eid in gone:
                del self.incidents[eid]
            if gone:
                self._persist()
                self._dirty = True
            return len(gone)

    def purge(self) -> int:
        cutoff = self.clock() - self.metadata_ttl_s
        with self.lock:
            gone = [eid for eid, i in self.incidents.items() if i.at < cutoff]
            for eid in gone:
                del self.incidents[eid]
            if gone:
                self._persist()
                self._dirty = True
            return len(gone)

    # ---- examiner Q&A ------------------------------------------------------------------------------
    def ask(self, question: str) -> dict:
        c = self.communities()
        with self.lock:
            content = " ".join(w for w in re.findall(r"[\w-]+", question.lower()) if w not in STOPWORDS)
            local = local_search(self.graph, content or question)
            anchor = self._anchor(question)
            if anchor:                                        # domain-aware match beats token overlap
                nbrs = [{"direction": "out", "relation": d.get("relation"), "entity": v}
                        for _, v, d in self.graph.out_edges(anchor, data=True)]
                nbrs += [{"direction": "in", "relation": d.get("relation"), "entity": u}
                         for u, _, d in self.graph.in_edges(anchor, data=True)]
                local.matched_entity, local.ranked_neighbors = anchor, rerank_local_neighbors(nbrs, question)
            glob = global_search(c["level0_summaries"], question, top_k=3)
            extractive = synthesize_global_answer(glob, question, c["level0_summaries"])
            facts = self._facts(local.matched_entity) if local.matched_entity else []
            me = local.matched_entity or ""
            rules = [r for r in self.rules if me and
                     (me.startswith(r.id + " ") or me in r.applies_to
                      or me.replace("about: ", "") in r.targets)]
        context = {
            "matched_entity": local.matched_entity,
            "neighbors": [{"relation": n.relation, "entity": n.entity, "score": n.score}
                          for n in local.ranked_neighbors[:12]],
            "facts": facts,
            "rules": [{"id": r.id, "title": r.title, "text": r.brief()} for r in rules[:4]],
            "communities": [{"id": rc.community_id, "score": round(rc.score, 2), "summary": rc.summary[:400]}
                            for rc in glob.ranked_communities],
        }
        answer, method = extractive, "extractive (GraphRAG global search)"
        lines = facts[:6] + [f"[{r['id']}] {r['text']}" for r in context["rules"]]
        if lines:
            answer = " ".join(lines)
            method = "graph traversal (GraphRAG local search)"
        if self.llm is not None:
            try:
                import hashlib
                from semantic_cache import shared_cache
                # answers are tied to the current rules + incidents; any change (new case, review,
                # erasure) gives a new version and purges the old cached answers
                kb = hashlib.sha1(json.dumps(self.stats(), sort_keys=True, default=str).encode()).hexdigest()[:12]
                llm = self.llm

                def compute():
                    text = llm.answer(question, context)
                    return text, int(getattr(llm, "last_tokens", 0) or 0)
                res = shared_cache().answer(question, "proctor-ask", kb, compute)
                if res["answer"]:
                    answer = res["answer"]
                    method = ("semantic cache (" + res["cache"]["backend"] + ")" if res["cache"]["hit"]
                              else f"LLM over graph context ({self.llm.model})")
                context["cache"] = res["cache"]
            except Exception as e:                          # keep the graph answer
                context["llm_error"] = str(e)[:200]
        return {"question": question, "answer": answer, "method": method, **context}

    ASK_KEYWORDS = {
        "phone": "about: phone", "mobile": "about: phone", "book": "about: book", "notes": "about: notes",
        "laptop": "about: laptop", "tablet": "about: tablet", "second screen": "about: second screen",
        "another person": "about: another person", "second person": "about: another person",
        "other voice": "about: another person", "talk": "VOICE_DETECTED", "voice": "VOICE_DETECTED",
        "speak": "VOICE_DETECTED", "whisper": "VOICE_DETECTED", "look away": "OFFSCREEN_SUSTAINED",
        "looking away": "OFFSCREEN_SUSTAINED", "glance": "REPEATED_GLANCES", "looking down": "about: down",
        "look down": "about: down", "face": "NO_FACE", "left the": "NO_FACE", "hand": "FOREIGN_HAND",
    }

    def _anchor(self, question: str) -> Optional[str]:
        q = question.lower()
        g = self.graph
        m = re.search(r"\bR-?(\d{1,3})\b", question, flags=re.I)
        if m:
            rid = f"R-{int(m.group(1)):02d}"
            node = next((n for n in g.nodes if str(n).startswith(rid + " ") and "(cited)" not in str(n)), None)
            if node:
                return node
        m = re.search(r"\b([0-9a-f]{8})\b", q)
        if m and g.has_node(f"session {m.group(1)}"):
            return f"session {m.group(1)}"
        for et in EVENT_SIGNALS:
            if et.lower() in q or et.lower().replace("_", " ") in q:
                return et if g.has_node(et) else None
        for kw, node in self.ASK_KEYWORDS.items():
            if kw in q and g.has_node(node):
                return node
        return None

    def _facts(self, entity: str) -> list[str]:
        """Aggregate incidents around a matched entity (event type, target, session, rule)."""
        g = self.graph
        incs: list[Incident] = []
        for u, _, d in g.in_edges(entity, data=True):
            if u.startswith("Incident "):
                i = self._incident_by_node(u)
                if i:
                    incs.append(i)
        if entity.startswith("Incident "):
            i = self._incident_by_node(entity)
            return [i.summary()] if i else []
        if not incs:
            return []
        by_session: dict[str, list[Incident]] = defaultdict(list)
        for i in incs:
            by_session[i.session_id[:8]].append(i)
        lines = [f"{len(incs)} incident(s) linked to '{entity}' across {len(by_session)} session(s)."]
        for sid, items in sorted(by_session.items(), key=lambda kv: -len(kv[1]))[:8]:
            outcomes = Counter(i.outcome for i in items)
            lines.append(f"Session {sid}: {len(items)} incident(s) ("
                         + ", ".join(f"{n} {o}" for o, n in outcomes.most_common()) + ").")
        return lines


class NIMChainLLM:
    """Examiner answers through exam_graph.nim.NIMText (Nemotron chain, catalogue discovery)."""

    def __init__(self):
        from exam_graph.nim import NIMText
        self.text = NIMText()
        self.last_tokens = 0

    @property
    def model(self) -> str:
        return self.text.last_model or self.text.models[0]

    def answer(self, question: str, context: dict) -> str:
        ctx = json.dumps({k: context[k] for k in ("facts", "rules", "neighbors", "communities") if context.get(k)},
                         default=str)[:6000]
        out = self.text.chat("You answer an exam examiner's question using ONLY the knowledge-graph context given "
                             "(proctoring rules and incident records). Cite rule ids and session ids. If the context "
                             "doesn't contain the answer, say so. Max 5 sentences.",
                             f"Question: {question}\nContext: {ctx}", max_tokens=400)
        self.last_tokens = self.text.last_tokens
        return out


class NIMTextLLM:
    """Optional: phrase examiner answers from the retrieved graph context with a small
    NVIDIA NIM text model (PROCTOR_KG_LLM, e.g. meta/llama-3.1-8b-instruct). The
    retrieval is GraphRAG; the LLM only writes the answer, and only from that context."""

    def __init__(self, model: str, api_key: str, base_url: str, client=None):
        import httpx
        self.model, self.api_key, self.base_url = model, api_key, base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=30)

    @classmethod
    def from_env(cls):
        model = os.environ.get("PROCTOR_KG_LLM", "").strip()
        key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_API_KEY")
        if not key or key.startswith("nvapi-xxx") or os.environ.get("PROCTOR_KG_LLM_OFF") == "1":
            return None
        if not model:                    # default: the same NVIDIA text model chain as the exam graph
            return NIMChainLLM()
        return cls(model, key, os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"))

    def answer(self, question: str, context: dict) -> str:
        ctx = json.dumps({k: context[k] for k in ("facts", "rules", "neighbors", "communities") if context.get(k)})[:6000]
        r = self.client.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"},
                             json={"model": self.model, "temperature": 0.0, "max_tokens": 300, "messages": [
                                 {"role": "system", "content": "You answer an exam examiner's question using ONLY the "
                                  "knowledge-graph context given (proctoring rules and incident records). Cite rule "
                                  "ids and session ids. If the context doesn't contain the answer, say so. Max 5 sentences."},
                                 {"role": "user", "content": f"Question: {question}\nContext: {ctx}"}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
