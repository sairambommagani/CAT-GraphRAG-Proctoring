"""Graph-grounded question generation with NVIDIA NIM, plus automatic answer-key verification.

For a topic, the generator retrieves from the knowledge graph (graph.topic_context):
the topic passage, its concepts, prerequisite topics and their concepts, and concept
relations. The NIM model writes MCQs that:

* test one named concept of the topic (so every item is tagged and blueprint-countable),
* quote the passage sentence that proves the answer (source_quote - checked verbatim-ish),
* include some items that connect the topic with a prerequisite concept (multi-hop via the
  graph - the questions plain RAG over one chunk can't write),
* spread over easy / medium / hard (difficulty priors for the CAT).

Every draft then passes checks before an examiner even sees it:
  1. structure: 4 distinct options, one valid key, concept belongs to the topic
  2. grounding: source_quote really appears in the syllabus passage (token overlap >= 0.8)
  3. independent verification: a second, separate NIM call answers the question from the
     passage WITHOUT seeing the key; the draft is kept only if it picks the same option
  4. dedupe against the bank (near-duplicate text)
Survivors go to the bank as status=draft; the examiner approves before candidates see them.
"""
from __future__ import annotations

import random
import re
from typing import Optional

from .bank import norm_tokens

GEN_SYSTEM = (
    "You write assessment exam questions. Use ONLY facts stated in the syllabus passage you are "
    "given; never add outside facts. Each question has exactly 4 options, exactly one correct, and "
    "plausible distractors that reflect common misconceptions. Do not use 'all of the above' or "
    "'none of the above'. Reply with a JSON array only."
)

GEN_USER = """Section: {section}
Topic: {topic}
Passage: {passage}
Concepts of this topic: {concepts}
Prerequisite topics and their concepts: {prereq}
Concept relations: {relations}

Write {n} multiple-choice questions for this topic: {mix}.
{multihop}Each item is a JSON object:
{{"text": "...", "options": ["...","...","...","..."], "correct_index": 0-3,
  "concept": "one concept name from the list above",
  "difficulty": "easy|medium|hard",
  "source_quote": "the exact sentence from the passage that proves the answer",
  "explanation": "one sentence"}}"""

VERIFY_SYSTEM = ("You are taking an exam. Answer using only the passage. "
                 'Reply with JSON only: {"answer_index": 0-3, "confidence": 0.0-1.0}')


def _overlap(quote: str, passage: str) -> float:
    q = norm_tokens(quote)
    return len(q & norm_tokens(passage)) / len(q) if q else 0.0


class QuestionGenerator:
    def __init__(self, llm, graph, bank, rng: Optional[random.Random] = None):
        self.llm, self.graph, self.bank = llm, graph, bank
        self.rng = rng or random.Random()

    def generate(self, topic_name: str, n: int = 3) -> dict:
        topic = self.graph.syllabus.topic(topic_name)
        if topic is None:
            raise KeyError(f"unknown topic {topic_name!r}")
        ctx = self.graph.topic_context(topic)
        mix = ", ".join(["1 easy", "1 medium", "1 hard"][: max(1, min(n, 3))])
        if n > 3:
            mix += f", and {n - 3} more of mixed difficulty"
        multihop = ("At least one question must connect this topic with a prerequisite concept listed above.\n"
                    if ctx["prerequisites"] else "")
        raw = self.llm.chat_json(GEN_SYSTEM, GEN_USER.format(
            section=ctx["section"], topic=ctx["topic"], passage=ctx["passage"], concepts=ctx["concepts"],
            prereq=ctx["prerequisites"] or "none", relations=ctx["relations"] or "none", n=n, mix=mix,
            multihop=multihop), max_tokens=2500, temperature=0.4)
        model = getattr(self.llm, "last_model", None)
        report = {"topic": topic.name, "requested": n, "returned": 0, "accepted": [], "rejected": []}
        items = raw if isinstance(raw, list) else [raw]
        report["returned"] = len(items)
        drafts = []
        for it in items:
            ok, why = self._check(it, topic, ctx)
            if not ok:
                txt = it.get("text", "") if isinstance(it, dict) else it
                report["rejected"].append({"text": str(txt)[:120], "reason": why})
                continue
            it = self._shuffle(it)
            verified, why = self._verify(it, ctx["passage"])
            if not verified:
                report["rejected"].append({"text": it["text"][:120], "reason": why})
                continue
            dup = self.bank.is_duplicate(it["text"])
            if dup:
                report["rejected"].append({"text": it["text"][:120], "reason": f"near-duplicate of {dup}"})
                continue
            drafts.append({"text": it["text"].strip(), "options": [str(o).strip() for o in it["options"]],
                           "correct_index": int(it["correct_index"]), "difficulty": it["difficulty"],
                           "section": topic.section, "topic_name": topic.name, "concepts": [it["concept"]],
                           "source_quote": it["source_quote"][:400], "explanation": str(it.get("explanation", ""))[:300],
                           "generator_model": model, "verified": True})
        report["accepted"] = self.bank.add_drafts(drafts)
        return report

    def _check(self, it, topic, ctx) -> tuple[bool, str]:
        if not isinstance(it, dict):
            return False, "malformed item (not an object)"
        if not isinstance(it.get("text"), str) or not it["text"].strip():
            return False, "malformed item (no question text)"
        if not isinstance(it.get("source_quote", ""), str):
            return False, "malformed item (source_quote)"
        if not isinstance(it.get("options"), list) or not all(isinstance(o, (str, int, float)) for o in it["options"]):
            return False, "malformed item (options)"
        try:
            opts = it["options"]
            key = int(it["correct_index"])
        except (KeyError, TypeError, ValueError):
            return False, "malformed item"
        if not isinstance(opts, list) or len(opts) != 4 or len({str(o).strip().lower() for o in opts}) != 4:
            return False, "needs 4 distinct options"
        if not 0 <= key < 4:
            return False, "invalid answer key"
        if any(re.search(r"\b(all|none) of the above\b", str(o), re.I) for o in opts):
            return False, "all/none of the above"
        allowed = {c.lower() for c in topic.concepts} | {c.lower() for p in ctx["prerequisites"] for c in p["concepts"]}
        if str(it.get("concept", "")).lower() not in allowed:
            return False, f"concept {it.get('concept')!r} is not in the syllabus graph"
        it["concept"] = next(c for c in topic.concepts + [c for p in ctx["prerequisites"] for c in p["concepts"]]
                             if c.lower() == str(it["concept"]).lower())
        if str(it.get("difficulty")) not in ("easy", "medium", "hard"):
            it["difficulty"] = "medium"
        quote = str(it.get("source_quote", ""))
        if _overlap(quote, ctx["passage"]) < 0.8:
            return False, "source_quote not found in the syllabus passage (ungrounded)"
        return True, ""

    def _shuffle(self, it: dict) -> dict:
        """Models put the key in position 0 far too often; shuffle so the key position is uniform."""
        opts = list(it["options"])
        right = opts[int(it["correct_index"])]
        self.rng.shuffle(opts)
        return {**it, "options": opts, "correct_index": opts.index(right)}

    def _verify(self, it: dict, passage: str) -> tuple[bool, str]:
        letters = "ABCD"
        user = (f"Passage: {passage}\n\nQuestion: {it['text']}\n"
                + "\n".join(f"{i}. {o}" for i, o in enumerate(it["options"])))
        try:
            out = self.llm.chat_json(VERIFY_SYSTEM, user, max_tokens=120, temperature=0.0)
            ans = int(out.get("answer_index"))
        except Exception as e:
            return False, f"verification failed ({str(e)[:60]})"
        if ans != int(it["correct_index"]):
            return False, f"verifier chose {letters[ans] if 0 <= ans < 4 else ans}, key is {letters[int(it['correct_index'])]}"
        return True, ""


def structure_document(llm, text: str, title_hint: str = "") -> str:
    """Convert a raw syllabus document (PDF/DOCX text) into the syllabus markdown format,
    for the examiner to review before it replaces the live syllabus."""
    system = ("Convert course material into this exact markdown format, using only content from "
              "the document: '# Title', then for each exam section '## Section' followed by 'weight: <0-1>', "
              "then '### Topic' blocks with an optional 'requires: <other topic names>' line and 1-3 paragraphs "
              "of the document's own text, with the key concepts wrapped in **bold**. Reply with the markdown only.")
    return llm.chat(system, f"Title hint: {title_hint}\nDocument:\n{text[:24000]}", max_tokens=6000, temperature=0.1)
