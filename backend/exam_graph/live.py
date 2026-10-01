"""Live (per-candidate) question generation: every next question is written for THIS
candidate, from THIS subject's syllabus graph, at THIS candidate's current ability.

How the next question is chosen
-------------------------------
1. Section  - blueprint balancing inside the chosen subject (Kingsbury & Zara): the section
              furthest below its share of the test so far.
2. Topic    - GraphRAG over the syllabus graph + the candidate's answers so far:
                * a topic whose concept the candidate just got WRONG -> its prerequisite topic
                  (probe the gap underneath), else
                * the least-covered topic of the section that hasn't been asked yet.
3. Difficulty - CAT: target difficulty b = current ability estimate theta (maximum Fisher
              information for a 2PL item is at b = theta), mapped to easy / medium / hard
              for the prompt, and stored as the item's b prior.
4. Generate - NVIDIA NIM writes ONE question from the topic passage, grounded and checked
              (structure, concept in the graph, source quote in the passage, independent
              verifier picks the same answer, not a duplicate).

Latency: while the candidate is answering question k, BOTH possible next questions are
generated in the background (one for "if correct" -> harder, one for "if wrong" -> easier),
so the next question is usually ready the moment they submit. If generation fails or is
too slow, the CAT falls back to the best approved bank question for that section, so an
exam never stalls.

Generated items are stored in the bank with status "live" (audit trail; an examiner can
approve good ones into the permanent bank). Their difficulty is an LLM-estimated prior and
is refined from real responses by the analytics/calibration step like any new item.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from app.irt.engine import Response, estimate_ability

from .bank import norm_tokens
from .generator import GEN_SYSTEM, QuestionGenerator

LIVE_USER = """Section: {section}
Topic: {topic}
Passage: {passage}
Concepts of this topic: {concepts}
Prerequisite topics and their concepts: {prereq}
Concept relations: {relations}

The candidate's current ability is {level} (difficulty target: {difficulty}).
{weak}Write exactly 1 multiple-choice question for this topic at {difficulty} difficulty
{focus}. It must be different from these questions already asked: {asked}
Reply with a JSON array containing one object:
{{"text": "...", "options": ["...","...","...","..."], "correct_index": 0-3,
  "concept": "one concept name from the lists above",
  "difficulty": "{difficulty}",
  "source_quote": "the exact sentence from the passage that proves the answer",
  "explanation": "one sentence"}}"""


def difficulty_for(theta: float) -> str:
    return "easy" if theta < -0.6 else "hard" if theta > 0.6 else "medium"


def level_for(theta: float) -> str:
    return ("beginner" if theta < -1 else "developing" if theta < 0 else "proficient" if theta < 1 else "advanced")


class LiveQuestionService:
    def __init__(self, llm, graph, bank, wait_s: float = 25.0, workers: int = 4):
        self.llm, self.graph, self.bank = llm, graph, bank
        self.gen = QuestionGenerator(llm, graph, bank)
        self.wait_s = wait_s
        self.first_wait_s = 30.0
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="live-q")
        self.lock = threading.Lock()
        self.stats = {"generated": 0, "fallback": 0, "failed": 0, "seconds": []}

    @property
    def available(self) -> bool:
        return self.llm is not None

    # ---- planning ---------------------------------------------------------------------------
    def plan(self, session, theta: float, sections: list[str], weights: dict[str, float]) -> Optional[dict]:
        """-> {section, topic, difficulty, theta, focus} for the next question."""
        syl = self.graph.syllabus
        counts: dict[str, int] = {}
        for q in session.administered:
            counts[q.get("section")] = counts.get(q.get("section"), 0) + 1
        n_next = sum(counts.values()) + 1
        sec = max(sections, key=lambda s: (weights.get(s, 0) * n_next - counts.get(s, 0), -sections.index(s)))
        section = syl.section(sec)
        if section is None or not section.topics:
            return None
        asked_topics = [q.get("topic_name") for q in session.administered]
        focus, topic = "", None
        # GraphRAG: last wrong answer in this section -> probe its prerequisite topic
        for q, r in zip(reversed(session.administered), reversed(session.responses)):
            if q.get("section") == sec and not r.correct:
                t = syl.topic(q.get("topic_name") or "")
                prereqs = [syl.topic(p) for p in (t.requires if t else [])]
                prereqs = [p for p in prereqs if p is not None and p.section == sec]
                if prereqs:
                    topic = min(prereqs, key=lambda p: asked_topics.count(p.name))
                    focus = (f"focusing on the prerequisite knowledge the candidate seems to be missing "
                             f"(they got a question on '{q.get('topic_name')}' wrong)")
                break
        fixed = getattr(session, "topic", None)
        if fixed and syl.topic(fixed) and syl.topic(fixed).section == sec:
            topic = syl.topic(fixed)                              # examiner chose a single topic
        if topic is None:
            topic = min(section.topics, key=lambda t: (asked_topics.count(t.name), section.topics.index(t)))
        missed = [c for q, r in zip(session.administered, session.responses) if not r.correct
                  for c in (q.get("concepts") or [])]
        # performance from earlier attempts: concepts missed last time (only those in this section)
        sec_concepts = {c.lower() for t in section.topics for c in t.concepts}
        missed += [c for c in getattr(session, "prior_weak", []) if c.lower() in sec_concepts and c not in missed]
        if not focus and missed and not (fixed and topic.name == fixed):
            weak_topic = next((t for t in section.topics if any(c.lower() in {x.lower() for x in t.concepts}
                                                                 for c in missed)
                               and asked_topics.count(t.name) == 0), None)
            if weak_topic is not None:
                topic = weak_topic
                focus = "targeting a concept the candidate has got wrong before"
        return {"section": sec, "topic": topic.name, "difficulty": difficulty_for(theta), "theta": theta,
                "focus": focus, "missed": missed[:6]}

    # ---- generation -------------------------------------------------------------------------
    def generate(self, plan: dict, asked_texts: list[str]) -> Optional[dict]:
        t0 = time.time()
        topic = self.graph.syllabus.topic(plan["topic"])
        ctx = self.graph.topic_context(topic)
        weak = (f"Concepts the candidate answered wrongly so far: {', '.join(plan['missed'])}.\n"
                if plan.get("missed") else "")
        try:
            raw = self.llm.chat_json(GEN_SYSTEM, LIVE_USER.format(
                section=ctx["section"], topic=ctx["topic"], passage=ctx["passage"], concepts=ctx["concepts"],
                prereq=ctx["prerequisites"] or "none", relations=ctx["relations"] or "none",
                level=level_for(plan["theta"]), difficulty=plan["difficulty"], weak=weak,
                focus=plan.get("focus") or "", asked=" | ".join(t[:80] for t in asked_texts[-8:]) or "none"),
                max_tokens=900, temperature=0.5)
        except Exception as e:
            self._count("failed")
            print(f"[live-q] generation failed for {plan['topic']}: {str(e)[:120]}", flush=True)
            return None
        items = raw if isinstance(raw, list) else [raw]
        asked = [norm_tokens(t) for t in asked_texts]
        for it in items[:2]:
            ok, why = self.gen._check(it, topic, ctx)
            if not ok:
                continue
            it = self.gen._shuffle(it)
            toks = norm_tokens(it["text"])
            if any(len(toks & a) / max(1, len(toks | a)) > 0.7 for a in asked):
                continue                                          # same as a question already asked
            ok, why = self.gen._verify(it, ctx["passage"])
            if not ok:
                continue
            b = max(-2.5, min(2.5, plan["theta"]))                # target difficulty = current ability (CAT)
            item = {"text": it["text"].strip(), "options": [str(o).strip() for o in it["options"]],
                    "correct_index": int(it["correct_index"]), "difficulty": plan["difficulty"],
                    "section": topic.section, "topic_name": topic.name, "concepts": [it["concept"]],
                    "source_quote": str(it["source_quote"])[:400], "explanation": str(it.get("explanation", ""))[:300],
                    "generator_model": getattr(self.llm, "last_model", None), "verified": True,
                    "status": "live", "origin": "live-generated", "a": 1.0, "b": round(b, 2),
                    "live_plan": {k: plan[k] for k in ("difficulty", "focus") if plan.get(k)}}
            added = self.bank.add_drafts([item])
            if not added:
                continue
            secs = time.time() - t0
            with self.lock:
                self.stats["generated"] += 1
                self.stats["seconds"] = (self.stats["seconds"] + [round(secs, 1)])[-50:]
            print(f"[live-q] {topic.section} / {topic.name} ({plan['difficulty']}, theta {plan['theta']:+.2f}) "
                  f"ready in {secs:.1f}s", flush=True)
            return added[0]
        self._count("failed")
        print(f"[live-q] no valid question for {plan['topic']} (checks rejected the draft)", flush=True)
        return None

    def _count(self, k: str) -> None:
        with self.lock:
            self.stats[k] += 1

    # ---- prefetch ---------------------------------------------------------------------------
    def prefetch(self, session, sections, weights) -> None:
        """While the candidate answers the current question, prepare the next one for both
        outcomes (correct -> ability goes up, wrong -> ability goes down)."""
        cur = session.current_question
        if cur is None or session.questions_administered + 1 >= session.max_questions:
            session.prefetched = {}
            return
        asked = [q["text"] for q in session.administered] + [cur["text"]]
        futures: dict[bool, Future] = {}
        for outcome in (True, False):
            hypo = session.responses + [Response(question_id=cur["id"], a=cur["a"], b=cur["b"], correct=outcome)]
            theta = estimate_ability(hypo, getattr(session, "start_theta", 0.0)).theta

            class _S:                                             # session as it would look after this answer
                administered = session.administered + [cur]
                responses = hypo
                prior_weak = getattr(session, "prior_weak", [])
                topic = getattr(session, "topic", None)
            plan = self.plan(_S, theta, sections, weights)
            if plan:
                futures[outcome] = self.pool.submit(self.generate, plan, asked)
        session.prefetched = futures

    def take(self, session, correct: bool) -> Optional[dict]:
        fut = (getattr(session, "prefetched", None) or {}).get(correct)
        session.prefetched = {}
        if fut is None:
            return None
        try:
            return fut.result(timeout=self.wait_s)
        except Exception:
            return None

    def fallback(self) -> None:
        self._count("fallback")

    def summary(self) -> dict:
        with self.lock:
            s = dict(self.stats)
        secs = s.pop("seconds")
        s["avg_seconds"] = round(sum(secs) / len(secs), 1) if secs else None
        return s
