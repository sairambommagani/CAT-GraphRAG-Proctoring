"""Question bank store: seed questions + NIM-generated questions with an examiner approval workflow.

Every question carries its graph tags (section, topic, concepts), IRT parameters (a, b),
provenance (seed / generated + model + source passage) and a status:

    draft     generated and verified, waiting for the examiner (never shown to candidates)
    approved  in the live bank the CAT draws from
    rejected  kept for audit / dedupe, never shown

Generated questions start with difficulty *priors* (easy/medium/hard -> b = -1 / 0 / +1,
a = 1.0); analytics.py re-estimates them from real responses (online calibration).
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Optional

SEED_BANK = Path(__file__).resolve().parent.parent / "app" / "data" / "question_bank.json"
DIFFICULTY_B = {"easy": -1.0, "medium": 0.0, "hard": 1.0}


def norm_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def near_duplicate(a: str, b: str, threshold: float = 0.8) -> bool:
    ta, tb = norm_tokens(a), norm_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def _validated_edits(q: dict, edits: dict) -> dict:
    """Examiner edits, type-checked on a copy; nothing is applied unless all of it is valid."""
    allowed = {"text", "options", "correct_index", "b", "explanation"}
    unknown = set(edits) - allowed
    if unknown:
        raise ValueError(f"cannot edit {sorted(unknown)}")
    new = {k: q.get(k) for k in allowed} | dict(edits)
    if not isinstance(new["text"], str) or not new["text"].strip():
        raise ValueError("text must be a non-empty string")
    opts = new["options"]
    if not (isinstance(opts, list) and 2 <= len(opts) <= 6 and all(isinstance(o, str) and o.strip() for o in opts)
            and len({o.strip().lower() for o in opts}) == len(opts)):
        raise ValueError("options must be 2-6 distinct non-empty strings")
    key = new["correct_index"]
    if not isinstance(key, int) or isinstance(key, bool) or not 0 <= key < len(opts):
        raise ValueError("correct_index must be an integer index into options")
    b = new["b"]
    if not isinstance(b, (int, float)) or isinstance(b, bool) or not -4.0 <= float(b) <= 4.0:
        raise ValueError("b must be a number in [-4, 4]")
    if new.get("explanation") is not None and not isinstance(new["explanation"], str):
        raise ValueError("explanation must be a string")
    return {k: new[k] for k in edits} | ({"b": float(b)} if "b" in edits else {})


class QuestionBank:
    def __init__(self, path: Path, graph, seed_path: Path = SEED_BANK):
        self.path = Path(path)
        self.graph = graph
        self.lock = threading.RLock()
        self.questions: dict[str, dict] = {}
        if self.path.exists():
            for q in json.loads(self.path.read_text(encoding="utf-8")):
                self.questions[q["id"]] = q
        else:
            self._import_seed(seed_path)
        self.retag()

    def _import_seed(self, seed_path: Path) -> None:
        for q in json.loads(Path(seed_path).read_text(encoding="utf-8")):
            q = {**q, "section": q.get("topic"), "status": "approved", "origin": "seed"}
            self.questions[q["id"]] = q

    def retag(self) -> None:
        """(Re)derive section/topic/concept tags from the knowledge graph, then link the bank
        into the graph (question --tests--> concept)."""
        with self.lock:
            for q in self.questions.values():
                tags = self.graph.tag_question(q)
                q["section"] = tags["section"] or q.get("section")
                q["topic_name"] = tags["topic_name"]
                if not q.get("concepts_locked"):
                    q["concepts"] = tags["concepts"]
                q["topic"] = q["section"]                    # CAT API field = exam section
            self.graph.link_questions(list(self.questions.values()))
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(self.questions.values()), indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # ---- queries ---------------------------------------------------------------------------------
    def approved(self) -> list[dict]:
        with self.lock:
            return [q for q in self.questions.values() if q.get("status") == "approved"]

    def get(self, qid: str) -> Optional[dict]:
        return self.questions.get(qid)

    def list(self, status: Optional[str] = None) -> list[dict]:
        with self.lock:
            return [q for q in self.questions.values() if status is None or q.get("status") == status]

    def is_duplicate(self, text: str) -> Optional[str]:
        with self.lock:
            for q in self.questions.values():
                if near_duplicate(text, q.get("text", "")):
                    return q["id"]
        return None

    # ---- workflow -------------------------------------------------------------------------------
    def _next_id(self) -> str:
        n = 1 + max([int(m.group(1)) for q in self.questions for m in [re.match(r"g(\d+)$", q)] if m] or [0])
        return f"g{n:04d}"

    def add_drafts(self, drafts: list[dict]) -> list[dict]:
        added = []
        with self.lock:
            for d in drafts:
                if self.is_duplicate(d["text"]):
                    continue
                qid = self._next_id()
                q = {"id": qid, "status": "draft", "origin": "generated", "created_at": time.time(),
                     "a": 1.0, "b": DIFFICULTY_B.get(d.get("difficulty", "medium"), 0.0), **d, "concepts_locked": True}
                q["id"] = qid
                q["topic"] = q.get("section")
                self.questions[qid] = q
                added.append(q)
            self.graph.link_questions(list(self.questions.values()))
            self._save()
        return added

    def review(self, qid: str, decision: str, reviewer: str, edits: Optional[dict] = None) -> dict:
        if decision not in ("approve", "reject"):
            raise ValueError("decision must be approve or reject")
        with self.lock:
            q = self.questions.get(qid)
            if q is None:
                raise KeyError(qid)
            if edits:
                q.update(_validated_edits(q, edits))
            q["status"] = "approved" if decision == "approve" else "rejected"
            q["review"] = {"decision": decision, "reviewer": reviewer, "at": time.time()}
            self.graph.link_questions(list(self.questions.values()))
            self._save()
            return q

    def update_params(self, params: dict[str, dict]) -> None:
        """Calibrated IRT parameters from analytics (only questions with enough responses)."""
        with self.lock:
            for qid, p in params.items():
                if qid in self.questions:
                    self.questions[qid].update(p)
            self._save()
