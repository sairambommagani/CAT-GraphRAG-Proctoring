"""Live per-candidate question generation, subjects and accounts."""
import ast
import re

import pytest

from exam_graph.live import LiveQuestionService, difficulty_for
from exam_graph.service import ExamService


class FakeLLM:
    """Writes one grounded question from the prompt's own passage; the verifier finds the key."""
    last_model, last_tokens = "nvidia/fake", 100

    def __init__(self):
        self.n = 0
        self.prompts = []

    def chat_json(self, system, user, **kw):
        if "taking an exam" in system:                       # verifier
            opts = re.findall(r"^(\d)\. (.*)$", user, re.M)
            return {"answer_index": next(int(i) for i, o in opts if o.startswith("RIGHT")), "confidence": 0.9}
        self.prompts.append(user)
        self.n += 1
        passage = re.search(r"Passage: (.*)", user).group(1)
        concepts = ast.literal_eval(re.search(r"Concepts of this topic: (.*)", user).group(1))
        diff = re.search(r"at (easy|medium|hard) difficulty", user).group(1)
        return [{"text": f"Unique question number {self.n} about {concepts[0]} variant {self.n * 7}?",
                 "options": [f"RIGHT {self.n}", f"wrong a{self.n}", f"wrong b{self.n}", f"wrong c{self.n}"],
                 "correct_index": 0, "concept": concepts[0], "difficulty": diff,
                 "source_quote": passage.split(". ")[0], "explanation": "x"}]

    def chat(self, *a, **k):
        return "answer"


@pytest.fixture
def svc(tmp_path):
    return ExamService(tmp_path, llm=FakeLLM())


def test_difficulty_follows_ability():
    assert difficulty_for(-1.2) == "easy" and difficulty_for(0.0) == "medium" and difficulty_for(1.3) == "hard"


def test_subjects_are_built_from_the_syllabus(svc):
    subs = svc.subjects()
    assert subs["python"]["sections"] == ["Python Basics", "OOP", "Data Structures & Algorithms"]
    assert subs["ml"]["sections"] == ["ML Fundamentals", "Model Evaluation"] and "full" in subs


def _api(svc, monkeypatch):
    import app.main as m
    from fastapi.testclient import TestClient
    monkeypatch.setattr(m, "exam", svc)
    monkeypatch.setattr(m, "_check_proctoring", lambda sid: None)
    return TestClient(m.app)


def test_live_exam_generates_every_question_for_the_candidate(svc, monkeypatch):
    c = _api(svc, monkeypatch)
    monkeypatch.setenv("CAT_REQUIRE_LOGIN", "1")
    assert c.post("/start-test", json={"subject": "ml"}).status_code == 401          # must sign in
    tok = c.post("/auth/register", json={"username": "ravi", "password": "secret1", "name": "Ravi"}).json()["token"]
    assert c.post("/auth/login", json={"username": "ravi", "password": "nope"}).status_code == 401
    h = {"Authorization": f"Bearer {tok}"}
    r = c.post("/start-test", json={"subject": "ml"}, headers=h).json()
    assert r["subject"] == "Machine Learning" and r["candidate"] == "Ravi"
    q, sid, sections, diffs = r["question"], r["session_id"], set(), []
    assert q["max_questions"] == 10
    while True:
        assert q["source"] == "live"
        sections.add(q["topic"])
        diffs.append(q["difficulty"])
        key = svc.bank.get(q["question_id"])["correct_index"]
        out = c.post("/submit-answer", json={"session_id": sid, "question_id": q["question_id"],
                                             "selected_index": key if len(diffs) <= 4 else (key + 1) % 4}).json()
        if out["finished"]:
            break
        q = out["next_question"]
    assert sections == {"ML Fundamentals", "Model Evaluation"}                  # only the chosen subject
    assert diffs[0] == "medium" and "hard" in diffs[1:5]                       # correct answers -> harder
    assert diffs[-1] in ("medium", "easy")                                      # wrong answers -> easier
    res = c.get(f"/result/{sid}").json()
    assert res["assessment"] == "Machine Learning"
    assert {s["section"] for s in res["sections"]} == {"ML Fundamentals", "Model Evaluation"}
    assert all(q["status"] == "live" for q in svc.bank.list("live"))


def test_falls_back_to_the_bank_when_generation_fails(svc, monkeypatch):
    c = _api(svc, monkeypatch)
    monkeypatch.setattr(svc.live, "generate", lambda plan, asked: None)
    r = c.post("/start-test", json={"subject": "python"}).json()
    assert r["question"]["source"] == "bank" and r["question"]["topic"] in (
        "Python Basics", "OOP", "Data Structures & Algorithms")


def test_wrong_answer_probes_the_prerequisite_topic(svc):
    from app.irt.engine import Response
    syl = svc.syllabus
    t = next(t for t in syl.topics if t.requires and syl.topic(t.requires[0]) and
             syl.topic(t.requires[0]).section == t.section)

    class S:
        administered = [{"section": t.section, "topic_name": t.name, "concepts": t.concepts[:1]}]
        responses = [Response(question_id="x", a=1, b=0, correct=False)]
    plan = svc.live.plan(S, -0.5, [t.section], {t.section: 1.0})
    assert plan["topic"] in t.requires and "prerequisite" in plan["focus"] and plan["missed"]


def test_performance_carries_to_the_next_attempt_and_result_reflects_the_syllabus(svc, monkeypatch):
    c = _api(svc, monkeypatch)
    monkeypatch.setenv("CAT_REQUIRE_LOGIN", "1")
    tok = c.post("/auth/register", json={"username": "meena", "password": "secret1"}).json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    r = c.post("/start-test", json={"subject": "ml"}, headers=h).json()
    sid, q = r["session_id"], r["question"]
    while True:                                                   # get everything right
        key = svc.bank.get(q["question_id"])["correct_index"]
        out = c.post("/submit-answer", json={"session_id": sid, "question_id": q["question_id"],
                                             "selected_index": (key + 1) % 4 if q["question_number"] <= 2 else key}).json()
        if out["finished"]:
            break
        q = out["next_question"]
    res = c.get(f"/result/{sid}").json()
    assert res["recommendations"] and res["recommendations"][0]["concepts"]       # study plan from misses
    assert len(res["history"]) == 1 and res["history"][0]["theta"] > 0.5
    # second attempt starts at the earlier ability, not at 0
    r2 = c.post("/start-test", json={"subject": "ml"}, headers=h).json()
    assert r2["question"]["difficulty"] == "hard"
