"""GraphRAG exam knowledge graph: syllabus graph, tagging, blueprint CAT, NIM question generation
(mocked), examiner workflow, analytics/calibration."""
import json
import random

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.irt.engine import p_correct
from exam_graph.analytics import ResponseLog, calibrate_b, item_stats
from exam_graph.cat_blueprint import section_results, select_next_balanced
from exam_graph.nim import NIMText
from exam_graph.service import ExamService, create_exam_router
from exam_graph.syllabus import load_syllabus, parse_syllabus

# hand-labelled topic of every seed question (ground truth for the tagger)
GOLD = {**{f"q{i:03d}": t for i, t in [(1, "Types and Operators"), (2, "Functions and Generators"),
        (3, "Types and Operators"), (4, "Collections and Indexing"), (5, "Types and Operators"),
        (6, "Collections and Indexing"), (7, "Collections and Indexing"), (8, "Functions and Generators"),
        (9, "Collections and Indexing"), (10, "References and Mutability"), (11, "Classes and Objects"),
        (12, "Classes and Objects"), (13, "Classes and Objects"), (14, "Inheritance and Overriding"),
        (15, "Inheritance and Overriding"), (16, "Encapsulation and Class Design"),
        (17, "Encapsulation and Class Design"), (18, "Multiple Inheritance and MRO"),
        (19, "Multiple Inheritance and MRO"), (20, "Encapsulation and Class Design")]},
        **{f"q{i:03d}": "Complexity and Arrays" for i in (21, 22, 23)},
        **{f"q{i:03d}": "Sorting" for i in (24, 25)}, **{f"q{i:03d}": "Trees and Heaps" for i in (26, 27, 28)},
        **{f"q{i:03d}": "Dynamic Programming and Union-Find" for i in (29, 30)},
        "q031": "Learning Paradigms", "q032": "Overfitting and Regularization", "q033": "Learning Paradigms",
        "q034": "Optimization", "q035": "Overfitting and Regularization", "q036": "Optimization",
        "q037": "Overfitting and Regularization", "q038": "Deep Network Training",
        "q039": "Overfitting and Regularization", "q040": "Deep Network Training",
        **{f"q{i:03d}": "Classification Metrics" for i in range(41, 46)}, "q046": "ROC and AUC",
        "q047": "ROC and AUC", "q048": "Validation Strategy", "q049": "Validation Strategy",
        "q050": "Probability Calibration"}


class FakeNIM:
    """Scripted NIM text model: generation replies from `gen`, verifier from `verify`."""

    models = ["fake-nemotron"]
    last_model = "fake-nemotron"

    def __init__(self, gen=None, verify=None):
        self.gen, self.verify, self.calls = gen or [], verify, []

    def chat_json(self, system, user, **kw):
        self.calls.append((system, user))
        if "taking an exam" in system:
            if self.verify is not None:
                return self.verify(user)
            # answer by matching the option the generator marked correct
            for item in self.gen:
                if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"] in user:
                    right = item["options"][item["correct_index"]]
                    for line in user.splitlines():
                        if line[:3].rstrip(". ").isdigit() and line.split(". ", 1)[1] == right:
                            return {"answer_index": int(line.split(".")[0]), "confidence": 0.9}
            return {"answer_index": 0}
        if "extract relations" in system:
            return [{"source": "L1 regularization", "relation": "contrasts_with", "target": "L2 regularization"},
                    {"source": "L1 regularization", "relation": "invented_by", "target": "Tibshirani"}]
        return self.gen

    def chat(self, system, user, **kw):
        return "# Cert\n## Sec A\nweight: 1\n### Topic A\nText about **thing**.\n"


@pytest.fixture
def svc(tmp_path):
    return ExamService(tmp_path / "exam", syllabus_path=_copy_syllabus(tmp_path), llm=FakeNIM())


def _copy_syllabus(tmp_path):
    src = load_syllabus()
    p = tmp_path / "syllabus.md"
    from exam_graph.syllabus import SYLLABUS_DIR
    p.write_text(sorted(SYLLABUS_DIR.glob("*.md"))[0].read_text(encoding="utf-8"), encoding="utf-8")
    return p


# ---- syllabus + graph ------------------------------------------------------------------------------

def test_syllabus_parses_into_sections_topics_concepts():
    s = load_syllabus()
    assert s.title == "Python & Machine Learning Assessment" and s.exam_length == 15
    assert [x.name for x in s.sections] == ["Python Basics", "OOP", "Data Structures & Algorithms",
                                             "ML Fundamentals", "Model Evaluation"]
    assert len(s.topics) == 20 and sum(s.weights().values()) == pytest.approx(1.0)
    t = s.topic("Deep Network Training")
    assert t.requires == ["Optimization", "Overfitting and Regularization"]
    assert "vanishing gradient problem" in t.concepts and "**" not in t.text


def test_graph_built_with_graphrag_pipeline(svc):
    st = svc.graph.stats()
    assert st["sections"] == 5 and st["topics"] == 20 and st["concepts"] > 80 and st["questions_linked"] == 50
    assert st["level0_communities"] >= 5
    g = svc.graph.graph
    assert g.has_edge("topic: Deep Network Training", "topic: Optimization")
    assert g.has_edge("section: ML Fundamentals", "topic: Optimization")
    assert g.has_edge("question q039", "concept: l1 regularization")


def test_seed_questions_are_tagged_accurately(svc):
    """Measured against hand labels: the graph tagger must put >= 90% of questions in the right topic."""
    hits = sum(svc.bank.get(q)["topic_name"] == t for q, t in GOLD.items())
    assert hits / len(GOLD) >= 0.9, hits
    assert all(svc.bank.get(q)["concepts"] for q in GOLD)


def test_topic_context_walks_prerequisites(svc):
    ctx = svc.graph.topic_context(svc.syllabus.topic("Deep Network Training"))
    assert {p["topic"] for p in ctx["prerequisites"]} == {"Optimization", "Overfitting and Regularization"}
    assert "Gradient descent" in ctx["prerequisites"][0]["concepts"] + ctx["prerequisites"][1]["concepts"]
    assert "vanishing gradient" in ctx["passage"]


def test_llm_concept_relations_are_constrained_to_the_syllabus(svc):
    n = svc.graph.extract_concept_relations()
    assert n == 1                                            # 'invented_by' / unknown entity rejected
    assert svc.graph.graph.has_edge("concept: l1 regularization", "concept: l2 regularization")


def test_coverage_points_at_topics_needing_questions(svc):
    cov = svc.graph.coverage()
    assert cov[0]["need"] > 0 and cov[0]["approved_questions"] < 3
    assert any(r["untested_concepts"] for r in cov)


# ---- blueprint CAT + decision ------------------------------------------------------------------------

def test_blueprint_balances_sections_and_uses_information_within(svc):
    bank = svc.bank.approved()
    weights = svc.syllabus.weights()
    administered, theta = [], 0.0
    for _ in range(15):
        cands = [q for q in bank if q not in administered]
        q = select_next_balanced(theta, cands, administered, weights)
        administered.append(q)
    per = {s: sum(q["section"] == s for q in administered) for s in weights}
    assert set(per.values()) == {3}
    # no two consecutive questions from the same topic when avoidable
    assert all(a["topic_name"] != b["topic_name"] for a, b in zip(administered, administered[1:]))


def test_uneven_weights_are_respected(svc):
    weights = {"Python Basics": 0.6, "OOP": 0.4}
    bank = [q for q in svc.bank.approved() if q["section"] in weights]
    adm = []
    for _ in range(10):
        adm.append(select_next_balanced(0.0, [q for q in bank if q not in adm], adm, weights))
    assert sum(q["section"] == "Python Basics" for q in adm) == 6


def test_section_results_list_missed_concepts(svc):
    qs = [svc.bank.get("q039"), svc.bank.get("q001")]
    rows = section_results(qs, [False, True], svc.syllabus)
    ml = next(r for r in rows if r["section"] == "ML Fundamentals")
    assert ml["total"] == 1 and ml["score"] == 0.0 and "L1 regularization" in ml["missed_concepts"]
    assert next(r for r in rows if r["section"] == "OOP")["total"] == 0


# ---- NIM question generation ----------------------------------------------------------------------------

PASSAGE_Q = "Too high a learning rate makes training diverge or oscillate, and too low a rate makes it very slow."


def item(text, key=1, concept="learning rate", quote=PASSAGE_Q, difficulty="medium", options=None):
    return {"text": text, "options": options or ["It converges faster", "Training diverges or oscillates",
                                                "Nothing changes", "The loss becomes zero"],
            "correct_index": key, "concept": concept, "difficulty": difficulty, "source_quote": quote,
            "explanation": "Stated in the syllabus."}


def test_generation_keeps_only_grounded_verified_items(svc):
    good = item("What happens when the gradient descent learning rate is too high?")
    ungrounded = item("Who invented gradient descent?", quote="Cauchy invented gradient descent in 1847.")
    bad_concept = item("What is Adam?", concept="Adam optimizer")
    aota = item("Which is true?", options=["A", "B", "C", "All of the above"])
    svc.llm.gen = [good, ungrounded, bad_concept, aota]
    rep = svc.generator.generate("Optimization", n=4)
    assert [q["text"] for q in rep["accepted"]] == [good["text"]]
    reasons = " ".join(r["reason"] for r in rep["rejected"])
    assert "ungrounded" in reasons and "not in the syllabus graph" in reasons and "all/none" in reasons
    q = rep["accepted"][0]
    assert q["status"] == "draft" and q["section"] == "ML Fundamentals" and q["concepts"] == ["learning rate"]
    assert q["options"][q["correct_index"]] == "Training diverges or oscillates"   # key survives shuffling
    assert q["b"] == 0.0 and q["generator_model"] == "fake-nemotron" and q["verified"]
    assert q not in svc.bank.approved()                      # drafts never reach candidates


def test_generation_prompt_is_graph_grounded_and_multihop(svc):
    svc.llm.gen = []
    svc.generator.generate("Deep Network Training", n=3)
    user = svc.llm.calls[-1][1]
    assert "Passage:" in user and "vanishing gradient" in user
    assert "Optimization" in user and "Gradient descent" in user          # prerequisite concepts (1 hop)
    assert "connect this topic with a prerequisite concept" in user


def test_verifier_disagreement_rejects(svc):
    svc.llm.gen = [item("What happens when the learning rate is too high?")]
    svc.llm.verify = lambda user: {"answer_index": 3}
    svc.generator.rng = random.Random(0)
    rep = svc.generator.generate("Optimization", n=1)
    assert rep["accepted"] == [] and "verifier chose" in rep["rejected"][0]["reason"]


def test_answer_key_positions_are_shuffled(svc):
    stems = ["What happens with an excessive step size?", "A learning rate far above the ideal leads to what?",
             "Which symptom shows a learning rate set too large?", "Raising the learning rate too much causes?",
             "What is the risk of an overly aggressive learning rate?", "Too big a gradient step typically produces?",
             "An engineer multiplies the learning rate by 100. Expected effect?",
             "Which outcome follows from a learning rate that is too high?"]
    svc.llm.gen = [item(stems[i], key=0,
                        options=["Training diverges or oscillates", "It converges faster", "Nothing changes",
                                 "The loss becomes exactly zero"]) for i in range(8)]
    svc.generator.rng = random.Random(1)
    rep = svc.generator.generate("Optimization", n=6)
    positions = {q["correct_index"] for q in rep["accepted"]}
    assert len(rep["accepted"]) >= 6 and len(positions) >= 3


def test_duplicates_of_bank_questions_are_dropped(svc):
    dup = svc.bank.get("q036")
    svc.llm.gen = [item(dup["text"])]
    rep = svc.generator.generate("Optimization", n=1)
    assert rep["accepted"] == [] and "near-duplicate of q036" in rep["rejected"][0]["reason"]


# ---- examiner API -----------------------------------------------------------------------------------------

def _client(svc):
    app = FastAPI()
    app.include_router(create_exam_router(svc, "tok"))
    return TestClient(app), {"X-Admin-Token": "tok"}


def test_examiner_workflow_generate_review_approve(svc):
    c, h = _client(svc)
    assert c.get("/exam/admin/graph").status_code == 401
    g = c.get("/exam/admin/graph", headers=h).json()
    assert g["topics"] == 20 and len(g["blueprint"]["sections"]) == 5
    svc.llm.gen = [item("What happens when the gradient descent learning rate is set too high?")]
    rep = c.post("/exam/admin/generate", json={"topic": "Optimization", "per_topic": 1}, headers=h).json()
    qid = rep["reports"][0]["accepted"][0]["id"]
    assert [q["id"] for q in c.get("/exam/admin/questions?status=draft", headers=h).json()] == [qid]
    bad = c.post(f"/exam/admin/questions/{qid}/review", json={"decision": "approve", "edits": {"correct_index": 9}},
                 headers=h)
    assert bad.status_code == 400
    ok = c.post(f"/exam/admin/questions/{qid}/review", json={"decision": "approve"}, headers=h).json()
    assert ok["status"] == "approved" and svc.bank.get(qid) in svc.bank.approved()
    assert svc.graph.graph.has_edge(f"question {qid}", "concept: learning rate")
    # persisted
    again = ExamService(svc.data_dir, syllabus_path=svc.syllabus_path, llm=None)
    assert again.bank.get(qid)["status"] == "approved"


def test_generate_without_key_is_503(tmp_path):
    s = ExamService(tmp_path / "e", syllabus_path=_copy_syllabus(tmp_path), llm=None)
    c, h = _client(s)
    assert c.post("/exam/admin/generate", json={}, headers=h).status_code == 503


def test_syllabus_edit_reindexes_graph_and_retags(svc):
    c, h = _client(svc)
    md = c.get("/exam/admin/syllabus", headers=h).json()["markdown"]
    md = md.replace("### Probability Calibration\nrequires: ROC and AUC",
                    "### Probability Calibration\nrequires: ROC and AUC, Validation Strategy")
    st = c.put("/exam/admin/syllabus", json={"markdown": md}, headers=h).json()
    assert st["topics"] == 20
    assert svc.graph.graph.has_edge("topic: Probability Calibration", "topic: Validation Strategy")
    assert c.put("/exam/admin/syllabus", json={"markdown": "no structure"}, headers=h).status_code == 400


def test_convert_uploaded_document_to_syllabus_draft(svc):
    c, h = _client(svc)
    r = c.post("/exam/admin/syllabus/convert", files={"file": ("course.txt", b"Some course text", "text/plain")},
               headers=h).json()
    assert r["topics"] == 1 and r["markdown"].startswith("# Cert")
    assert svc.syllabus.title == "Python & Machine Learning Assessment"   # draft only, not applied


def test_examiner_questions_over_the_graph(svc):
    c, h = _client(svc)
    a = c.post("/exam/admin/ask", json={"question": "What does Deep Network Training require?"}, headers=h).json()
    g = lambda a: a.get("graph_answer", a["answer"])          # retrieval result (before the LLM/cache)
    assert "Optimization" in g(a)
    a = c.post("/exam/admin/ask", json={"question": "Which concepts have no questions yet? coverage"},
               headers=h).json()
    assert "untested concepts" in g(a)
    a = c.post("/exam/admin/ask", json={"question": "tell me about l1 regularization"}, headers=h).json()
    assert "q039" in g(a)


# ---- analytics + calibration -----------------------------------------------------------------------------

def test_online_calibration_recovers_difficulty():
    rng = np.random.default_rng(0)
    thetas = rng.normal(0, 1, 400)
    true_b = 0.8
    ys = [int(rng.random() < p_correct(t, 1.0, true_b)) for t in thetas]
    assert calibrate_b(list(thetas), ys, 1.0, b0=0.0) == pytest.approx(true_b, abs=0.2)


def test_item_stats_flags_leaks_and_calibrates(svc, tmp_path):
    log = ResponseLog(tmp_path / "r.jsonl")
    rng = np.random.default_rng(1)
    for i in range(60):
        sid = f"s{i}"
        th = float(rng.normal(0, 1))
        log.record(sid, "q050", True, 4.0)                         # hard item, everyone right -> leaked
        log.record(sid, "q036", bool(rng.random() < p_correct(th, 1.7, 0.6)), 20.0)
        log.finish(sid, th)
    st = item_stats(log, svc.bank, min_n=30)
    rows = {r["id"]: r for r in st["items"]}
    assert any("possible leak" in f for f in rows["q050"]["flags"])
    assert not any("leak" in f for f in rows["q036"]["flags"])
    assert "q036" in st["calibrated"] and st["responses"] == 120 and st["sessions"] == 60


def test_cat_logs_responses_for_analytics():
    from app.main import app, exam
    client = TestClient(app)
    start = client.post("/start-test").json()
    q = start["question"]
    client.post("/submit-answer", json={"session_id": start["session_id"], "question_id": q["question_id"],
                                        "selected_index": exam.bank.get(q["question_id"])["correct_index"]})
    answers, _ = exam.log.rows()
    assert any(a["session"] == start["session_id"] and a["correct"] for a in answers)


# ---- NIM client ----------------------------------------------------------------------------------------------

def test_nim_text_chain_falls_back_and_strips_reasoning():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if "nemotron" in body["model"]:
            return httpx.Response(404, json={"detail": "not enabled"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '<think>hmm</think>[{"a": 1}]'}}],
                                         "usage": {"total_tokens": 10}})
    llm = NIMText(api_key="nvapi-test", client=httpx.Client(transport=httpx.MockTransport(handler)), usage_path="")
    assert llm.chat_json("sys", "user") == [{"a": 1}]
    assert [b["model"] for b in seen] == ["nvidia/llama-3.3-nemotron-super-49b-v1.5", "meta/llama-3.3-70b-instruct"]
    assert seen[0]["messages"][0]["content"].startswith("/no_think")
    assert llm.last_model == "meta/llama-3.3-70b-instruct"


# ---- review fixes (independent code review) -----------------------------------------------------------------

@pytest.mark.parametrize("edits", [{"correct_index": "1"}, {"b": "hard"}, {"options": ["x"]},
                                   {"correct_index": None}, {"b": "<img src=x onerror=alert(1)>"}, {"id": "zzz"}])
def test_bad_review_edits_are_rejected_and_not_applied(svc, edits):
    before = json.dumps(svc.bank.get("q001"), sort_keys=True)
    with pytest.raises(ValueError):
        svc.bank.review("q001", "approve", "examiner", edits)
    assert json.dumps(svc.bank.get("q001"), sort_keys=True) == before


def test_valid_review_edit_is_typed(svc):
    q = svc.bank.review("q001", "approve", "examiner", {"correct_index": 2, "b": 1})
    assert q["correct_index"] == 2 and q["b"] == 1.0 and isinstance(q["b"], float)


def test_section_without_questions_is_not_assessed(svc):
    rows = section_results([svc.bank.get("q001")], [True], svc.syllabus)
    oop = next(r for r in rows if r["section"] == "OOP")
    assert oop["total"] == 0 and oop["score"] is None


def test_graph_warns_about_empty_sections(svc):
    for q in svc.bank.list():
        if q["section"] == "OOP":
            svc.bank.review(q["id"], "reject", "examiner")
    c, h = _client(svc)
    assert any("OOP" in w for w in c.get("/exam/admin/graph", headers=h).json()["warnings"])


def test_corrupt_log_line_does_not_break_startup(tmp_path):
    p = tmp_path / "exam"
    p.mkdir()
    (p / "responses.jsonl").write_text('{"session": "s1", "question": "q001", "correct": true}\n{"sess')
    s = ExamService(p, syllabus_path=_copy_syllabus(tmp_path), llm=None)
    assert s.refresh_analytics()["responses"] == 1


def test_calibration_skips_uniform_and_leaked_items(svc, tmp_path):
    log = ResponseLog(tmp_path / "r.jsonl")
    for i in range(40):
        log.record(f"s{i}", "q001", True)
        log.finish(f"s{i}", 0.0)
    assert "q001" not in item_stats(log, svc.bank, min_n=30)["calibrated"]
    assert -4 < calibrate_b([0.0] * 30, [1] * 30, 1.0, b0=0.5) < 0.5     # prior keeps it finite


@pytest.mark.parametrize("bad", [["just a string"], [{"options": ["a", "b", "c", "d"]}], [{"text": 5}]])
def test_malformed_llm_items_are_rejected_not_500(svc, bad):
    good = item("What happens when the gradient descent learning rate is too high?")
    svc.llm.gen = bad + [good]
    rep = svc.generator.generate("Optimization", n=2)
    assert [q["text"] for q in rep["accepted"]] == [good["text"]] and "malformed" in rep["rejected"][0]["reason"]


def test_exam_grades_the_question_as_served_and_blocks_double_submit():
    from app.main import app, exam
    client = TestClient(app)
    start = client.post("/start-test").json()
    q = start["question"]
    orig = exam.bank.get(q["question_id"])
    key = orig["correct_index"]
    exam.bank.review(q["question_id"], "approve", "examiner", {"correct_index": (key + 1) % len(orig["options"])})
    try:
        r = client.post("/submit-answer", json={"session_id": start["session_id"], "question_id": q["question_id"],
                                                "selected_index": key}).json()
        assert r["correct"] is True                       # graded against what the candidate saw
        again = client.post("/submit-answer", json={"session_id": start["session_id"],
                                                    "question_id": q["question_id"], "selected_index": key})
        assert again.status_code == 400                   # no double-recording
    finally:
        exam.bank.review(q["question_id"], "approve", "examiner", {"correct_index": key})


def test_retired_text_models_fall_back_to_the_catalogue():
    """NVIDIA retires free-API models (HTTP 410): the generator finds a working one in the key's catalogue."""
    seen = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "nvidia/nemotron-3-super-120b-a12b"},
                                                      {"id": "nvidia/nv-embedqa-e5-v5"}]})
        body = json.loads(request.content)
        seen.append(body["model"])
        if body["model"] != "nvidia/nemotron-3-super-120b-a12b":
            return httpx.Response(410, json={"detail": "model retired"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}], "usage": {}})
    llm = NIMText(api_key="nvapi-test", client=httpx.Client(transport=httpx.MockTransport(handler)), usage_path="")
    assert llm.chat_json("s", "u") == []
    assert seen[-1] == "nvidia/nemotron-3-super-120b-a12b" and "nvidia/nv-embedqa-e5-v5" not in seen
