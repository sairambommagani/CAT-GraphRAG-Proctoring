from fastapi.testclient import TestClient

from app.main import app
from app.main import exam


def bank_by_id():
    return {q["id"]: q for q in exam.bank.list()}

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["questions_loaded"] == 50


def test_full_adaptive_test_flow_always_correct():
    """Answering every question correctly should push theta up over the run."""
    start = client.post("/start-test").json()
    session_id = start["session_id"]
    question = start["question"]

    theta_trace = []
    bank = bank_by_id()

    for _ in range(15):
        correct_index = bank[question["question_id"]]["correct_index"]
        resp = client.post("/submit-answer", json={
            "session_id": session_id,
            "question_id": question["question_id"],
            "selected_index": correct_index,
        }).json()
        assert resp["correct"] is True
        theta_trace.append(resp["theta_estimate"])
        if resp["finished"]:
            break
        question = resp["next_question"]

    # ability estimate should trend upward as correct answers accumulate
    assert theta_trace[-1] > theta_trace[0]

    result = client.get(f"/result/{session_id}").json()
    assert result["questions_administered"] == 15
    assert result["correct_count"] == 15
    assert result["proficiency"] in ("Advanced", "Expert")
    assert sum(t["total"] for t in result["topic_breakdown"]) == 15
    # blueprint: 5 sections x 20% of 15 items -> exactly 3 per section
    assert [s["total"] for s in result["sections"]] == [3, 3, 3, 3, 3]
    assert all(s["score"] == 1.0 and not s["missed_concepts"] for s in result["sections"])
    assert result["assessment"] == "Python & Machine Learning Assessment"
    assert "eligibility" not in result


def test_full_adaptive_test_flow_always_wrong():
    start = client.post("/start-test").json()
    session_id = start["session_id"]
    question = start["question"]

    for _ in range(15):
        bank = bank_by_id()
        correct_index = bank[question["question_id"]]["correct_index"]
        wrong_index = (correct_index + 1) % len(question["options"])
        resp = client.post("/submit-answer", json={
            "session_id": session_id,
            "question_id": question["question_id"],
            "selected_index": wrong_index,
        }).json()
        assert resp["correct"] is False
        if resp["finished"]:
            break
        question = resp["next_question"]

    result = client.get(f"/result/{session_id}").json()
    assert result["proficiency"] in ("Novice", "Beginner")
    assert all(s["missed_concepts"] for s in result["sections"])      # what to study, per section


def test_submitting_wrong_question_id_is_rejected():
    start = client.post("/start-test").json()
    resp = client.post("/submit-answer", json={
        "session_id": start["session_id"],
        "question_id": "not-the-current-question",
        "selected_index": 0,
    })
    assert resp.status_code == 400


def test_unknown_session_returns_404():
    resp = client.post("/submit-answer", json={
        "session_id": "does-not-exist",
        "question_id": "q001",
        "selected_index": 0,
    })
    assert resp.status_code == 404
