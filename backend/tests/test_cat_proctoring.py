"""CAT + proctoring integration: answers gated on a live, calibrated proctoring
session; malpractice alert during the test; integrity report in the result."""
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, exam, proctoring
from proctor.calibration import TARGETS
from tests.proctoring.synth import ScriptedLandmarker, looking_at


def bank_by_id():
    return {q["id"]: q for q in exam.bank.list()}

FRAME = __import__("cv2").imencode(".jpg", np.full((360, 480, 3), 90, np.uint8))[1].tobytes()
FPS = 10


class Camera:
    def __init__(self):
        self.fn = lambda: [looking_at(0, 0)]
        self.rng = np.random.default_rng(0)

    def __call__(self, t):
        return self.fn()


@pytest.fixture
def cam(monkeypatch):
    cam = Camera()
    monkeypatch.setenv("CAT_REQUIRE_PROCTORING", "1")
    monkeypatch.setattr(proctoring, "landmarker_factory", lambda: ScriptedLandmarker(cam))
    return cam


class Clock:
    t = 0.0


def push(ws, seconds, clock, statuses):
    for _ in range(int(seconds * FPS)):
        ws.send_bytes(struct.pack("<d", clock.t * 1000) + FRAME)
        clock.t += 1 / FPS
        # server emits a status every 0.25 s; read it to stay in lock-step with the worker
        if round(clock.t * FPS) % 3 == 0:
            m = ws.receive_json()
            statuses.append(m)
            while m["type"] != "status":
                m = ws.receive_json()
                statuses.append(m)


def answer(client, sid, q, correct=True):
    bank = bank_by_id()
    idx = bank[q["question_id"]]["correct_index"]
    if not correct:
        idx = (idx + 1) % len(q["options"])
    return client.post("/submit-answer", json={"session_id": sid, "question_id": q["question_id"],
                                                "selected_index": idx})


def test_answers_blocked_without_proctoring(cam):
    client = TestClient(app)
    start = client.post("/start-test").json()
    r = answer(client, start["session_id"], start["question"])
    assert r.status_code == 403 and "Proctoring is not active" in r.json()["detail"]


def test_full_proctored_cat_run(cam):
    client = TestClient(app)
    start = client.post("/start-test").json()
    sid, q = start["session_id"], start["question"]
    policy = client.get("/proctor/policy").json()
    ps = client.post("/proctor/sessions", json={"consent": True, "policy_version": policy["version"],
                                                 "exam_ref": sid}).json()
    msgs, clock = [], Clock()
    with client.websocket_connect(f"/proctor/ws/{ps['session_id']}") as ws:
        assert ws.receive_json()["type"] == "ready"
        # still calibrating -> answers refused
        r = answer(client, sid, q)
        assert r.status_code == 403 and "calibration" in r.json()["detail"]

        for name, (sx, sy) in TARGETS.items():
            cam.fn = lambda sx=sx, sy=sy: [looking_at(sx, sy, cam.rng, 1.0)]
            ws.send_json({"type": "calib_target", "name": name})
            push(ws, 2.0, clock, msgs)
            ws.send_json({"type": "calib_target_end"})
        ws.send_json({"type": "calib_finish"})
        m = ws.receive_json()
        while m["type"] != "calibration":
            m = ws.receive_json()
        assert m["ok"], m

        cam.fn = lambda: [looking_at(0.1, 0.1, cam.rng, 1.0)]
        for i in range(4):                                   # a few honest answers
            push(ws, 1.0, clock, msgs)
            r = answer(client, sid, q).json()
            q = r["next_question"]

        cam.fn = lambda: [looking_at(0.0, 2.6, cam.rng, 1.0)]   # phone below the screen
        push(ws, 5.0, clock, msgs)
        cam.fn = lambda: [looking_at(0.0, 0.0, cam.rng, 1.0)]
        push(ws, 5.0, clock, msgs)
        for _ in range(200):
            if any(x["type"] == "alert" for x in msgs):
                break
            msgs.append(ws.receive_json())
        alert = next(x for x in msgs if x["type"] == "alert")
        assert alert["event_type"] == "OFFSCREEN_SUSTAINED"

        mid = client.get(f"/result/{sid}").json()["integrity"]          # live report
        assert mid["proctored"] and mid["status"] == "alerted"

        while True:                                           # finish the test
            r = answer(client, sid, q).json()
            if r["finished"]:
                break
            q = r["next_question"]
        ws.send_json({"type": "end"})
        while ws.receive_json()["type"] != "ended":
            pass

    result = client.get(f"/result/{sid}").json()             # stored report after session end
    assert result["questions_administered"] == 15
    integ = result["integrity"]
    assert integ["status"] == "alerted" and integ["alerts"] == 1 and integ["verdicts"].get("fraud") == 1
    # no live session remains -> further answers would be refused
    assert proctoring.find_live(sid) is None


def test_unproctored_session_reports_not_proctored(monkeypatch):
    monkeypatch.setenv("CAT_REQUIRE_PROCTORING", "0")
    client = TestClient(app)
    start = client.post("/start-test").json()
    answer(client, start["session_id"], start["question"])
    integ = client.get(f"/result/{start['session_id']}").json()["integrity"]
    assert integ == {**integ, "proctored": False, "status": "not_proctored"}


def test_frontend_served_and_health():
    client = TestClient(app)
    assert client.get("/ui/").status_code == 200
    assert "proctor/proctor.js" in client.get("/ui/").text
    assert client.get("/ui/proctor/proctor.js").status_code == 200
    h = client.get("/health").json()
    assert h["proctoring"]["judge_model"]
