"""Acceptance tests for Chandan's three requirements, run through the real
session pipeline (frames -> features -> baseline -> detector -> clip -> judge -> alert):

 1. track the candidate's eye/head movement (all four directions)
 2. unusual activity is flagged and a trimmed clip goes to the AI judge
 3. if the judge confirms fraud, a popup (alert) is raised

Regressions from the real-webcam recordings on 2026-09-25 are included:
calibration failing for narrow-eyed candidates (EAR < 0.20), and left/right/up
producing no popup."""
import asyncio

import cv2
import numpy as np
import pytest

from proctor.judge import MockJudge
from proctor.retention import EvidenceStore
from proctor.service import ProctorSession
from synth import ScriptedLandmarker, looking_at, looking_down_at_lap, synthetic_face

FPS = 8
JPEG = cv2.imencode(".jpg", np.full((360, 480, 3), 90, np.uint8))[1].tobytes()


class Harness:
    def __init__(self, tmp_path, eye_open=1.0, head_gain=1.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.eye_open, self.head_gain = eye_open, head_gain
        self.fn = self.at(0, 0)
        self.session = ProctorSession("s1", ScriptedLandmarker(lambda t: self.fn()), MockJudge(),
                                      EvidenceStore(tmp_path / "store"))
        self.t = 0.0
        self.events, self.alerts, self.states = [], [], []

    def at(self, sx, sy, eye_open=None):
        eo = self.eye_open if eye_open is None else eye_open
        return lambda: [looking_at(sx, sy, self.rng, 1.0, head_gain=self.head_gain, eye_open=eo)]

    async def feed(self, fn, seconds):
        self.fn = fn
        for _ in range(int(round(seconds * FPS))):
            status, events, ready = self.session.ingest(self.t, JPEG)
            self.events += events
            self.states.append(self.session.detector.state)
            for ev in ready:
                rec = await self.session.process_evidence(ev, self._notify)
            self.t += 1 / FPS

    async def _notify(self, msg):
        if msg["type"] == "alert":
            self.alerts.append(msg)

    async def calibrate(self):
        self.session.calib_target("center")
        await self.feed(self.at(0, 0), 3.0)
        self.session.calib_target_end()
        return self.session.calib_finish()


def run(coro):
    return asyncio.run(coro)


# ---- calibration can't fail on a visible face --------------------------------------

@pytest.mark.parametrize("eye_open", [1.0, 0.8, 0.65, 0.5])
def test_calibration_succeeds_for_any_eye_shape(tmp_path, eye_open):
    """eye_open=0.65 -> EAR ~0.19: the old 5-point calibration failed here (the recording)."""
    h = Harness(tmp_path, eye_open=eye_open)
    res = run(h.calibrate())
    assert res["ok"] and res["mode"] == "monitoring", res


def test_calibration_without_face_retries_then_presence_only(tmp_path):
    h = Harness(tmp_path)
    async def go():
        h.session.calib_target("center"); await h.feed(lambda: [], 3); h.session.calib_target_end()
        r1 = h.session.calib_finish()
        h.session.calib_target("center"); await h.feed(lambda: [], 3); h.session.calib_target_end()
        return r1, h.session.calib_finish()
    r1, r2 = run(go())
    assert r1["retry"] and not r1["ok"]
    assert r2["mode"] == "presence_only"


# ---- requirement 1+2+3 in every direction ------------------------------------------------

@pytest.mark.parametrize("eye_open", [1.0, 0.65])
@pytest.mark.parametrize("target,direction", [((2.6, 0.0), "right"), ((-2.6, 0.0), "left"),
                                              ((0.0, -2.6), "up"), ((0.0, 2.6), "down")])
def test_sustained_look_away_flags_judges_and_pops_up(tmp_path, target, direction, eye_open):
    h = Harness(tmp_path, eye_open=eye_open)
    async def go():
        assert (await h.calibrate())["ok"]
        await h.feed(h.at(0.1, 0.1), 3)
        await h.feed(h.at(*target), 5)          # look away 5 s
        await h.feed(h.at(0.0, 0.0), 5)         # back; post-roll elapses -> judge -> alert
    run(go())
    assert [(e.type, e.direction) for e in h.events] == [("OFFSCREEN_SUSTAINED", direction)]
    assert len(h.alerts) == 1 and direction in h.alerts[0]["message"]
    items = h.session.store.list_evidence("s1")
    assert len(items) == 1 and items[0]["verdict"]["verdict"] == "fraud" and items[0]["n_frames"] == 8


def test_eyes_only_look_away_is_flagged(tmp_path):
    """Candidate keeps the head still and only moves the eyes to the side."""
    h = Harness(tmp_path, head_gain=0.0)
    async def go():
        await h.calibrate()
        await h.feed(h.at(0, 0), 2)
        await h.feed(h.at(2.2, 0), 5)
        await h.feed(h.at(0, 0), 5)
    run(go())
    assert [e.direction for e in h.events] == ["right"] and h.alerts


def test_looking_down_at_lap_with_drooping_eyelids(tmp_path):
    h = Harness(tmp_path)
    async def go():
        await h.calibrate()
        await h.feed(h.at(0, 0), 2)
        await h.feed(lambda: [looking_down_at_lap(h.rng, 1.0)], 5)
        await h.feed(h.at(0, 0), 5)
    run(go())
    assert [e.direction for e in h.events] == ["down"] and h.alerts


def test_repeated_glances_flagged(tmp_path):
    h = Harness(tmp_path)
    async def go():
        await h.calibrate()
        for _ in range(3):
            await h.feed(h.at(0, 0), 3)
            await h.feed(h.at(-2.6, 0), 1.2)    # 1.2 s glances to the left
        await h.feed(h.at(0, 0), 5)
    run(go())
    assert [(e.type, e.direction) for e in h.events] == [("REPEATED_GLANCES", "left")] and h.alerts


def test_second_person_and_absence(tmp_path):
    h = Harness(tmp_path)
    async def go():
        await h.calibrate()
        await h.feed(h.at(0, 0), 2)
        await h.feed(lambda: [looking_at(0, 0), synthetic_face(scale=0.5, shift=(170, -30))], 2)
        await h.feed(h.at(0, 0), 4)
        await h.feed(lambda: [], 4)
        await h.feed(h.at(0, 0), 4)
    run(go())
    assert [e.type for e in h.events] == ["MULTIPLE_FACES", "NO_FACE"] and len(h.alerts) == 2


# ---- things that must NOT raise a popup --------------------------------------------------

def test_normal_exam_behaviour_is_quiet(tmp_path):
    """Reading all four answer options, blinking, a brief 1 s glance, looking at the
    bottom of the screen - over ~40 s - must produce no event."""
    h = Harness(tmp_path, eye_open=0.8)
    async def go():
        await h.calibrate()
        for sx, sy in [(-0.6, -0.5), (0.6, -0.5), (-0.6, 0.3), (0.6, 0.3), (0.0, 0.9), (-0.9, 0.9), (0.9, -0.9)]:
            await h.feed(h.at(sx, sy), 3)
            await h.feed(h.at(sx, sy, eye_open=0.08), 0.25)      # blink
        await h.feed(h.at(2.4, 0), 1.0)                            # one brief glance
        await h.feed(h.at(0, 0), 8)
    run(go())
    assert h.events == [] and h.alerts == []


# ---- popup timing and wording (demo feedback 2026-09-28) ----------------------

from proctor.service import alert_message, flag_message


def test_messages_name_exactly_what_happened():
    assert flag_message("OFFSCREEN_SUSTAINED", "up", {"duration_s": 4.2})["title"] == "Looking up"
    assert "You looked up, away from the screen for about 4 seconds" in alert_message("OFFSCREEN_SUSTAINED", "up", {"duration_s": 4.2})
    assert "to your left" in alert_message("OFFSCREEN_SUSTAINED", "left", {"duration_s": 3})
    assert "3 times" in alert_message("REPEATED_GLANCES", "right", {"glances": 3})
    assert flag_message("PROHIBITED_OBJECT", "phone")["title"] == "Phone detected"
    assert "keep your eyes on the screen" in flag_message("NO_FACE", None)["message"]


def test_only_the_ai_confirmed_popup_reaches_the_candidate(tmp_path):
    """One popup per incident: nothing is shown until the AI judge confirms fraud."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from proctor.api import create_router
    from proctor.service import SessionManager
    import struct

    h = Harness(tmp_path)
    mgr = SessionManager(lambda: ScriptedLandmarker(lambda t: h.fn()), MockJudge(), h.session.store)
    app = FastAPI(); app.include_router(create_router(mgr))
    client = TestClient(app)
    v = client.get("/proctor/policy").json()["version"]
    sid = client.post("/proctor/sessions", json={"consent": True, "policy_version": v}).json()["session_id"]
    t, seen = 0.0, []
    with client.websocket_connect(f"/proctor/ws/{sid}") as ws:
        ws.receive_json()
        def push(secs, fn):
            nonlocal t
            h.fn = fn
            for _ in range(int(secs * FPS)):
                ws.send_bytes(struct.pack("<d", t * 1000) + JPEG)
                t += 1 / FPS
                if round(t * FPS) % 2 == 0:
                    m = ws.receive_json(); seen.append(m)
                    while m["type"] != "status":
                        m = ws.receive_json(); seen.append(m)
        ws.send_json({"type": "calib_target", "name": "center"}); push(3, h.at(0, 0))
        ws.send_json({"type": "calib_target_end"}); ws.send_json({"type": "calib_finish"})
        while ws.receive_json()["type"] != "calibration":
            pass
        push(2, h.at(0, 0)); push(5, h.at(0, -2.6)); push(4, h.at(0, 0))
        for _ in range(100):
            if any(m["type"] == "alert" for m in seen):
                break
            seen.append(ws.receive_json())
        ws.send_json({"type": "end"})
    kinds = [m["type"] for m in seen if m["type"] in ("flag", "alert")]
    assert kinds == ["alert"]                      # no separate heads-up
    alert = next(m for m in seen if m["type"] == "alert")
    assert alert["title"] == "Looking up" and "You looked up" in alert["message"]
