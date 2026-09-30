"""End-to-end: consent -> WebSocket calibration -> cheating behaviour -> judge -> popup -> retention."""
import struct

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from proctor.api import create_router
from proctor.calibration import TARGETS
from proctor.judge import MockJudge
from proctor.retention import EvidenceStore
from proctor.service import SessionManager
from synth import ScriptedLandmarker, looking_at, synthetic_face

FPS = 10
TOKEN = "t0ken"
FRAME = cv2.imencode(".jpg", np.full((360, 480, 3), 90, np.uint8))[1].tobytes()


class Script:
    """Timeline of what the fake camera 'sees'. Tests append segments."""

    def __init__(self):
        self.segments = []        # (t_end, fn)
        self.rng = np.random.default_rng(0)

    def add(self, t_end, fn):
        self.segments.append((t_end, fn))

    def __call__(self, t):
        for t_end, fn in self.segments:
            if t < t_end:
                return fn()
        return self.segments[-1][1]()


def build(tmp_path, judge=None):
    script = Script()
    store = EvidenceStore(tmp_path / "store")
    mgr = SessionManager(lambda: ScriptedLandmarker(script), judge or MockJudge(), store)
    app = FastAPI()
    app.include_router(create_router(mgr, admin_token=TOKEN))
    return TestClient(app), script, mgr, store


def drain(ws, until_type, limit=500):
    seen = []
    for _ in range(limit):
        m = ws.receive_json()
        seen.append(m)
        if m["type"] == until_type:
            return m, seen
    raise AssertionError(f"{until_type} not received; got {[s['type'] for s in seen][-10:]}")


def create_session(client):
    policy = client.get("/proctor/policy").json()
    r = client.post("/proctor/sessions", json={"consent": True, "policy_version": policy["version"],
                                               "exam_ref": "cat-123"})
    assert r.status_code == 200
    return r.json()


def calibrate(ws, script, t=0.0):
    rng = script.rng
    for name, (sx, sy) in TARGETS.items():
        script.add(t + 2.0, lambda sx=sx, sy=sy: [looking_at(sx, sy, rng, 1.0)])
        ws.send_json({"type": "calib_target", "name": name})
        # stream frames one at a time and wait for status to keep the worker in lock-step
        for i in range(20):
            ws.send_bytes(struct.pack("<d", (t + i / FPS) * 1000) + FRAME)
            _wait_status(ws, t + i / FPS)
        ws.send_json({"type": "calib_target_end"})
        t += 2.0
    ws.send_json({"type": "calib_finish"})
    cal, _ = drain(ws, "calibration")
    return cal, t


_last = {}


def _wait_status(ws, t):
    """The server sends a status at most every 0.25 s; frames in between are silent.
    Sync by only waiting when a status is due."""
    key = id(ws)
    if t - _last.get(key, -1) >= 0.25 - 1e-6:
        _last[key] = t
        m = ws.receive_json()
        while m["type"] != "status":
            m = ws.receive_json()
        return m


def stream(ws, script, t, seconds, fn, collect):
    script.add(t + seconds, fn)
    for i in range(int(seconds * FPS)):
        tt = t + i / FPS
        ws.send_bytes(struct.pack("<d", tt * 1000) + FRAME)
        if tt - _last.get(id(ws), -1) >= 0.25 - 1e-6:
            _last[id(ws)] = tt
            while True:
                m = ws.receive_json()
                collect.append(m)
                if m["type"] == "status":
                    break
    return t + seconds


def test_consent_required(tmp_path):
    client, *_ = build(tmp_path)
    assert client.post("/proctor/sessions", json={"consent": False, "policy_version": "x"}).status_code == 400
    v = client.get("/proctor/policy").json()["version"]
    assert client.post("/proctor/sessions", json={"consent": True, "policy_version": "old"}).status_code == 409
    assert client.post("/proctor/sessions", json={"consent": True, "policy_version": v}).status_code == 200


def test_full_flow_phone_below_screen_triggers_popup(tmp_path):
    _last.clear()
    client, script, mgr, store = build(tmp_path)
    s = create_session(client)
    sid = s["session_id"]
    msgs = []
    with client.websocket_connect(f"/proctor/ws/{sid}") as ws:
        assert ws.receive_json()["type"] == "ready"
        cal, t = calibrate(ws, script)
        assert cal["ok"] and cal["mode"] == "monitoring", cal
        rng = script.rng
        t = stream(ws, script, t, 3, lambda: [looking_at(0.1, 0.2, rng, 1.0)], msgs)     # normal work
        t = stream(ws, script, t, 5, lambda: [looking_at(0.0, 2.6, rng, 1.0)], msgs)     # staring at a phone below
        t = stream(ws, script, t, 5, lambda: [looking_at(0.0, 0.0, rng, 1.0)], msgs)     # back; post-roll elapses
        # the judge runs asynchronously; wait for its alert
        if not any(m["type"] == "alert" for m in msgs):
            alert, more = drain(ws, "alert")
            msgs += more
        alert = next(m for m in msgs if m["type"] == "alert")
        assert alert["event_type"] == "OFFSCREEN_SUSTAINED"
        assert "down" in alert["message"] and "examiner" in alert["message"]
        ws.send_json({"type": "ack", "event_id": alert["event_id"]})
        ws.send_json({"type": "end"})
        ended, _ = drain(ws, "ended")
    assert ended["summary"]["alerts"] == 1 and ended["summary"]["judge_calls"] >= 1
    statuses = [m for m in msgs if m["type"] == "status"]
    assert any(m["state"] == "off_screen" for m in statuses)

    # examiner view: evidence exists, encrypted frames retrievable with token only
    hdr = {"X-Admin-Token": TOKEN}
    assert client.get("/proctor/admin/evidence").status_code == 401
    ev = client.get(f"/proctor/admin/evidence?session_id={sid}", headers=hdr).json()
    assert len(ev) == 1 and ev[0]["verdict"]["verdict"] == "fraud"
    img = client.get(f"/proctor/admin/evidence/{ev[0]['evidence_id']}/frames/0.jpg", headers=hdr)
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    # live buffer wiped, session gone from memory
    assert mgr.get(sid) is None
    # examiner dismisses -> frames deleted -> 410
    client.post(f"/proctor/admin/evidence/{ev[0]['evidence_id']}/review", headers=hdr, json={"decision": "dismiss"})
    assert client.get(f"/proctor/admin/evidence/{ev[0]['evidence_id']}/frames/0.jpg", headers=hdr).status_code == 410
    # erasure
    assert client.delete(f"/proctor/admin/sessions/{sid}", headers=hdr).json()["erased_evidence"] == 1
    actions = {a["action"] for a in client.get("/proctor/admin/audit", headers=hdr).json()}
    assert {"consent", "evidence_saved", "verdict", "alert_ack", "review", "session_erased"} <= actions


def test_benign_behaviour_no_popup_and_frames_deleted(tmp_path):
    _last.clear()
    client, script, mgr, store = build(tmp_path, judge=MockJudge(fixed="benign"))
    sid = create_session(client)["session_id"]
    msgs = []
    with client.websocket_connect(f"/proctor/ws/{sid}") as ws:
        ws.receive_json()
        cal, t = calibrate(ws, script)
        rng = script.rng
        t = stream(ws, script, t, 2, lambda: [looking_at(0, 0, rng, 1.0)], msgs)
        t = stream(ws, script, t, 4, lambda: [looking_at(-2.5, -0.3, rng, 1.0)], msgs)   # long look left
        t = stream(ws, script, t, 5, lambda: [looking_at(0, 0, rng, 1.0)], msgs)
        ws.send_json({"type": "end"})
        ended, more = drain(ws, "ended")
        msgs += more
    assert not any(m["type"] == "alert" for m in msgs)
    items = store.list_evidence(sid)
    assert len(items) == 1 and items[0]["frames_deleted"]            # benign -> deleted immediately


def test_second_person_detected(tmp_path):
    _last.clear()
    client, script, mgr, store = build(tmp_path)
    sid = create_session(client)["session_id"]
    msgs = []
    with client.websocket_connect(f"/proctor/ws/{sid}") as ws:
        ws.receive_json()
        cal, t = calibrate(ws, script)
        rng = script.rng
        t = stream(ws, script, t, 2, lambda: [looking_at(0, 0, rng, 1.0)], msgs)
        t = stream(ws, script, t, 2, lambda: [looking_at(0, 0, rng, 1.0),
                                              synthetic_face(scale=0.5, shift=(180, -20))], msgs)
        t = stream(ws, script, t, 4, lambda: [looking_at(0, 0, rng, 1.0)], msgs)
        ws.send_json({"type": "end"})
        _, more = drain(ws, "ended")
        msgs += more
    alerts = [m for m in msgs if m["type"] == "alert"]
    assert alerts and alerts[0]["event_type"] == "MULTIPLE_FACES"


def test_failed_calibration_falls_back_to_presence_only(tmp_path):
    _last.clear()
    client, script, mgr, store = build(tmp_path)
    sid = create_session(client)["session_id"]
    with client.websocket_connect(f"/proctor/ws/{sid}") as ws:
        ws.receive_json()
        for attempt in range(2):
            ws.send_json({"type": "calib_finish"})               # no samples at all
            cal, _ = drain(ws, "calibration")
        assert cal["ok"] is False and cal["retry"] is False and cal["mode"] == "presence_only"
        ws.send_json({"type": "end"})
        drain(ws, "ended")


def test_unknown_session_ws_closed(tmp_path):
    client, *_ = build(tmp_path)
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/proctor/ws/nope") as ws:
            ws.receive_json()
