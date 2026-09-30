"""Microphone proctoring: VAD, lip-sync speaker attribution, audio evidence, retention.

Chandan's v2 request: "integrate with mic". The accuracy-critical part is WHO is
speaking - the candidate (lips move with the voice) or another person in the room
(voice while the candidate's lips are still) - tested here end to end through the
real session pipeline and the WebSocket API."""
import asyncio
import io
import struct
import wave

import cv2
import numpy as np
import pytest

from proctor.audio import (AudioRingBuffer, SAMPLE_RATE, SpeechConfig, SpeechMonitor, VADFrame, VoiceActivity,
                           attribute_speaker, pack_audio, to_wav, unpack_audio)
from proctor.judge import MockJudge
from proctor.knowledge import ProctorKnowledge
from proctor.retention import EvidenceStore
from proctor.service import ProctorSession
from synth import ScriptedLandmarker, looking_at, room_noise, voice, voice_stream

FPS = 8
JPEG = cv2.imencode(".jpg", np.full((360, 480, 3), 90, np.uint8))[1].tobytes()


# ---- units --------------------------------------------------------------------------

def test_wire_format_roundtrip():
    pcm = (np.arange(4000) % 300 - 150).astype(np.int16)
    t, back = unpack_audio(pack_audio(1234.5, pcm))
    assert t == pytest.approx(1.2345) and np.array_equal(back, pcm)
    assert unpack_audio(b"\xff\xd8 not audio") is None


def test_ring_buffer_window_and_retention():
    buf = AudioRingBuffer(seconds=5)
    for i in range(1, 41):                                # 40 x 0.25 s = 10 s
        buf.add(i * 0.25, np.full(4000, i, np.int16))
    assert buf.seconds_held <= 5.25
    t0, pcm = buf.window(8.0, 9.0)
    assert t0 == pytest.approx(8.0, abs=0.01) and len(pcm) == pytest.approx(16000, abs=10)
    assert buf.window(0.0, 2.0)[1].size == 0             # older audio is gone


def test_vad_separates_voice_from_room_noise():
    rng = np.random.default_rng(1)
    va = VoiceActivity()
    va.process(0.0, room_noise(3, rng), calibrating=True)
    floor = va.finish_calibration()
    assert -70 < floor < -45
    voiced = va.process(3.0, voice(2, rng))
    quiet = va.process(5.0, room_noise(2, rng))
    assert np.mean([f.speech for f in voiced]) > 0.9
    assert np.mean([f.speech for f in quiet]) < 0.05


def _rows(mouths, face=1):
    return [{"face_count": face, "mouth": m} for m in mouths]


def test_lip_sync_attribution():
    cfg = SpeechConfig()
    t = np.arange(16) / 8
    talking = 0.25 + 0.2 * np.sin(2 * np.pi * 3 * t)
    still = 0.02 + 0.003 * np.sin(t)
    assert attribute_speaker(_rows(talking), cfg).speaker == "candidate"
    assert attribute_speaker(_rows(still), cfg).speaker == "another person"
    assert attribute_speaker(_rows(still, face=0), cfg).speaker == "unattributed"
    assert attribute_speaker([], cfg).speaker == "unattributed"
    # a candidate whose lips jitter a lot even when silent needs more movement to count as talking
    assert attribute_speaker(_rows(0.05 + 0.02 * np.sin(2 * np.pi * 3 * t)), cfg,
                             mouth_baseline_std=0.012).speaker == "another person"


def test_short_sounds_do_not_fire():
    mon = SpeechMonitor()
    frames = [VADFrame(0.03 * i, i < 17, -30) for i in range(80)]      # 0.5 s cough then silence
    assert mon.update(frames, lambda a, b: _rows([0.02] * 8)) == []


def test_speech_episode_fires_once():
    mon = SpeechMonitor()
    frames = [VADFrame(0.03 * i, (i // 10) % 3 != 2, -30 if (i // 10) % 3 != 2 else -55)
              for i in range(200)]                                               # 6 s of chatty speech
    evs = mon.update(frames, lambda a, b: _rows([0.02] * 20))
    assert len(evs) == 1 and evs[0].type == "VOICE_DETECTED" and evs[0].direction == "another person"
    assert evs[0].details["speech_s"] >= 1.0


def _chunks_through(sig, calib_s=3.0, seed=0):
    """Feed audio like the browser does (250 ms chunks, jittered timestamps) through VAD + monitor."""
    rng = np.random.default_rng(seed)
    va, mon, buf, evs, calibrated = VoiceActivity(), SpeechMonitor(), AudioRingBuffer(), [], False
    for i in range(len(sig) // 4000):
        pcm = sig[i * 4000:(i + 1) * 4000]
        t_end = (i + 1) * 0.25 + rng.uniform(-0.02, 0.02)
        t0 = buf.add(t_end, pcm)
        cal = t_end <= calib_s
        frames = va.process(t0, pcm, calibrating=cal)
        if cal:
            continue
        if not calibrated:
            va.finish_calibration()
            calibrated = True
        evs += mon.update(frames, lambda a, b: [])
    return evs


@pytest.mark.skipif(not hasattr(VoiceActivity(), "vad") or VoiceActivity().vad is None, reason="webrtcvad not installed")
def test_normal_volume_syllabic_speech_is_flagged_but_typing_fan_and_hum_are_not():
    """Regression: real speech (syllables with pauses) at laptop-mic level never fired, because
    WebRTC VAD was only shown the loud frames and never learnt the room noise."""
    from synth import syllabic_speech, pink_noise, typing_clicks
    for seed in range(4):
        rng = np.random.default_rng(seed)
        sp = syllabic_speech(3.0, rng, amp=2000)                # ~ -24 dBFS peaks, no auto-gain
        bg = lambda s: pink_noise(s, rng, 100)
        talk = np.concatenate([bg(5), sp + bg(len(sp) / SAMPLE_RATE), bg(3)]).astype(np.int16)
        assert len(_chunks_through(talk, seed=seed)) == 1
        t = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
        for noise in (typing_clicks(4, rng, 6000), pink_noise(4, rng, 3000), 3000 * np.sin(2 * np.pi * 50 * t)):
            sig = np.concatenate([bg(5), noise + bg(4), bg(2)]).astype(np.int16)
            assert _chunks_through(sig, seed=seed) == []
    rng = np.random.default_rng(7)
    word = syllabic_speech(0.5, rng, amp=2000)[:SAMPLE_RATE // 2]      # a single "yes"
    sig = np.concatenate([pink_noise(5, rng, 100), word + pink_noise(0.5, rng, 100)[:len(word)],
                          pink_noise(3, rng, 100)]).astype(np.int16)
    assert _chunks_through(sig) == []


def test_wav_is_valid():
    rng = np.random.default_rng(0)
    data = to_wav(voice(1, rng))
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getnframes() == SAMPLE_RATE


# ---- through the session pipeline ------------------------------------------------------------

class FakeTranscriber:
    enabled = True

    def __init__(self, text="the answer to question five is B"):
        self.text, self.calls = text, 0

    def transcribe(self, pcm):
        self.calls += 1
        return {"text": self.text, "language": "en"}


class AVHarness:
    def __init__(self, tmp_path, judge=None, knowledge=True):
        self.rng = np.random.default_rng(3)
        self.face = lambda t: [looking_at(0, 0, self.rng, 1.0)]
        self.store = EvidenceStore(tmp_path / "store")
        self.transcriber = FakeTranscriber()
        self.kg = ProctorKnowledge(tmp_path / "kg") if knowledge else None
        self.session = ProctorSession("s1", ScriptedLandmarker(lambda t: self.face(t)), judge or MockJudge(),
                                      self.store, transcriber=self.transcriber, knowledge=self.kg)
        self.t = 0.0
        self.events, self.alerts, self.records = [], [], []

    async def _notify(self, msg):
        if msg["type"] == "alert":
            self.alerts.append(msg)

    async def feed(self, seconds, face, audio):
        """face(t) -> landmarks; audio(seconds) -> int16 samples for that slice."""
        self.face = face
        for _ in range(int(round(seconds * FPS))):
            self.events += self.session.ingest_audio(self.t + 1 / FPS, audio(1 / FPS))
            _, events, ready = self.session.ingest(self.t, JPEG)
            self.events += events
            for ev in ready:
                self.records.append(await self.session.process_evidence(ev, self._notify))
            self.t += 1 / FPS

    async def calibrate(self):
        self.session.calib_target("center")
        await self.feed(3.0, lambda t: [looking_at(0, 0, self.rng, 1.0)], lambda s: room_noise(s, self.rng))
        self.session.calib_target_end()
        return self.session.calib_finish()


def still(h):
    return lambda t: [looking_at(0.05, 0.1, h.rng, 1.0, mouth_open=0.03)]


def talking(h):
    return lambda t: [looking_at(0.05, 0.1, h.rng, 1.0, mouth_open=0.35 + 0.3 * np.sin(2 * np.pi * 3.3 * t))]


@pytest.mark.parametrize("who", ["another person", "candidate"])
def test_voice_flag_goes_to_judge_with_audio_transcript_and_rules(tmp_path, who):
    h = AVHarness(tmp_path)

    async def go():
        cal = await h.calibrate()
        assert cal["ok"] and cal["mic"]["active"] and cal["mic"]["noise_floor_db"] < -45
        await h.feed(3, still(h), lambda s: room_noise(s, h.rng))                     # quiet work
        await h.feed(4, still(h) if who == "another person" else talking(h),
                     voice_stream(h.rng))                                        # speech
        await h.feed(3, still(h), lambda s: room_noise(s, h.rng))                     # post-roll
    asyncio.run(go())

    voice_events = [e for e in h.events if e.type == "VOICE_DETECTED"]
    assert len(voice_events) == 1 and voice_events[0].direction == who
    assert not [e for e in h.events if e.type != "VOICE_DETECTED"]                   # no false video flags
    assert len(h.alerts) == 1
    title = h.alerts[0]["title"]
    assert title == ("Another voice in the room" if who == "another person" else "Talking during the exam")

    [m] = h.store.list_evidence()
    assert m["has_audio"] and m["speech_s"] >= 1.2
    assert m["transcript"]["text"] == "the answer to question five is B" and h.transcriber.calls == 1
    assert m["rules"][0] == ("R-10" if who == "another person" else "R-09")         # GraphRAG rule retrieval
    wav = h.store.load_audio(m["evidence_id"], "examiner")
    assert wav[:4] == b"RIFF" and len(wav) > SAMPLE_RATE              # the flagged seconds, not the exam
    assert any(line.startswith("voice (") for line in m["timeline"])
    # the incident is now knowledge for future judgements
    assert list(h.kg.incidents.values())[0].type == "VOICE_DETECTED"


def test_silent_candidate_no_voice_flags(tmp_path):
    h = AVHarness(tmp_path)

    async def go():
        await h.calibrate()
        await h.feed(10, still(h), lambda s: room_noise(s, h.rng))
    asyncio.run(go())
    assert h.events == [] and h.alerts == [] and h.transcriber.calls == 0


def test_benign_verdict_deletes_audio_and_transcript(tmp_path):
    h = AVHarness(tmp_path, judge=MockJudge(fixed="benign"))

    async def go():
        await h.calibrate()
        await h.feed(3, still(h), lambda s: room_noise(s, h.rng))
        await h.feed(3, still(h), voice_stream(h.rng))
        await h.feed(3, still(h), lambda s: room_noise(s, h.rng))
    asyncio.run(go())
    [m] = h.store.list_evidence()
    assert m["frames_deleted"] and "transcript" not in m
    assert h.store.load_audio(m["evidence_id"], "examiner") is None
    assert h.alerts == []


def test_audio_only_in_ram_until_flagged(tmp_path):
    h = AVHarness(tmp_path)

    async def go():
        await h.calibrate()
        await h.feed(6, still(h), lambda s: room_noise(s, h.rng))
    asyncio.run(go())
    assert h.session.audio.seconds_held > 5
    assert not list((tmp_path / "store").rglob("*.enc"))              # nothing on disk
    asyncio.run(h.session.end())
    assert h.session.audio.seconds_held == 0                        # wiped at session end


# ---- WebSocket wire protocol -------------------------------------------------------------------

def test_websocket_accepts_audio_and_examiner_can_listen(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from proctor.api import create_router
    from proctor.service import SessionManager

    rng = np.random.default_rng(5)
    state = {"face": lambda: [looking_at(0, 0, rng, 1.0, mouth_open=0.03)]}
    store = EvidenceStore(tmp_path / "store")
    kg = ProctorKnowledge(tmp_path / "kg")
    mgr = SessionManager(lambda: ScriptedLandmarker(lambda t: state["face"]()), MockJudge(), store,
                         transcriber=FakeTranscriber("tell me the answer"), knowledge=kg)
    app = FastAPI()
    app.include_router(create_router(mgr, admin_token="tok"))
    client = TestClient(app)
    policy = client.get("/proctor/policy").json()
    assert "microphone" in " ".join(policy["points"]).lower()
    sid = client.post("/proctor/sessions", json={"consent": True, "policy_version": policy["version"]}).json()["session_id"]
    msgs = []
    with client.websocket_connect(f"/proctor/ws/{sid}") as ws:
        assert ws.receive_json()["type"] == "ready"
        t = 0.0

        def step(audio_fn, n):
            nonlocal t
            for _ in range(n):
                ws.send_bytes(pack_audio((t + 1 / FPS) * 1000, audio_fn(1 / FPS)))
                ws.send_bytes(struct.pack("<d", t * 1000) + JPEG)
                t += 1 / FPS
                m = ws.receive_json()                     # one status per 2 frames at 8 fps (0.25 s)
                msgs.append(m)
                while m["type"] != "status":
                    m = ws.receive_json()
                    msgs.append(m)
                ws.send_bytes(pack_audio((t + 1 / FPS) * 1000, audio_fn(1 / FPS)))
                ws.send_bytes(struct.pack("<d", t * 1000) + JPEG)
                t += 1 / FPS

        ws.send_json({"type": "calib_target", "name": "center"})
        step(lambda s: room_noise(s, rng), 12)
        ws.send_json({"type": "calib_target_end"})
        ws.send_json({"type": "calib_finish"})
        m = ws.receive_json()
        while m["type"] != "calibration":
            m = ws.receive_json()
        assert m["ok"] and m["mic"]["active"]
        step(lambda s: room_noise(s, rng), 8)
        step(voice_stream(rng), 16)
        step(lambda s: room_noise(s, rng), 12)
        for _ in range(50):
            if any(x["type"] == "alert" for x in msgs):
                break
            msgs.append(ws.receive_json())
        assert any(x.get("mic") and x.get("speaking") for x in msgs if x["type"] == "status")
        alert = next(x for x in msgs if x["type"] == "alert")
        assert alert["title"] == "Another voice in the room"
        ws.send_json({"type": "end"})
        m = ws.receive_json()
        while m["type"] != "ended":                       # like proctor.js: wait for the server's cleanup
            m = ws.receive_json()

    hdr = {"X-Admin-Token": "tok"}
    [m] = client.get("/proctor/admin/evidence", headers=hdr).json()
    r = client.get(f"/proctor/admin/evidence/{m['evidence_id']}/audio.wav", headers=hdr)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
    assert client.get(f"/proctor/admin/evidence/{m['evidence_id']}/audio.wav").status_code == 401
    assert m["transcript"]["text"] == "tell me the answer"
    # examiner decision flows into the knowledge graph
    client.post(f"/proctor/admin/evidence/{m['evidence_id']}/review", json={"decision": "confirm"}, headers=hdr)
    assert kg.incidents[m["evidence_id"]].review == "confirm"
    ans = client.post("/proctor/admin/knowledge/ask", json={"question": "which sessions had another voice?"},
                      headers=hdr).json()
    assert sid[:8] in ans["answer"] and "confirmed by examiner" in ans["answer"]
    u = client.get("/proctor/admin/usage", headers=hdr)
    assert u.status_code == 200 and u.json()["models"][0]["available"]
    assert client.get("/proctor/admin/knowledge/stats", headers=hdr).json()["incidents"] == 1
    assert len(client.get("/proctor/admin/knowledge/rules", headers=hdr).json()) == 14
    # erasure also removes the incident from the graph
    assert client.delete(f"/proctor/admin/sessions/{sid}", headers=hdr).json()["erased_incidents"] == 1


def test_vad_reanchors_after_a_gap():
    va = VoiceActivity()
    rng = np.random.default_rng(0)
    f1 = va.process(0.0, room_noise(0.1, rng)[:1000])            # leaves a partial 30 ms frame
    f2 = va.process(5.0, room_noise(0.1, rng)[:1000])            # 5 s later (dropout)
    assert f2[0].t == pytest.approx(5.0 + 0.03, abs=1e-6)


def test_erasure_during_judging_does_not_resurrect_the_incident(tmp_path):
    class SlowJudge(MockJudge):
        def judge(self, ev):
            h.store.erase_session("s1", "dpo")                     # erased while the judge was thinking
            return super().judge(ev)
    h = AVHarness(tmp_path)
    h.session.judge = SlowJudge()

    async def go():
        await h.calibrate()
        await h.feed(3, still(h), lambda s: room_noise(s, h.rng))
        await h.feed(3, still(h), voice_stream(h.rng))
        await h.feed(5, still(h), lambda s: room_noise(s, h.rng))
    asyncio.run(go())
    assert h.kg.incidents == {} and h.store.list_evidence() == []


def test_integrity_is_never_clear_while_verdicts_are_pending(tmp_path):
    """Regression (2026-09-30 recording): the result page said 'No issues detected' after 4 alerts,
    because it was read while the session was finalising (mode=ended, judge calls in flight)."""
    from proctor.service import SessionManager
    store = EvidenceStore(tmp_path / "store")
    mgr = SessionManager(lambda: ScriptedLandmarker(lambda t: []), MockJudge(), store)
    s = mgr.create("cat-1", store.policy.version)
    s.mode = "ended"                                          # finalising
    s.events.append({"id": "ev1", "type": "OFFSCREEN_SUSTAINED", "status": "judging"})
    assert mgr.integrity_report("cat-1")["status"] == "under_review"
    s.events[0].update(verdict={"verdict": "fraud"}, alerted=True)
    assert mgr.integrity_report("cat-1")["status"] == "alerted"
    # session gone from memory and never ended cleanly: rebuilt from stored evidence
    del mgr.sessions[s.id]
    eid = store.save(s.id, {"type": "OFFSCREEN_SUSTAINED"}, [], [], [])
    store.apply_verdict(eid, {"verdict": "fraud", "confidence": 0.9})
    store.annotate(eid, alerted=True)
    r = mgr.integrity_report("cat-1")
    assert r["status"] == "alerted" and r["alerts"] == 1
