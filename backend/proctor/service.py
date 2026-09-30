"""Per-candidate proctoring session: frames in -> status, events, verdicts, alerts out."""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import cv2
import numpy as np

from .audio import AudioRingBuffer, NullTranscriber, SpeechMonitor, VoiceActivity, to_wav
from .calibration import TARGETS, BaselineCalibration, BaselineCollector, GazeCalibration
from .clip import BufferedFrame, ClipBuilder, Evidence, FrameRingBuffer
from .detector import AnomalyDetector, DetectorConfig, gaze_direction
from .features import extract_features
from .pipeline import Smoother
from .retention import EvidenceStore

Notify = Callable[[dict], Awaitable[None]]


@dataclass
class SessionConfig:
    alert_threshold: float = 0.70      # judge confidence needed to show the fraud popup
    max_judge_calls: int = 8           # per session; further events go straight to human review
    status_interval_s: float = 0.25
    max_calibration_attempts: int = 2
    detector: DetectorConfig = field(default_factory=DetectorConfig)


class DownCue:
    """Catches looking down (lap, phone, notes), which iris tracking alone misses.

    When people look down their eyelids droop: the eye aspect ratio falls into
    "blink" range, the iris estimate is frozen (by design, for real blinks) and
    the regression keeps predicting on-screen. Two iris-free signals, both
    relative to this candidate's own calibration, override the vertical axis:
      * eye openness below EAR_RATIO x their on-screen median for >= HOLD_S
        (real blinks last ~0.1-0.3 s, so they don't count)
      * head pitch more than PITCH_MARGIN deg below the lowest pitch they used
        to look at the bottom calibration targets
    """

    EAR_RATIO = 0.65
    HOLD_S = 0.6
    PITCH_MARGIN = 8.0
    FORCED_Y = 1.6          # clearly beyond the detector's enter threshold

    def __init__(self, calibration):
        self.cal = calibration
        self.low_since: Optional[float] = None

    def active(self, t: float, row: dict) -> bool:
        cal = self.cal
        if cal is None or row.get("face_count") != 1:
            self.low_since = None
            return False
        eyes_low = False
        ear = row.get("ear_s")
        if cal.ear_base and ear is not None and ear < self.EAR_RATIO * cal.ear_base:
            if self.low_since is None:
                self.low_since = t
            eyes_low = t - self.low_since >= self.HOLD_S
        else:
            self.low_since = None
        # Head tilt is already scored by the baseline model; the extra pitch guard
        # only applies to the 5-point regression model (which can under-use pitch).
        head_down = False
        if isinstance(cal, GazeCalibration):
            pitch = row.get("pitch_s")
            head_down = cal.pitch_lo is not None and pitch is not None and pitch < cal.pitch_lo - self.PITCH_MARGIN
        return eyes_low or head_down

    def apply(self, t: float, row: dict, gaze: Optional[tuple[float, float]]):
        if gaze is None:
            return gaze
        if self.active(t, row):
            return gaze[0], max(gaze[1], self.FORCED_Y)
        return gaze


class ProctorSession:
    def __init__(self, session_id: str, landmarker, judge, store: EvidenceStore,
                 config: Optional[SessionConfig] = None, exam_ref: Optional[str] = None,
                 scene_analyzer=None, transcriber=None, knowledge=None):
        self.id = session_id
        self.lock = threading.RLock()             # video and audio arrive on different threads
        self.transcriber = transcriber or NullTranscriber()
        self.knowledge = knowledge                 # knowledge.ProctorKnowledge (GraphRAG) or None
        self.scene_analyzer = scene_analyzer       # objects.SceneAnalyzer or None
        self._scene_row: dict = {}
        self.background: list = []                 # object boxes seen during calibration = the room
        self.exam_ref = exam_ref           # host app's session id (e.g. the CAT session)
        self.landmarker = landmarker
        self.judge = judge
        self.store = store
        self.cfg = config or SessionConfig()
        self.smoother = Smoother()
        self.buffer = FrameRingBuffer(store.policy.live_buffer_s)
        self.audio = AudioRingBuffer(store.policy.live_buffer_s)
        self.vad = VoiceActivity()
        self.speech = SpeechMonitor()
        self.mic_active = False
        self._calib_mouth: list[float] = []
        self.clips = ClipBuilder(self.buffer, audio=self.audio, speech=self.speech)
        self.detector = AnomalyDetector(self.cfg.detector)
        self.collector = BaselineCollector()
        self.calibration: Optional[BaselineCalibration] = None
        self.down_cue = DownCue(None)
        self.calibration_attempts = 0
        self.mode = "calibrating"          # calibrating | monitoring | presence_only | ended
        self.events: list[dict] = []       # event log (no images)
        self.judge_calls = 0
        self.frames_seen = 0
        self._t_last = -1.0
        self._last_status = -1e9
        self.tasks: set[asyncio.Task] = set()

    # ---- control messages ------------------------------------------------------
    def calib_target(self, name: str) -> None:
        if self.mode != "calibrating":
            return
        self.collector.start_target(name, self._t_last)

    def calib_target_end(self) -> None:
        self.collector.end_target()

    def calib_finish(self) -> dict:
        self.calibration_attempts += 1
        try:
            cal = self.collector.fit()
        except ValueError as e:
            ok, info = False, {"error": str(e)}
            cal = None
        else:
            ok, info = cal.ok, cal.to_dict()
        retry = not ok and self.calibration_attempts < self.cfg.max_calibration_attempts
        noise_floor = self.vad.finish_calibration() if self.mic_active else None
        print(f"[proctor] mic: {'on, room noise floor ' + format(noise_floor, '.1f') + ' dBFS, ' + self.vad.backend if noise_floor is not None else 'NOT receiving audio'}",
              flush=True)
        if len(self._calib_mouth) >= 5:
            self.speech.mouth_baseline_std = float(np.std(self._calib_mouth))
        self._calib_mouth = []
        if ok:
            self.calibration = cal
            self.down_cue = DownCue(cal)
            self.smoother.blink_threshold = cal.blink_threshold()
            self.mode = "monitoring"
        elif retry:
            self.collector = BaselineCollector()
        else:
            # Gaze unreliable for this person/setup: keep presence checks (no face / extra face).
            self.mode = "presence_only"
        return {"type": "calibration", "ok": ok, "retry": retry, "mode": self.mode,
                "baseline": {k: round(v, 3) for k, v in info.items() if isinstance(v, float)} if cal else None,
                "samples": info.get("n_samples"), "error": info.get("error"),
                "mic": {"active": self.mic_active, "noise_floor_db": None if noise_floor is None else round(noise_floor, 1),
                        "vad": self.vad.backend}}

    # ---- frame path (CPU-bound; call from a worker thread) ------------------------
    def ingest(self, t: float, jpeg: bytes) -> tuple[Optional[dict], list, list[Evidence]]:
        """Process one frame. Returns (status message or None, new events, ready evidence)."""
        with self.lock:
            return self._ingest(t, jpeg)

    def _ingest(self, t: float, jpeg: bytes) -> tuple[Optional[dict], list, list[Evidence]]:
        if self.mode == "ended" or t <= self._t_last:
            return None, [], []
        self._t_last = t
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, [], []
        self.frames_seen += 1
        h, w = img.shape[:2]
        faces = self.landmarker(img, int(t * 1000))
        f = extract_features(faces, t, w, h)
        self.smoother.update(f)
        row = f.to_row()

        if self.mode == "calibrating":
            self.collector.add(t, row)
            if f.face_count == 1 and f.mouth is not None:
                self._calib_mouth.append(f.mouth)

        gaze = self.calibration.predict(row) if (self.calibration and f.face_count >= 1) else None
        gaze = self.down_cue.apply(t, row, gaze)
        scene = self.scene_analyzer(img, int(t * 1000), faces) if self.scene_analyzer else None
        if scene is not None:
            from .objects import relevant_objects
            if self.mode == "calibrating":
                self.background.extend(scene.objects)      # whatever sits in the room already
                scene.objects = []
            else:
                scene.objects = relevant_objects(scene.objects, scene.hands, scene.face, self.background)
            self._scene_row = {"objects": scene.prohibited, "persons": scene.persons,
                               "foreign_hands": scene.foreign_hands}
        brow = {"face_count": f.face_count, **self._scene_row}
        if f.face_count == 1 and f.mouth is not None:
            brow["mouth"] = round(f.mouth, 4)
        if gaze is not None:
            gx, gy = gaze
            brow.update(gaze_x=round(gx, 3), gaze_y=round(gy, 3),
                        gaze_score=round(max(abs(gx), abs(gy)), 3), gaze_dir=gaze_direction(gx, gy))
        self.buffer.add(BufferedFrame(t, jpeg, brow))

        events = []
        if self.mode in ("monitoring", "presence_only"):
            events = self.detector.update(t, f.face_count, gaze if self.mode == "monitoring" else None)
            if scene is not None:
                events += self.detector.update_scene(t, scene)
            for ev in events:
                self.clips.add_event(ev)
                self.events.append({**ev.to_dict(), "status": "detected"})
        ready = self.clips.poll(t)

        status = None
        if t - self._last_status >= self.cfg.status_interval_s:
            self._last_status = t
            status = {"type": "status", "t": round(t, 2), "mode": self.mode, "face_count": f.face_count,
                      "state": self.detector.state if self.mode != "calibrating" else "calibrating",
                      "gaze": [brow.get("gaze_x"), brow.get("gaze_y")] if gaze else None,
                      "objects": self._scene_row.get("objects", []),
                      "foreign_hands": self._scene_row.get("foreign_hands", 0),
                      "persons": self._scene_row.get("persons", 0),
                      "mic": self.mic_active, "speaking": self.speech.speaking}
        return status, events, ready

    # ---- audio path ------------------------------------------------------------------
    def ingest_audio(self, t_end: float, pcm) -> list:
        """One microphone chunk (16 kHz int16, `t_end` = time of its last sample on the
        same clock as the video frames). Returns new events (already queued for clips)."""
        with self.lock:
            if self.mode == "ended" or len(pcm) == 0:
                return []
            self.mic_active = True
            t_start = self.audio.add(t_end, pcm)
            frames = self.vad.process(t_start, pcm, calibrating=self.mode == "calibrating")
            if self.mode not in ("monitoring", "presence_only"):
                return []
            raw = self.speech.update(frames, lambda a, b: [f.row for f in self.buffer.window(a, b)])
            if self.speech.last_rejected:
                r, self.speech.last_rejected = self.speech.last_rejected, None
                print(f"[proctor] mic: heard {r['speech_s']}s of sound (pitch {r['pitched_s']}s, "
                      f"loudness swing {r['modulation_db']} dB, peak {r['peak_db']} dBFS) - too short or "
                      f"not speech-like, not flagged", flush=True)
            events = []
            for ev in raw:
                events += self.detector.emit_external(ev)
            for ev in events:
                self.clips.add_event(ev)
                self.events.append({**ev.to_dict(), "status": "detected"})
            return events

    # ---- judging (async) -----------------------------------------------------------
    async def process_evidence(self, ev: Evidence, notify: Notify) -> dict:
        wav = to_wav(ev.audio) if ev.audio is not None and len(ev.audio) else None
        eid = await asyncio.to_thread(self.store.save, self.id, ev.event.to_dict(),
                                      [f.jpeg for f in ev.frames], ev.rel_times, ev.timeline,
                                      audio=wav, extra={"speech_s": ev.speech_s})
        rec = self._event_record(ev.event.id)
        rec.update(status="judging", evidence_id=eid)

        # Context for the judge: what was said (flagged clip only) and which exam rules /
        # past cases apply (GraphRAG over the proctoring knowledge graph).
        if ev.has_speech and getattr(self.transcriber, "enabled", False):
            try:
                ev.transcript = await asyncio.to_thread(self.transcriber.transcribe, ev.audio)
            except Exception as e:
                ev.transcript = {"text": "", "error": str(e)[:200]}
        if self.knowledge is not None:
            try:
                ev.knowledge = await asyncio.to_thread(self.knowledge.context_for, ev, self.id)
            except Exception as e:
                print(f"[proctor] knowledge graph lookup failed: {e}", flush=True)
                ev.knowledge = {}
        if ev.transcript or ev.knowledge:
            await asyncio.to_thread(self.store.annotate, eid, transcript=ev.transcript or None,
                                    rules=[r["id"] for r in ev.knowledge.get("rules", [])] or None)

        if self.judge_calls >= self.cfg.max_judge_calls:
            verdict = {"verdict": "unavailable", "confidence": 0.0,
                       "reason": "per-session judge cap reached; queued for human review"}
        else:
            self.judge_calls += 1
            v = await asyncio.to_thread(self.judge.judge, ev)
            verdict = v.to_dict()
        exists = await asyncio.to_thread(self.store.apply_verdict, eid, verdict)
        rec.update(status="judged", verdict=verdict)
        if exists and self.knowledge is not None and verdict.get("verdict") in ("fraud", "suspicious", "benign"):
            try:                               # the incident becomes knowledge for future cases
                await asyncio.to_thread(self.knowledge.record_incident, self.id, ev, verdict, eid)
            except Exception as e:
                print(f"[proctor] knowledge graph update failed: {e}", flush=True)
        where = f" ({ev.event.direction})" if ev.event.direction else ""
        print(f"[proctor] AI judge: {ev.event.type}{where} -> {verdict.get('verdict')} "
              f"{verdict.get('confidence', 0):.2f} | {verdict.get('reason', '')} "
              f"| {verdict.get('latency_s', 0)}s {verdict.get('model', '')}", flush=True)

        if verdict["verdict"] == "fraud" and verdict.get("confidence", 0) >= self.cfg.alert_threshold:
            rec["alerted"] = True
            await asyncio.to_thread(self.store.annotate, eid, alerted=True)
            await notify({"type": "alert", "event_id": ev.event.id, "event_type": ev.event.type,
                          "title": describe(ev.event.type, ev.event.direction, ev.event.details)[0],
                          "message": alert_message(ev.event.type, ev.event.direction, ev.event.details),
                          "reason": verdict.get("reason", "")})
        await notify({"type": "event", "event": {k: rec.get(k) for k in
                                                 ("id", "type", "direction", "status", "t_start", "t_trigger")},
                      "verdict": verdict.get("verdict")})
        return rec

    def dispatch(self, evidences: list[Evidence], notify: Notify) -> None:
        for ev in evidences:
            task = asyncio.create_task(self.process_evidence(ev, notify))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    def _event_record(self, event_id: str) -> dict:
        for r in self.events:
            if r["id"] == event_id:
                return r
        r = {"id": event_id}
        self.events.append(r)
        return r

    async def end(self, notify: Optional[Notify] = None) -> dict:
        if self.mode == "ended":
            return self.summary()
        pending = self.clips.flush()
        self.mode = "ended"
        if notify:
            self.dispatch(pending, notify)
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.buffer.clear()                        # live frames are gone for good
        self.audio.clear()                         # and so is live audio
        summary = self.summary()
        await asyncio.to_thread(self.store.end_session, self.id, summary)
        return summary

    def summary(self) -> dict:
        verdicts: dict[str, int] = {}
        for e in self.events:
            v = (e.get("verdict") or {}).get("verdict", "pending")
            verdicts[v] = verdicts.get(v, 0) + 1
        return {"session_id": self.id, "mode": self.mode, "frames": self.frames_seen,
                "events": len(self.events), "verdicts": verdicts, "judge_calls": self.judge_calls,
                "alerts": sum(1 for e in self.events if e.get("alerted"))}


_WHERE = {"up": "up", "down": "down", "left": "to your left", "right": "to your right"}


def describe(event_type: str, direction: Optional[str], details: Optional[dict] = None) -> tuple[str, str]:
    """(short title, plain sentence) naming exactly what the candidate did."""
    d = details or {}
    where = _WHERE.get(direction or "", direction or "away")
    secs = d.get("duration_s") or d.get("speech_s")
    about = f" for about {round(secs)} seconds" if secs else ""
    return {
        "OFFSCREEN_SUSTAINED": (f"Looking {where.replace('to your ', '')}",
                                f"You looked {where}, away from the screen{about}."),
        "REPEATED_GLANCES": (f"Repeated glances {where.replace('to your ', '')}",
                             f"You glanced {where}, away from the screen, {d.get('glances', 'several')} times in a short period."),
        "NO_FACE": ("Face not visible", "Your face was not visible to the camera for several seconds."),
        "MULTIPLE_FACES": ("Another person", "Another person's face appeared in your camera view."),
        "EXTRA_PERSON": ("Another person", "Another person was detected near you."),
        "FOREIGN_HAND": ("Another person's hand", "Another person's hand or signal appeared in your camera view."),
        "VOICE_DETECTED": {
            "candidate": ("Talking during the exam", f"You were speaking during the exam{about}."),
            "another person": ("Another voice in the room",
                               "Another person's voice was heard while you were taking the exam."),
        }.get(direction or "", ("Voice detected", "Speech was picked up by your microphone during the exam.")),
        "PROHIBITED_OBJECT": (f"{(direction or 'Prohibited item').capitalize()} detected",
                              f"A {direction or 'prohibited item'} was visible in your camera view."),
    }.get(event_type, ("Unusual activity", "Unusual activity was detected."))


def flag_message(event_type: str, direction: Optional[str], details: Optional[dict] = None) -> dict:
    """Instant, non-accusing notice sent the moment a flag fires (before the AI judge)."""
    title, sentence = describe(event_type, direction, details)
    return {"title": title, "message": f"{sentence} Please keep your eyes on the screen."}


def alert_message(event_type: str, direction: Optional[str], details: Optional[dict] = None) -> str:
    _, sentence = describe(event_type, direction, details)
    return (f"{sentence} Our AI reviewer confirmed this as possible malpractice. It has been recorded "
            "for an examiner to review. Please keep your eyes on the screen and continue.")


class SessionManager:
    def __init__(self, landmarker_factory, judge, store: EvidenceStore,
                 config_factory: Callable[[], SessionConfig] = SessionConfig, scene_factory=None,
                 transcriber=None, knowledge=None):
        self.landmarker_factory = landmarker_factory
        self.transcriber = transcriber
        self.knowledge = knowledge
        self.scene_factory = scene_factory
        self.judge = judge
        self.store = store
        self.config_factory = config_factory
        self.sessions: dict[str, ProctorSession] = {}

    def create(self, exam_ref: Optional[str], policy_version: str) -> ProctorSession:
        sid = uuid.uuid4().hex
        self.store.record_consent(sid, exam_ref, policy_version)
        scene = None
        if self.scene_factory is not None:
            try:
                scene = self.scene_factory()
            except Exception as e:      # object/hand models unavailable: keep face tracking running
                print(f"[proctor] object/hand detection disabled: {e}")
        s = ProctorSession(sid, self.landmarker_factory(), self.judge, self.store, self.config_factory(),
                           exam_ref=exam_ref, scene_analyzer=scene, transcriber=self.transcriber,
                           knowledge=self.knowledge)
        self.sessions[sid] = s
        return s

    def get(self, sid: str) -> Optional[ProctorSession]:
        return self.sessions.get(sid)

    def find_live(self, exam_ref: str) -> Optional[ProctorSession]:
        """The active proctoring session for a host exam session, if any."""
        for s in reversed(list(self.sessions.values())):
            if s.exam_ref == exam_ref and s.mode != "ended":
                return s
        return None

    def _find_any(self, exam_ref: str) -> Optional[ProctorSession]:
        """Live OR still-finalising session (ended, judge calls in flight) for an exam."""
        for s in reversed(list(self.sessions.values())):
            if s.exam_ref == exam_ref:
                return s
        return None

    def integrity_report(self, exam_ref: str) -> dict:
        """Proctoring summary for a host exam session: in-memory session (live or still
        finalising) or the stored records. Never reports 'clear' while a verdict is pending."""
        live = self._find_any(exam_ref)
        if live is not None:
            summary, events = live.summary(), live.events
        else:
            metas = self.store.sessions_for_exam(exam_ref)
            if not metas:
                return {"proctored": False, "status": "not_proctored", "alerts": 0, "flags": 0}
            summary = {k: 0 for k in ("events", "alerts", "judge_calls")}
            summary["verdicts"] = {}
            for m in metas:
                sm = m.get("summary")
                if not sm:                     # session never finished cleanly: rebuild from its evidence
                    ev = self.store.list_evidence(m["session_id"])
                    sm = {"events": len(ev), "alerts": sum(1 for x in ev if x.get("alerted")), "verdicts": {}}
                    for x in ev:
                        v = (x.get("verdict") or {}).get("verdict", "pending")
                        sm["verdicts"][v] = sm["verdicts"].get(v, 0) + 1
                for k in ("events", "alerts", "judge_calls"):
                    summary[k] += sm.get(k, 0)
                for v, n in (sm.get("verdicts") or {}).items():
                    summary["verdicts"][v] = summary["verdicts"].get(v, 0) + n
            summary["mode"] = (metas[-1].get("summary") or {}).get("mode", "ended")
            events = []
        verdicts = summary.get("verdicts", {})
        needs_review = sum(n for v, n in verdicts.items() if v in ("fraud", "suspicious", "error", "unavailable", "pending"))
        status = "alerted" if summary.get("alerts") else ("under_review" if needs_review else "clear")
        return {"proctored": True, "status": status, "alerts": summary.get("alerts", 0),
                "flags": summary.get("events", 0), "needs_review": needs_review, "verdicts": verdicts,
                "mode": summary.get("mode"),
                "recent": [{k: e.get(k) for k in ("type", "direction", "status")} |
                           {"verdict": (e.get("verdict") or {}).get("verdict")} for e in events[-5:]]}

    async def end(self, sid: str, notify: Optional[Notify] = None) -> Optional[dict]:
        s = self.sessions.get(sid)
        if not s:
            return None
        summary = await s.end(notify)
        for obj in (s.landmarker, s.scene_analyzer):
            close = getattr(obj, "close", None)
            if close:
                close()
        del self.sessions[sid]
        return summary
