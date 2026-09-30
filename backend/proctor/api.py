"""FastAPI integration: mount with `install_proctoring(app)`.

Candidate endpoints
    GET  /proctor/policy                 consent + retention text (single source of truth)
    POST /proctor/sessions               {consent: true, policy_version, exam_ref?} -> session
    WS   /proctor/ws/{session_id}        binary frames in, status/calibration/alerts out
    POST /proctor/sessions/{id}/end      finish; flushes pending clips, wipes live buffer

Examiner endpoints (header X-Admin-Token = $PROCTOR_ADMIN_TOKEN)
    GET    /proctor/admin/evidence?session_id=
    GET    /proctor/admin/evidence/{eid}/frames/{i}.jpg
    GET    /proctor/admin/evidence/{eid}/audio.wav
    POST   /proctor/admin/evidence/{eid}/review   {decision: confirm|dismiss, note?}
    DELETE /proctor/admin/sessions/{sid}          right to erasure
    POST   /proctor/admin/purge                   run retention sweep now
    GET    /proctor/admin/knowledge/stats         knowledge graph size, communities
    GET    /proctor/admin/knowledge/rules         exam rules the judge cites
    POST   /proctor/admin/knowledge/ask           {question} -> GraphRAG answer over rules + incidents
    POST   /proctor/admin/knowledge/rebuild       reload knowledge/policy/*.md
    GET    /proctor/admin/usage                   NIM free-tier usage today
    GET    /proctor/admin/audit

WebSocket protocol
    client -> server  binary: 8-byte little-endian float64 capture time (ms) + JPEG bytes   (video frame)
                      binary: b"AUD1" + float64 time of last sample (ms) + int16 LE PCM, 16 kHz mono (mic)
                      text:   {"type": "calib_target", "name": "top_left"} | {"type": "calib_target_end"}
                              {"type": "calib_finish"} | {"type": "ack", "event_id": ...} | {"type": "end"}
    server -> client  {"type": "ready", ...} | {"type": "status", ...} | {"type": "calibration", ...}
                      {"type": "event", ...} | {"type": "alert", ...} | {"type": "ended", ...}
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import struct
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

from .audio import AUDIO_MAGIC, unpack_audio
from .calibration import TARGETS
from .service import SessionManager

PURGE_INTERVAL_S = 3600


class CreateSession(BaseModel):
    consent: bool
    policy_version: str
    exam_ref: Optional[str] = None     # opaque CAT session id; never sent to the judge


class ReviewBody(BaseModel):
    decision: str
    note: str = ""


class AskBody(BaseModel):
    question: str


def create_router(manager: SessionManager, admin_token: Optional[str] = None,
                  stream_fps: int = 8) -> APIRouter:
    r = APIRouter(prefix="/proctor", tags=["proctoring"])
    store = manager.store
    state = {"purge_task": None}

    def ensure_purger() -> None:
        if state["purge_task"] is None:
            async def loop():
                while True:
                    await asyncio.to_thread(store.purge)
                    if manager.knowledge is not None:
                        await asyncio.to_thread(manager.knowledge.purge)
                    await asyncio.sleep(PURGE_INTERVAL_S)
            state["purge_task"] = asyncio.get_running_loop().create_task(loop())

    def admin(x_admin_token: Optional[str] = Header(default=None)) -> str:
        if not admin_token:
            raise HTTPException(503, "examiner API disabled: set PROCTOR_ADMIN_TOKEN")
        if not x_admin_token or not hmac.compare_digest(x_admin_token, admin_token):
            raise HTTPException(401, "invalid admin token")
        return "examiner"

    # ---- candidate ------------------------------------------------------------
    @r.get("/policy")
    def policy():
        return store.policy.consent_text()

    @r.post("/sessions")
    async def create(body: CreateSession):
        if not body.consent:
            raise HTTPException(400, "proctoring requires consent")
        try:
            s = manager.create(body.exam_ref, body.policy_version)
        except ValueError as e:
            raise HTTPException(409, str(e))
        ensure_purger()
        return {"session_id": s.id, "policy_version": store.policy.version,
                "calibration": {"targets": [{"name": "center", "x": 0.0, "y": 0.0}],
                                "seconds_per_target": 3.0},
                "stream": {"fps": stream_fps, "width": 640, "height": 480, "jpeg_quality": 0.75},
                "judge": {"model": manager.judge.cfg.model, "available": bool(manager.judge.available)}}

    @r.post("/sessions/{sid}/end")
    async def end(sid: str):
        summary = await manager.end(sid)
        if summary is None:
            raise HTTPException(404, "unknown session")
        return summary

    @r.websocket("/ws/{sid}")
    async def ws(websocket: WebSocket, sid: str):
        session = manager.get(sid)
        await websocket.accept()
        if session is None:
            await websocket.close(code=4404)
            return
        send_lock = asyncio.Lock()

        async def notify(msg: dict) -> None:
            try:
                async with send_lock:
                    await websocket.send_json(msg)
            except Exception:
                pass            # client gone; the event log and store still have everything

        # Latest-frame slot: if the model is slower than the stream, drop stale frames
        # instead of queueing them (keeps latency bounded).
        latest: dict = {"frame": None}
        have_frame = asyncio.Event()
        stop = asyncio.Event()

        async def worker():
            while not stop.is_set():
                await have_frame.wait()
                have_frame.clear()
                item, latest["frame"] = latest["frame"], None
                if item is None:
                    continue
                status, events, ready = await asyncio.to_thread(session.ingest, *item)
                for ev in events:            # logged only; the candidate sees a popup only if the AI judge confirms
                    print(f"[proctor] FLAG: {ev.type}{f' ({ev.direction})' if ev.direction else ''} "
                          f"at {ev.t_trigger:.1f}s -> sent to AI judge", flush=True)
                if ready:
                    session.dispatch(ready, notify)
                if status:
                    await notify(status)

        # Audio is processed in order on its own worker thread: the session lock is shared with
        # the (slow) video path, and the event loop must never wait on it.
        audio_q: asyncio.Queue = asyncio.Queue(maxsize=64)

        async def audio_worker():
            while True:
                chunk = await audio_q.get()
                if chunk is None:
                    return
                events = await asyncio.to_thread(session.ingest_audio, *chunk)
                for ev in events:
                    print(f"[proctor] FLAG: {ev.type} ({ev.direction}) at {ev.t_trigger:.1f}s "
                          f"-> sent to AI judge", flush=True)

        worker_task = asyncio.create_task(worker())
        audio_task = asyncio.create_task(audio_worker())
        await notify({"type": "ready", "session_id": sid, "mode": session.mode})
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    data = msg["bytes"]
                    if data[:4] == AUDIO_MAGIC:
                        chunk = unpack_audio(data)
                        if chunk is not None:
                            try:
                                audio_q.put_nowait(chunk)
                            except asyncio.QueueFull:     # worker far behind: drop rather than stall
                                pass
                        continue
                    if len(data) > 8:
                        (t_ms,) = struct.unpack("<d", data[:8])
                        latest["frame"] = (t_ms / 1000.0, data[8:])
                        have_frame.set()
                    continue
                try:
                    cmd = json.loads(msg.get("text") or "{}")
                except json.JSONDecodeError:
                    continue
                kind = cmd.get("type")
                if kind == "calib_target":
                    session.calib_target(cmd.get("name", ""))
                elif kind == "calib_target_end":
                    session.calib_target_end()
                elif kind == "calib_finish":
                    await asyncio.sleep(0.2)          # let the last frames land
                    await notify(session.calib_finish())
                elif kind == "ack":
                    for e in session.events:
                        if e.get("id") == cmd.get("event_id"):
                            e["acknowledged"] = True
                    store.audit("alert_ack", actor="candidate", session_id=sid, event_id=cmd.get("event_id"))
                elif kind == "end":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            stop.set()
            have_frame.set()
            while not audio_q.empty():                 # session over: unprocessed audio is dropped
                audio_q.get_nowait()
            audio_q.put_nowait(None)
            await asyncio.gather(worker_task, audio_task, return_exceptions=True)
            summary = await manager.end(sid, notify)
            if summary is not None:
                await notify({"type": "ended", "summary": summary})
            try:
                await websocket.close()
            except Exception:
                pass

    # ---- examiner -----------------------------------------------------------------
    @r.get("/admin/evidence")
    def list_evidence(session_id: Optional[str] = None, who: str = Depends(admin)):
        return store.list_evidence(session_id)

    @r.get("/admin/evidence/{eid}/frames/{i}.jpg")
    def frame(eid: str, i: int, who: str = Depends(admin)):
        meta = store.meta(eid)
        if meta is None:
            raise HTTPException(404, "unknown evidence")
        if meta["frames_deleted"]:
            raise HTTPException(410, "frames deleted under retention policy")
        frames = store.load_frames(eid, who)
        if not 0 <= i < len(frames):
            raise HTTPException(404, "no such frame")
        return Response(frames[i], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @r.get("/admin/evidence/{eid}/audio.wav")
    def audio(eid: str, who: str = Depends(admin)):
        meta = store.meta(eid)
        if meta is None:
            raise HTTPException(404, "unknown evidence")
        if meta["frames_deleted"]:
            raise HTTPException(410, "audio deleted under retention policy")
        data = store.load_audio(eid, who)
        if data is None:
            raise HTTPException(404, "no audio for this evidence")
        return Response(data, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @r.post("/admin/evidence/{eid}/review")
    def review(eid: str, body: ReviewBody, who: str = Depends(admin)):
        try:
            meta = store.review(eid, body.decision, who, body.note)
            if manager.knowledge is not None:          # examiner decisions teach future judgements
                manager.knowledge.record_review(eid, body.decision)
            return meta
        except KeyError:
            raise HTTPException(404, "unknown evidence")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @r.delete("/admin/sessions/{sid}")
    def erase(sid: str, who: str = Depends(admin)):
        forgotten = manager.knowledge.forget_session(sid) if manager.knowledge is not None else 0
        return {"erased_evidence": store.erase_session(sid, who), "erased_incidents": forgotten}

    @r.post("/admin/purge")
    def purge(who: str = Depends(admin)):
        out = store.purge()
        if manager.knowledge is not None:
            out["incidents"] = manager.knowledge.purge()
        return out

    # ---- knowledge graph (GraphRAG) ----------------------------------------------------
    def kg():
        if manager.knowledge is None:
            raise HTTPException(503, "knowledge graph disabled (PROCTOR_KNOWLEDGE_GRAPH=0)")
        return manager.knowledge

    @r.get("/admin/knowledge/stats")
    def kg_stats(who: str = Depends(admin)):
        return kg().stats()

    @r.get("/admin/knowledge/rules")
    def kg_rules(who: str = Depends(admin)):
        return [r.to_dict() for r in kg().rules]

    @r.post("/admin/knowledge/ask")
    def kg_ask(body: AskBody, who: str = Depends(admin)):
        if not body.question.strip():
            raise HTTPException(400, "empty question")
        store.audit("knowledge_query", actor=who)
        return kg().ask(body.question[:500])

    @r.post("/admin/knowledge/rebuild")
    def kg_rebuild(who: str = Depends(admin)):
        n = kg().reload_rules()
        return {"rules": n, **kg().stats()}

    @r.get("/admin/usage")
    def usage(who: str = Depends(admin)):
        return {"model": manager.judge.cfg.model, "models": getattr(manager.judge, "status", lambda: [])(),
                **manager.judge.usage.snapshot()}

    @r.get("/admin/audit")
    def audit(who: str = Depends(admin)):
        return store.read_audit()[-500:]

    return r


def install_proctoring(app: FastAPI, manager: Optional[SessionManager] = None,
                       admin_token: Optional[str] = None) -> SessionManager:
    """One-line integration for an existing FastAPI app (e.g. the CAT backend)."""
    if manager is None:
        from .judge import MockJudge, NIMJudge
        from .retention import EvidenceStore
        from .objects import SceneAnalyzer
        from .tracker import FaceTracker

        from .audio import Transcriber
        from .knowledge import ProctorKnowledge

        judge = NIMJudge()
        if os.environ.get("PROCTOR_MOCK_JUDGE") == "1":
            judge = MockJudge()
        scene_factory = None if os.environ.get("PROCTOR_OBJECT_DETECTION", "1") == "0" else SceneAnalyzer
        knowledge = None if os.environ.get("PROCTOR_KNOWLEDGE_GRAPH", "1") == "0" else ProctorKnowledge.from_env()
        transcriber = Transcriber()
        if transcriber.enabled and os.environ.get("PROCTOR_ASR_WARMUP", "1") != "0":
            # load (and on first run download) the speech-to-text model now, in the background,
            # so the first voice flag isn't delayed by the download
            import threading
            threading.Thread(target=transcriber.warm_up, daemon=True).start()
        manager = SessionManager(lambda: FaceTracker(), judge, EvidenceStore.from_env(), scene_factory=scene_factory,
                                 transcriber=transcriber, knowledge=knowledge)
    app.include_router(create_router(manager, admin_token or os.environ.get("PROCTOR_ADMIN_TOKEN")))
    return manager
