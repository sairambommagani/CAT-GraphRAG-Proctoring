"""Data-retention policy and encrypted evidence store.

Lifecycle of webcam data
------------------------
1. No consent, no capture. A session can't be created until the candidate has
   accepted the policy text below. The consent record stores the policy version
   and timestamp.
2. Live frames stay in RAM only, as a rolling ~25 s buffer that drops old frames
   continuously. They are never written to disk.
3. Unflagged footage is never stored. Only the event log (types, timestamps,
   verdicts) is kept, with no images.
4. For a flagged event, only a few downscaled frames and the audio of the
   trimmed window are stored, encrypted at rest (Fernet / AES-128-CBC + HMAC).
   A transcript of speech in that window is kept in the metadata and deleted
   together with the frames and audio.
5. After the judge rules, the frame retention depends on the verdict:
       benign                       -> frames deleted immediately
       suspicious / error / unavail -> kept 7 days for human review
       fraud                        -> kept 30 days (appeal window)
       reviewer "dismiss"           -> frames deleted immediately
       reviewer "confirm"           -> kept 30 days from the review
   Event metadata (no images) is deleted after 90 days.
6. A purge sweeper enforces the expiry times. Erasure requests delete a whole
   session at once.
7. Every store, view, verdict, review and deletion goes to an append-only audit
   log, which never contains image data.
8. Data sent to the judge is minimised: downscaled frames and tracker numbers
   only, with no name, email or candidate ID. A self-hosted NIM keeps this inside
   your infrastructure (see judge.py).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet

POLICY_VERSION = "2026-09-29.v2"
DAY = 86400.0


@dataclass
class RetentionPolicy:
    live_buffer_s: float = 25.0
    benign_ttl_s: float = 0.0
    review_ttl_s: float = 7 * DAY          # suspicious / error / unavailable
    fraud_ttl_s: float = 30 * DAY
    metadata_ttl_s: float = 90 * DAY
    version: str = POLICY_VERSION

    def ttl_for(self, verdict: str) -> float:
        if verdict == "benign":
            return self.benign_ttl_s
        if verdict == "fraud":
            return self.fraud_ttl_s
        return self.review_ttl_s

    def consent_text(self) -> dict:
        d = lambda s: int(round(s / DAY))
        return {
            "version": self.version,
            "title": "Webcam & microphone proctoring, data retention",
            "points": [
                "Your webcam and microphone are analysed during the exam to detect possible malpractice "
                "(looking away repeatedly, phones or notes, another person, talking or another voice, "
                "leaving the camera).",
                f"Live video and audio are processed in memory only, as a rolling {int(self.live_buffer_s)}-second "
                "window. They are never recorded or stored in full.",
                "If something unusual is detected, a few low-resolution frames and the audio of those seconds "
                "are encrypted and reviewed by an AI model (NVIDIA NIM). Speech in that clip may be transcribed "
                "to text. No name or ID is sent with them.",
                f"Frames judged normal are deleted immediately. Unclear cases are kept up to {d(self.review_ttl_s)} days "
                f"for human review. Confirmed malpractice evidence is kept up to {d(self.fraud_ttl_s)} days for appeals.",
                f"Event logs without images are kept for {d(self.metadata_ttl_s)} days. "
                "You can request deletion at any time.",
                "An alert never ends your exam automatically. A human reviews every flag.",
            ],
        }


class EvidenceStore:
    """Encrypted, TTL-governed storage for flagged evidence + an audit log."""

    def __init__(self, root: str | os.PathLike, key: Optional[bytes] = None,
                 policy: Optional[RetentionPolicy] = None, clock=time.time):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = policy or RetentionPolicy()
        self.clock = clock
        self.ephemeral_key = key is None
        self.fernet = Fernet(key or Fernet.generate_key())
        self.lock = threading.RLock()
        (self.root / "evidence").mkdir(exist_ok=True)
        (self.root / "sessions").mkdir(exist_ok=True)

    @classmethod
    def from_env(cls, root: str = "data/proctoring") -> "EvidenceStore":
        key = os.environ.get("PROCTOR_EVIDENCE_KEY")
        store = cls(os.environ.get("PROCTOR_DATA_DIR", root), key.encode() if key else None)
        if store.ephemeral_key:
            print("[proctor] WARNING: PROCTOR_EVIDENCE_KEY not set - using an ephemeral key; "
                  "stored evidence becomes unreadable after restart.")
        return store

    # ---- audit -----------------------------------------------------------
    def audit(self, action: str, actor: str = "system", **fields) -> None:
        rec = {"ts": self.clock(), "action": action, "actor": actor, **fields}
        with self.lock, open(self.root / "audit.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def read_audit(self) -> list[dict]:
        p = self.root / "audit.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    # ---- sessions / consent ---------------------------------------------------
    def record_consent(self, session_id: str, exam_ref: Optional[str], policy_version: str) -> None:
        if policy_version != self.policy.version:
            raise ValueError("consent given for an outdated policy version")
        meta = {"session_id": session_id, "exam_ref": exam_ref, "policy_version": policy_version,
                "consented_at": self.clock(), "ended_at": None}
        with self.lock:
            (self.root / "sessions" / f"{session_id}.json").write_text(json.dumps(meta))
        self.audit("consent", actor="candidate", session_id=session_id, policy_version=policy_version)

    def end_session(self, session_id: str, summary: dict) -> None:
        p = self.root / "sessions" / f"{session_id}.json"
        with self.lock:
            if p.exists():
                meta = json.loads(p.read_text())
                meta.update(ended_at=self.clock(), summary=summary)
                p.write_text(json.dumps(meta))
        self.audit("session_end", session_id=session_id)

    def sessions_for_exam(self, exam_ref: str) -> list[dict]:
        out = []
        for p in (self.root / "sessions").glob("*.json"):
            m = json.loads(p.read_text())
            if m.get("exam_ref") == exam_ref:
                out.append(m)
        return sorted(out, key=lambda m: m.get("consented_at", 0))

    def get_session(self, session_id: str) -> Optional[dict]:
        p = self.root / "sessions" / f"{session_id}.json"
        return json.loads(p.read_text()) if p.exists() else None

    # ---- evidence --------------------------------------------------------------
    def _dir(self, eid: str) -> Path:
        if not eid.replace("-", "").isalnum():
            raise ValueError("bad evidence id")
        return self.root / "evidence" / eid

    def save(self, session_id: str, event: dict, frames: list[bytes], rel_times: list[float],
             timeline: list[str], audio: Optional[bytes] = None, extra: Optional[dict] = None) -> str:
        eid = uuid.uuid4().hex
        d = self._dir(eid)
        now = self.clock()
        with self.lock:
            d.mkdir(parents=True)
            for i, jpg in enumerate(frames):
                (d / f"{i:03d}.bin").write_bytes(self.fernet.encrypt(jpg))
            if audio:
                (d / "audio.enc").write_bytes(self.fernet.encrypt(audio))
            meta = {"evidence_id": eid, "session_id": session_id, "event": event,
                    "rel_times": rel_times, "timeline": timeline, "n_frames": len(frames),
                    "has_audio": bool(audio), **(extra or {}),
                    "created_at": now, "verdict": None, "review": None,
                    # until judged, keep for the review window (covers judge outages)
                    "frames_expire_at": now + self.policy.review_ttl_s,
                    "meta_expire_at": now + self.policy.metadata_ttl_s, "frames_deleted": False}
            (d / "meta.json").write_text(json.dumps(meta))
        self.audit("evidence_saved", evidence_id=eid, session_id=session_id,
                   event_type=event.get("type"), n_frames=len(frames))
        return eid

    def meta(self, eid: str) -> Optional[dict]:
        p = self._dir(eid) / "meta.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _write_meta(self, eid: str, meta: dict) -> None:
        (self._dir(eid) / "meta.json").write_text(json.dumps(meta))

    def annotate(self, eid: str, **fields) -> None:
        """Add derived, non-image data (transcript, cited rules) to an evidence record."""
        with self.lock:
            meta = self.meta(eid)
            if meta is None:
                return
            if meta.get("frames_deleted"):
                fields.pop("transcript", None)    # evidence already deleted: don't bring the words back
            meta.update({k: v for k, v in fields.items() if v is not None})
            self._write_meta(eid, meta)

    def load_audio(self, eid: str, actor: str) -> Optional[bytes]:
        p = self._dir(eid) / "audio.enc"
        if not p.exists():
            return None
        with self.lock:
            data = self.fernet.decrypt(p.read_bytes())
        self.audit("audio_heard", actor=actor, evidence_id=eid)
        return data

    def load_frames(self, eid: str, actor: str) -> list[bytes]:
        d = self._dir(eid)
        with self.lock:
            files = sorted(d.glob("[0-9]*.bin"))
            frames = [self.fernet.decrypt(p.read_bytes()) for p in files]
        self.audit("evidence_viewed", actor=actor, evidence_id=eid)
        return frames

    def _delete_frames(self, eid: str, reason: str) -> None:
        d = self._dir(eid)
        with self.lock:
            for p in list(d.glob("*.bin")) + list(d.glob("*.enc")):
                p.unlink()
            meta = self.meta(eid)
            if meta:
                meta["frames_deleted"] = True
                meta.pop("transcript", None)          # what was said goes with the recording
                v = meta.get("verdict")
                if isinstance(v, dict):               # model text can describe or quote the clip
                    v.pop("reasoning", None)
                    v["observations"] = []
                    if meta.get("speech_s", 0) >= 0.5:
                        v["reason"] = "(removed with the evidence under the retention policy)"
                meta["frames_expire_at"] = None
                self._write_meta(eid, meta)
        self.audit("frames_deleted", evidence_id=eid, reason=reason)

    def apply_verdict(self, eid: str, verdict: dict) -> bool:
        """Record the judge's verdict and set the evidence expiry. False if the evidence no
        longer exists (e.g. erased while the judge was running)."""
        with self.lock:
            meta = self.meta(eid)
            if meta is None:
                return False
            meta["verdict"] = verdict
            ttl = self.policy.ttl_for(verdict.get("verdict", "error"))
            meta["frames_expire_at"] = self.clock() + ttl
            self._write_meta(eid, meta)
        self.audit("verdict", evidence_id=eid, verdict=verdict.get("verdict"),
                   confidence=verdict.get("confidence"))
        if ttl <= 0:
            self._delete_frames(eid, "benign verdict")
        return True

    def review(self, eid: str, decision: str, reviewer: str, note: str = "") -> dict:
        if decision not in ("confirm", "dismiss"):
            raise ValueError("decision must be confirm or dismiss")
        with self.lock:
            meta = self.meta(eid)
            if meta is None:
                raise KeyError(eid)
            meta["review"] = {"decision": decision, "reviewer": reviewer, "note": note[:500], "at": self.clock()}
            if decision == "confirm" and not meta["frames_deleted"]:
                meta["frames_expire_at"] = self.clock() + self.policy.fraud_ttl_s
            self._write_meta(eid, meta)
        self.audit("review", actor=reviewer, evidence_id=eid, decision=decision)
        if decision == "dismiss" and not meta["frames_deleted"]:
            self._delete_frames(eid, "dismissed by reviewer")
        return self.meta(eid)

    def list_evidence(self, session_id: Optional[str] = None) -> list[dict]:
        out = []
        for d in sorted((self.root / "evidence").iterdir()):
            m = self.meta(d.name)
            if m and (session_id is None or m["session_id"] == session_id):
                out.append(m)
        return sorted(out, key=lambda m: m["created_at"])

    # ---- enforcement -----------------------------------------------------------
    def purge(self) -> dict:
        now = self.clock()
        frames_purged = meta_purged = sessions_purged = 0
        for d in list((self.root / "evidence").iterdir()):
            m = self.meta(d.name)
            if m is None:
                continue
            if not m["frames_deleted"] and m["frames_expire_at"] is not None and now >= m["frames_expire_at"]:
                self._delete_frames(d.name, "ttl expired")
                frames_purged += 1
            if now >= m["meta_expire_at"]:
                with self.lock:
                    shutil.rmtree(d, ignore_errors=True)
                self.audit("metadata_deleted", evidence_id=d.name, reason="ttl expired")
                meta_purged += 1
        for p in list((self.root / "sessions").glob("*.json")):
            s = json.loads(p.read_text())
            if now - s.get("consented_at", now) >= self.policy.metadata_ttl_s:
                p.unlink()
                sessions_purged += 1
        if frames_purged or meta_purged or sessions_purged:
            self.audit("purge", frames=frames_purged, metadata=meta_purged, sessions=sessions_purged)
        return {"frames": frames_purged, "metadata": meta_purged, "sessions": sessions_purged}

    def erase_session(self, session_id: str, actor: str) -> int:
        """Right-to-erasure: remove every artefact of a session (audit entry kept, no content)."""
        n = 0
        for m in self.list_evidence(session_id):
            with self.lock:
                shutil.rmtree(self._dir(m["evidence_id"]), ignore_errors=True)
            n += 1
        p = self.root / "sessions" / f"{session_id}.json"
        if p.exists():
            p.unlink()
        self.audit("session_erased", actor=actor, session_id=session_id, evidence_items=n)
        return n
