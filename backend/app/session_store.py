"""
In-memory session store.

A demo-scale stand-in for a real session store (Redis, a DB table, etc).
Fine for a single-process demo; swap this module out first if this ever
needs to run behind multiple workers or survive a restart.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from app.irt.engine import Response


@dataclass
class TestSession:
    session_id: str
    used_question_ids: set[str] = field(default_factory=set)
    responses: list[Response] = field(default_factory=list)
    response_topics: list[str] = field(default_factory=list)
    current_question_id: str | None = None
    max_questions: int = 10
    administered: list[dict] = field(default_factory=list)   # frozen copies of served questions, in order
    current_question: dict | None = None                     # frozen copy of the question on screen
    served_at: float | None = None
    finished_logged: bool = False
    username: str | None = None
    subject: str | None = None                               # subject key (see /subjects)
    sections: list[str] = field(default_factory=list)       # syllabus sections of that subject
    topic: str | None = None                                 # examiner narrowed the exam to one topic
    prefetched: dict = field(default_factory=dict, repr=False)   # live generation: outcome -> Future
    sources: list[str] = field(default_factory=list)
    start_theta: float = 0.0                                # from the candidate's last attempt
    prior_weak: list[str] = field(default_factory=list)     # concepts missed in the last attempt
    attempt_saved: bool = False        # "live" / "bank" per question served
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def questions_administered(self) -> int:
        return len(self.responses)


_sessions: dict[str, TestSession] = {}


def create_session(max_questions: int = 10) -> TestSession:
    session = TestSession(session_id=str(uuid.uuid4()), max_questions=max_questions)
    _sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> TestSession | None:
    return _sessions.get(session_id)
