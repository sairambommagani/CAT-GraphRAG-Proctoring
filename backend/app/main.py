import os
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app import accounts, attempts
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.irt.engine import (
    Response,
    estimate_ability,
    select_next_question,
    proficiency_label,
)
from exam_graph.cat_blueprint import section_results, select_next_balanced
from exam_graph.service import ExamService, create_exam_router
from app.schemas import (
    StartTestRequest,
    StartTestResponse,
    QuestionOut,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    ResultResponse,
    TopicBreakdown,
    IntegrityReport,
    SectionResult,
)
from app.session_store import create_session, get_session
from proctor.api import install_proctoring

app = FastAPI(title="CAT-GraphRAG-Proctoring API", version="1.0.0",
              description="Adaptive testing (CAT) on a GraphRAG exam knowledge "
                          "graph, with webcam + microphone AI proctoring judged by NVIDIA NIM models.")

# Demo-only: wide-open CORS so the static frontend can call this from any
# origin/port. Tighten this to specific origins before this leaves a laptop.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- exam content: syllabus -> GraphRAG knowledge graph -> question bank (backend/exam_graph) ----
exam = ExamService.from_env()

# ---- webcam proctoring (see backend/proctor and README "Proctoring") ----------
# Routes live under /proctor. Answers are only accepted while a proctoring
# session for this CAT session is live and calibrated; the check happens here on
# the server, so skipping it in the browser doesn't help.
proctoring = install_proctoring(app)
app.include_router(create_exam_router(exam, os.environ.get("PROCTOR_ADMIN_TOKEN")))


def proctoring_required() -> bool:
    return os.environ.get("CAT_REQUIRE_PROCTORING", "1") == "1"


def _check_proctoring(session_id: str) -> None:
    if not proctoring_required():
        return
    live = proctoring.find_live(session_id)
    if live is None:
        raise HTTPException(403, "Proctoring is not active for this session. Enable your webcam to continue.")
    if live.mode == "calibrating":
        raise HTTPException(403, "Please complete the webcam calibration first.")


def _question_out(question: dict, question_number: int, max_questions: int) -> QuestionOut:
    return QuestionOut(
        question_id=question["id"],
        topic=question.get("section") or question.get("topic", ""),
        text=question["text"],
        options=question["options"],
        question_number=question_number,
        max_questions=max_questions,
        source="live" if question.get("origin") == "live-generated" else "bank",
        difficulty=question.get("difficulty"),
    )


def _weights(session) -> dict:
    w = {k: v for k, v in exam.syllabus.weights().items() if not session.sections or k in session.sections}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def _bank_question(session, theta: float) -> dict | None:
    candidates = [q for q in exam.bank.approved() if q["id"] not in session.used_question_ids
                  and (not session.sections or q.get("section") in session.sections)
                  and (not session.topic or q.get("topic_name") == session.topic)]
    return select_next_balanced(theta, candidates, session.administered, _weights(session))


def _next_question(session, theta: float, correct: bool | None = None) -> dict | None:
    """Live generation for this candidate (CAT ability + GraphRAG syllabus), bank as fallback."""
    live = exam.live
    if live is not None:
        q = None
        if correct is None:                                   # first question: generate now
            plan = live.plan(session, theta, session.sections or [s.name for s in exam.syllabus.sections],
                             _weights(session))
            if plan:
                fut = live.pool.submit(live.generate, plan, [])
                try:
                    q = fut.result(timeout=live.first_wait_s)       # don't keep the candidate waiting
                except Exception:
                    q = None
        else:
            q = live.take(session, correct)
        if q is not None:
            return q
        live.fallback()
    return _bank_question(session, theta)


def _after_serve(session) -> None:
    if exam.live is not None:
        exam.live.prefetch(session, session.sections or [s.name for s in exam.syllabus.sections], _weights(session))


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def login_required() -> bool:
    return os.environ.get("CAT_REQUIRE_LOGIN", "0") == "1"      # accounts are optional (off by default)


class _Creds(BaseModel):
    username: str
    password: str
    name: str = ""


@app.post("/auth/register")
def auth_register(body: _Creds):
    try:
        return accounts.register(body.username, body.password, body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/auth/login")
def auth_login(body: _Creds):
    try:
        return accounts.login(body.username, body.password)
    except PermissionError as e:
        raise HTTPException(401, str(e))


@app.get("/exam-scope")
def exam_scope():
    return exam.get_scope()


@app.get("/subjects")
def list_subjects():
    return [{"key": k, "name": v["name"], "sections": v["sections"]} for k, v in exam.subjects().items()]


FROZEN_KEYS = ("id", "section", "topic_name", "concepts", "text", "options", "correct_index", "a", "b",
               "origin", "difficulty")


def _serve(session, question: dict) -> None:
    # Grade and report from what the candidate actually saw, even if an examiner edits,
    # re-keys, rejects or re-tags the bank question while this exam is running.
    session.current_question = {k: (list(question[k]) if isinstance(question.get(k), list) else question.get(k))
                                for k in FROZEN_KEYS}
    session.current_question_id = question["id"]
    session.used_question_ids.add(question["id"])
    session.sources.append("live" if question.get("origin") == "live-generated" else "bank")
    session.served_at = time.time()


@app.post("/start-test", response_model=StartTestResponse)
def start_test(body: StartTestRequest | None = None, authorization: str | None = Header(None)):
    user = accounts.user_for(_bearer(authorization))
    if user is None and login_required():
        raise HTTPException(401, "Please sign in first")
    subjects = exam.subjects()
    scope = exam.get_scope()                                  # set by the examiner on the question-bank page
    if body and body.subject:
        if body.subject not in subjects:
            raise HTTPException(400, f"unknown subject {body.subject!r}")
        key, secs, topic, label = body.subject, subjects[body.subject]["sections"], None, subjects[body.subject]["name"]
    else:
        key, secs, topic, label = scope["subject"], scope["sections"], scope["topic"], scope["name"]
    length = min(exam.syllabus.exam_length, 5 * len(secs)) if not topic else min(exam.syllabus.exam_length, 5)
    session = create_session(max_questions=length)
    session.username = user["username"] if user else None
    session.subject, session.sections, session.topic = key, list(secs), topic
    prev = attempts.last(session.username, key)
    if prev:                                                  # carry performance across attempts
        session.start_theta = float(prev.get("theta", 0.0))
        session.prior_weak = list(prev.get("missed_concepts", []))[:10]
    # First question: aimed at the last ability estimate (prior mean 0 for a first attempt).
    first = _next_question(session, session.start_theta)
    if first is None:
        raise HTTPException(500, "Question bank is empty")
    _serve(session, first)
    _after_serve(session)
    return StartTestResponse(
        session_id=session.session_id,
        question=_question_out(first, question_number=1, max_questions=session.max_questions),
        subject=label, candidate=user["name"] if user else None,
    )


@app.post("/submit-answer", response_model=SubmitAnswerResponse)
def submit_answer(req: SubmitAnswerRequest):
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    _check_proctoring(req.session_id)
    if not session.lock.acquire(blocking=False):             # a second click / retry racing the first
        raise HTTPException(409, "An answer for this question is already being processed")
    try:
        return _submit(session, req)
    finally:
        session.lock.release()


def _submit(session, req: SubmitAnswerRequest) -> SubmitAnswerResponse:
    if req.question_id != session.current_question_id or session.current_question is None:
        raise HTTPException(400, "This question is not the current question for this session")
    question = session.current_question
    session.current_question_id = None                       # consumed: a duplicate submit gets 400

    correct = req.selected_index == question["correct_index"]
    exam.log.record(session.session_id, question["id"], correct,
                    time.time() - session.served_at if session.served_at else None)

    session.responses.append(
        Response(question_id=question["id"], a=question["a"], b=question["b"], correct=correct)
    )
    session.response_topics.append(question.get("section") or "")
    session.administered.append(question)

    estimate = estimate_ability(session.responses)

    finished = session.questions_administered >= session.max_questions
    next_out = None

    if not finished:
        sel_theta = estimate_ability(session.responses, session.start_theta).theta   # selection: carries history
        nxt = _next_question(session, sel_theta, correct)
        if nxt is None:
            finished = True
        else:
            _serve(session, nxt)
            _after_serve(session)
            next_out = _question_out(
                nxt,
                question_number=session.questions_administered + 1,
                max_questions=session.max_questions,
            )
    if finished:
        session.current_question_id = None
        session.current_question = None
        if not session.finished_logged:
            exam.log.finish(session.session_id, estimate.theta)
            session.finished_logged = True
        _save_attempt(session, estimate)

    return SubmitAnswerResponse(
        correct=correct,
        theta_estimate=round(estimate.theta, 3),
        se=round(estimate.se, 3),
        finished=finished,
        next_question=next_out,
    )


def _recommendations(session) -> list[dict]:
    """Study plan from the syllabus graph: each topic answered wrongly, its missed concepts,
    and the prerequisite topics to revise first."""
    out: dict[str, dict] = {}
    for q, r in zip(session.administered, session.responses):
        if r.correct:
            continue
        t = exam.syllabus.topic(q.get("topic_name") or "")
        if t is None:
            continue
        rec = out.setdefault(t.name, {"topic": t.name, "section": t.section, "concepts": [],
                                      "prerequisites": list(t.requires)})
        for c in q.get("concepts") or []:
            if c not in rec["concepts"]:
                rec["concepts"].append(c)
    return list(out.values())[:6]


def _save_attempt(session, estimate) -> None:
    if not session.username or session.attempt_saved or not session.responses:
        return
    session.attempt_saved = True
    missed = [c for q, r in zip(session.administered, session.responses) if not r.correct for c in q.get("concepts") or []]
    attempts.record(session.username, {
        "session_id": session.session_id, "subject": session.subject, "theta": round(estimate.theta, 3),
        "se": round(estimate.se, 3), "correct": sum(r.correct for r in session.responses),
        "total": len(session.responses), "missed_concepts": list(dict.fromkeys(missed)),
        "live_questions": session.sources.count("live")})


@app.get("/result/{session_id}", response_model=ResultResponse)
def get_result(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if not session.responses:
        raise HTTPException(400, "No responses recorded yet for this session")

    estimate = estimate_ability(session.responses)

    topic_stats: dict[str, dict[str, int]] = {}
    for topic, r in zip(session.response_topics, session.responses):
        stats = topic_stats.setdefault(topic, {"correct": 0, "total": 0})
        stats["total"] += 1
        if r.correct:
            stats["correct"] += 1
    breakdown = [
        TopicBreakdown(topic=t, correct=s["correct"], total=s["total"])
        for t, s in sorted(topic_stats.items())
    ]
    sections = section_results(session.administered, [r.correct for r in session.responses], exam.syllabus)
    integrity = proctoring.integrity_report(session_id)

    return ResultResponse(
        session_id=session.session_id,
        theta_estimate=round(estimate.theta, 3),
        se=round(estimate.se, 3),
        proficiency=proficiency_label(estimate.theta),
        questions_administered=session.questions_administered,
        correct_count=sum(1 for r in session.responses if r.correct),
        topic_breakdown=breakdown,
        integrity=IntegrityReport(**integrity),
        assessment=(exam.subjects().get(session.subject or "full", {}).get("name") or exam.syllabus.title)
                   + (f" · {session.topic}" if session.topic else ""),
        sections=[SectionResult(**s) for s in sections if not session.sections or s["section"] in session.sections],
        subject=session.subject,
        recommendations=_recommendations(session),
        history=[{"theta": h.get("theta"), "correct": h.get("correct"), "total": h.get("total"),
                  "finished_at": h.get("finished_at")} for h in attempts.history(session.username, session.subject)][-6:],
    )


@app.get("/health")
def health():
    return {"status": "ok", "product": "CAT-GraphRAG-Proctoring", "assessment": exam.syllabus.title,
            "questions_loaded": len(exam.bank.approved()),
            "exam_graph": {"topics": len(exam.syllabus.topics), "question_generator": exam.generator is not None,
                           "llm_models": getattr(exam.llm, "models", None)},
            "proctoring": {"required": proctoring_required(), "judge_model": proctoring.judge.cfg.model,
                           "judge_chain": list(getattr(proctoring.judge.cfg, "models", ()) or [proctoring.judge.cfg.model]),
                           "judge_available": bool(proctoring.judge.available),
                           "microphone": True,
                           "speech_to_text": bool(getattr(proctoring.transcriber, "enabled", False)),
                           "knowledge_graph": proctoring.knowledge is not None},
            "semantic_cache": _cache_summary(),
            "live_questions": exam.live.summary() if exam.live is not None else None,
            "login_required": login_required()}


def _cache_summary():
    from semantic_cache import shared_cache
    return shared_cache().summary()


# Serve the frontend from the API origin: http://localhost:8000/ui/
# (webcam access needs a secure context; http://localhost counts as one).
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui/")
