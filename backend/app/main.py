import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
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
    )


def _next_question(session, theta: float) -> dict | None:
    candidates = [q for q in exam.bank.approved() if q["id"] not in session.used_question_ids]
    return select_next_balanced(theta, candidates, session.administered, exam.syllabus.weights())


FROZEN_KEYS = ("id", "section", "topic_name", "concepts", "text", "options", "correct_index", "a", "b")


def _serve(session, question: dict) -> None:
    # Grade and report from what the candidate actually saw, even if an examiner edits,
    # re-keys, rejects or re-tags the bank question while this exam is running.
    session.current_question = {k: (list(question[k]) if isinstance(question.get(k), list) else question.get(k))
                                for k in FROZEN_KEYS}
    session.current_question_id = question["id"]
    session.used_question_ids.add(question["id"])
    session.served_at = time.time()


@app.post("/start-test", response_model=StartTestResponse)
def start_test():
    session = create_session(max_questions=exam.syllabus.exam_length)
    # First question: theta estimate is the prior mean (0); the blueprint picks the section.
    first = _next_question(session, 0.0)
    if first is None:
        raise HTTPException(500, "Question bank is empty")
    _serve(session, first)
    return StartTestResponse(
        session_id=session.session_id,
        question=_question_out(first, question_number=1, max_questions=session.max_questions),
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
        nxt = _next_question(session, estimate.theta)
        if nxt is None:
            finished = True
        else:
            _serve(session, nxt)
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

    return SubmitAnswerResponse(
        correct=correct,
        theta_estimate=round(estimate.theta, 3),
        se=round(estimate.se, 3),
        finished=finished,
        next_question=next_out,
    )


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
        assessment=exam.syllabus.title,
        sections=[SectionResult(**s) for s in sections],
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
                           "knowledge_graph": proctoring.knowledge is not None}}


# Serve the frontend from the API origin: http://localhost:8000/ui/
# (webcam access needs a secure context; http://localhost counts as one).
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui/")
