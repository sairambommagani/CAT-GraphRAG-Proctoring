from pydantic import BaseModel


class QuestionOut(BaseModel):
    question_id: str
    topic: str
    text: str
    options: list[str]
    question_number: int
    max_questions: int


class StartTestResponse(BaseModel):
    session_id: str
    question: QuestionOut


class SubmitAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    selected_index: int


class SubmitAnswerResponse(BaseModel):
    correct: bool
    theta_estimate: float
    se: float
    finished: bool
    next_question: QuestionOut | None = None


class TopicBreakdown(BaseModel):
    topic: str
    correct: int
    total: int


class IntegrityReport(BaseModel):
    proctored: bool
    status: str                     # not_proctored | clear | under_review | alerted
    alerts: int = 0
    flags: int = 0
    needs_review: int = 0
    verdicts: dict[str, int] = {}
    mode: str | None = None
    recent: list[dict] = []


class SectionResult(BaseModel):
    section: str
    correct: int
    total: int
    score: float | None = None      # None = not assessed (no approved questions in this section)
    missed_concepts: list[str] = []


class ResultResponse(BaseModel):
    session_id: str
    theta_estimate: float
    se: float
    proficiency: str
    questions_administered: int
    correct_count: int
    topic_breakdown: list[TopicBreakdown]
    integrity: IntegrityReport | None = None
    assessment: str | None = None
    sections: list[SectionResult] = []
