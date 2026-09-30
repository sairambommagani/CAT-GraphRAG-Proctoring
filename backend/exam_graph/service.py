"""Exam content service: syllabus -> GraphRAG knowledge graph -> question bank -> CAT, plus the
examiner API (/exam/admin/*)."""
from __future__ import annotations

import hmac
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from .analytics import ResponseLog, item_stats
from .bank import QuestionBank
from .generator import QuestionGenerator, structure_document
from .graph import ExamGraph
from .nim import LLMUnavailable, NIMText
from .syllabus import SYLLABUS_DIR, document_text, load_syllabus, parse_syllabus


class ExamService:
    def __init__(self, data_dir: Path, syllabus_path: Optional[Path] = None, llm=None, seed_path=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.syllabus_path = Path(syllabus_path) if syllabus_path else sorted(SYLLABUS_DIR.glob("*.md"))[0]
        self.llm = llm
        self.lock = threading.RLock()
        self.syllabus = load_syllabus(self.syllabus_path)
        self.graph = ExamGraph(self.syllabus, llm=llm)
        kw = {"seed_path": seed_path} if seed_path else {}
        self.bank = QuestionBank(self.data_dir / "question_bank.json", self.graph, **kw)
        self.log = ResponseLog(self.data_dir / "responses.jsonl")
        self.generator = QuestionGenerator(llm, self.graph, self.bank) if llm is not None else None
        self.refresh_analytics()

    @classmethod
    def from_env(cls) -> "ExamService":
        root = Path(os.environ.get("EXAM_DATA_DIR", "data/exam"))
        llm = NIMText(usage_path=str(root / "nim_usage.json"))
        return cls(root, llm=llm if llm.available else None)

    # ---- syllabus ---------------------------------------------------------------------------------
    def replace_syllabus(self, markdown: str, name: str = "syllabus.md") -> dict:
        new = parse_syllabus(markdown, name)
        if not new.sections or not new.topics:
            raise ValueError("no '## Section' / '### Topic' structure found")
        with self.lock:
            self.syllabus_path.write_text(markdown, encoding="utf-8")
            self.syllabus = new
            self.graph.set_syllabus(new)
            self.bank.retag()
        return self.graph.stats()

    def refresh_analytics(self) -> dict:
        st = item_stats(self.log, self.bank)
        self.graph.performance = st["performance"]
        return st


class ReviewBody(BaseModel):
    decision: str
    edits: Optional[dict] = None


class GenerateBody(BaseModel):
    topic: Optional[str] = None            # None -> the topics that need questions most
    per_topic: int = 3
    topics: int = 1


class AskBody(BaseModel):
    question: str


class SyllabusBody(BaseModel):
    markdown: str


def create_exam_router(svc: ExamService, admin_token: Optional[str]) -> APIRouter:
    r = APIRouter(prefix="/exam/admin", tags=["exam content (GraphRAG)"])

    def admin(x_admin_token: Optional[str] = Header(default=None)) -> str:
        if not admin_token:
            raise HTTPException(503, "examiner API disabled: set PROCTOR_ADMIN_TOKEN")
        if not x_admin_token or not hmac.compare_digest(x_admin_token, admin_token):
            raise HTTPException(401, "invalid admin token")
        return "examiner"

    def need_llm():
        if svc.generator is None:
            raise HTTPException(503, "question generation needs NVIDIA_API_KEY (NVIDIA NIM)")

    @r.get("/graph")
    def graph_stats(who: str = Depends(admin)):
        have = {q.get("section") for q in svc.bank.approved()}
        empty = [s.name for s in svc.syllabus.sections if s.name not in have]
        return {**svc.graph.stats(), "blueprint": svc.graph.blueprint(), "llm": getattr(svc.llm, "models", None),
                "warnings": [f"section '{n}' has no approved questions - candidates can't be assessed on it"
                             for n in empty]}

    @r.get("/syllabus")
    def syllabus(who: str = Depends(admin)):
        return {"markdown": svc.syllabus_path.read_text(encoding="utf-8"), "file": svc.syllabus_path.name}

    @r.put("/syllabus")
    def put_syllabus(body: SyllabusBody, who: str = Depends(admin)):
        try:
            return svc.replace_syllabus(body.markdown, svc.syllabus_path.name)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @r.post("/syllabus/convert")
    async def convert(file: UploadFile = File(...), who: str = Depends(admin)):
        """PDF / DOCX / TXT course material -> syllabus markdown draft (NIM). Not applied until PUT /syllabus."""
        need_llm()
        suffix = Path(file.filename or "doc.txt").suffix.lower()
        if suffix not in (".pdf", ".docx", ".txt", ".md"):
            raise HTTPException(400, "upload a .pdf, .docx, .txt or .md file")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f"upload{suffix}"
            data = await file.read(20 * 1024 * 1024 + 1)
            if len(data) > 20 * 1024 * 1024:
                raise HTTPException(413, "document larger than 20 MB")
            p.write_bytes(data)
            text = document_text(p)
        try:
            md = structure_document(svc.llm, text, Path(file.filename or "").stem)
        except LLMUnavailable as e:
            raise HTTPException(503, str(e))
        parsed = parse_syllabus(md)
        return {"markdown": md, "sections": len(parsed.sections), "topics": len(parsed.topics)}

    @r.post("/graph/relations")
    def relations(who: str = Depends(admin)):
        need_llm()
        return {"concept_relations": svc.graph.extract_concept_relations(), **svc.graph.stats()}

    @r.get("/coverage")
    def coverage(who: str = Depends(admin)):
        return svc.graph.coverage()

    @r.post("/generate")
    def generate(body: GenerateBody, who: str = Depends(admin)):
        need_llm()
        topics = [body.topic] if body.topic else [c["topic"] for c in svc.graph.coverage()[:max(1, body.topics)]]
        reports = []
        for t in topics:
            if svc.syllabus.topic(t) is None:
                raise HTTPException(404, f"unknown topic {t!r}")
            try:
                rep = svc.generator.generate(t, n=max(1, min(body.per_topic, 6)))
            except Exception as e:                    # model/transport/parse problems -> report, not 500
                rep = {"topic": t, "error": str(e)[:300], "accepted": [], "rejected": []}
            reports.append(rep)
        return {"reports": reports}

    @r.get("/questions")
    def questions(status: Optional[str] = None, who: str = Depends(admin)):
        return svc.bank.list(status)

    @r.post("/questions/{qid}/review")
    def review(qid: str, body: ReviewBody, who: str = Depends(admin)):
        try:
            return svc.bank.review(qid, body.decision, who, body.edits)
        except KeyError:
            raise HTTPException(404, "unknown question")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @r.get("/analytics")
    def analytics(who: str = Depends(admin)):
        st = svc.refresh_analytics()
        return {k: st[k] for k in ("items", "responses", "sessions")} | {"calibration_ready": len(st["calibrated"])}

    @r.post("/analytics/calibrate")
    def calibrate(who: str = Depends(admin)):
        st = svc.refresh_analytics()
        svc.bank.update_params(st["calibrated"])
        return {"updated": len(st["calibrated"]), "params": st["calibrated"]}

    @r.post("/ask")
    def ask(body: AskBody, who: str = Depends(admin)):
        if not body.question.strip():
            raise HTTPException(400, "empty question")
        svc.refresh_analytics()
        return svc.graph.ask(body.question[:500])

    return r
