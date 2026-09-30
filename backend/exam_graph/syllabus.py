"""Assessment syllabus: the authored source the exam knowledge graph is built from.

Format (markdown, examiner-editable; see knowledge/syllabus/*.md):

    # Assessment title
    exam_length: 15            (optional: number of adaptive questions)

    ## Section name
    weight: 0.2                (share of the exam blueprint)

    ### Topic name
    requires: Other Topic, Another Topic
    Text of the topic with the key **concepts** in bold.

Raw documents (PDF / DOCX / TXT) that don't follow this format can be
converted into it by the NIM model (see graph.structure_document), and the
examiner reviews the result before it is indexed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SYLLABUS_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "syllabus"


@dataclass
class Topic:
    name: str
    section: str
    text: str = ""
    requires: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return slug(f"{self.section}-{self.name}")


@dataclass
class Section:
    name: str
    weight: float = 0.0
    topics: list[Topic] = field(default_factory=list)


@dataclass
class Syllabus:
    title: str
    sections: list[Section]
    exam_length: int = 15
    source: str = ""

    @property
    def topics(self) -> list[Topic]:
        return [t for s in self.sections for t in s.topics]

    def section(self, name: str) -> Optional[Section]:
        return next((s for s in self.sections if s.name.lower() == name.lower()), None)

    def topic(self, name: str) -> Optional[Topic]:
        return next((t for t in self.topics if t.name.lower() == name.lower()), None)

    def weights(self) -> dict[str, float]:
        """Normalised blueprint weights (equal shares when none are given)."""
        raw = {s.name: s.weight for s in self.sections}
        total = sum(raw.values())
        if total <= 0:
            return {k: 1.0 / len(raw) for k in raw} if raw else {}
        return {k: v / total for k, v in raw.items()}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _concepts(text: str) -> list[str]:
    out, seen = [], set()
    for m in re.finditer(r"\*\*(.+?)\*\*", text):
        c = m.group(1).replace("`", "").strip()
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def parse_syllabus(md: str, source: str = "") -> Syllabus:
    title, sections = "Assessment", []
    meta: dict[str, str] = {}
    sec: Optional[Section] = None
    top: Optional[Topic] = None
    body: list[str] = []

    def flush_topic():
        nonlocal top, body
        if top is not None:
            top.text = " ".join(" ".join(body).split())
            top.concepts = _concepts(top.text)
            top.text = top.text.replace("**", "")
        top, body = None, []

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("### "):
            flush_topic()
            if sec is None:
                sec = Section("General")
                sections.append(sec)
            top = Topic(line[4:].strip(), sec.name)
            sec.topics.append(top)
            continue
        if line.startswith("## "):
            flush_topic()
            sec = Section(line[3:].strip())
            sections.append(sec)
            continue
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        kv = re.match(r"^\s*(exam_length|weight|requires):\s*(.*)$", line)
        if kv:
            key, val = kv.group(1), kv.group(2).strip()
            if key == "requires" and top is not None:
                top.requires = [x.strip() for x in val.split(",") if x.strip()]
            elif key == "weight" and sec is not None and top is None:
                sec.weight = float(val)
            elif key == "exam_length":
                meta[key] = val
            continue
        if top is not None:
            body.append(line)
    flush_topic()
    sections = [s for s in sections if s.topics]
    return Syllabus(title, sections, exam_length=int(meta.get("exam_length", 15)), source=source)


def load_syllabus(path: Optional[Path] = None) -> Syllabus:
    """The first syllabus file in knowledge/syllabus (or the given path)."""
    if path is None:
        files = sorted(SYLLABUS_DIR.glob("*.md"))
        if not files:
            raise FileNotFoundError(f"no syllabus in {SYLLABUS_DIR}")
        path = files[0]
    return parse_syllabus(Path(path).read_text(encoding="utf-8"), Path(path).name)


def document_text(path: Path) -> str:
    """Plain text of an uploaded document (.md/.txt/.pdf/.docx)."""
    suf = path.suffix.lower()
    if suf in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suf == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    if suf == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    raise ValueError(f"unsupported document type {suf}")
