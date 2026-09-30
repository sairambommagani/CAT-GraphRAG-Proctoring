"""Blueprint-constrained CAT and per-section results.

Selection (content balancing, Kingsbury & Zara 1989): before each item, pick the exam
section whose share of administered items is furthest below its blueprint weight, then
the maximum-Fisher-information unused item *within* that section. Pure max-information
CAT tends to stay in whichever section has the most discriminating items; the assessment
must cover the whole syllabus, so coverage is a hard constraint and information decides
within it. Consecutive items from the same topic are avoided when there's a choice.
"""
from __future__ import annotations

from app.irt.engine import fisher_information


def pick_section(weights: dict[str, float], counts: dict[str, int], available: dict[str, int]) -> str | None:
    n_next = sum(counts.values()) + 1
    best, best_deficit = None, None
    for sec, w in weights.items():
        if available.get(sec, 0) == 0:
            continue
        deficit = w * n_next - counts.get(sec, 0)
        if best is None or deficit > best_deficit + 1e-9:
            best, best_deficit = sec, deficit
    return best


def select_next_balanced(theta: float, candidates: list[dict], administered: list[dict],
                         weights: dict[str, float]) -> dict | None:
    if not candidates:
        return None
    counts: dict[str, int] = {}
    for q in administered:
        counts[q.get("section")] = counts.get(q.get("section"), 0) + 1
    available: dict[str, int] = {}
    for q in candidates:
        available[q.get("section")] = available.get(q.get("section"), 0) + 1
    sec = pick_section({s: w for s, w in weights.items()}, counts, available)
    pool = [q for q in candidates if q.get("section") == sec] if sec else list(candidates)
    last_topic = administered[-1].get("topic_name") if administered else None
    varied = [q for q in pool if q.get("topic_name") != last_topic]
    pool = varied or pool
    return max(pool, key=lambda q: fisher_information(theta, q["a"], q["b"]))


def section_results(administered: list[dict], correct: list[bool], syllabus) -> list[dict]:
    rows = {}
    for q, ok in zip(administered, correct):
        r = rows.setdefault(q.get("section"), {"section": q.get("section"), "correct": 0, "total": 0,
                                                "missed_concepts": []})
        r["total"] += 1
        r["correct"] += int(ok)
        if not ok:
            r["missed_concepts"] += [c for c in q.get("concepts", []) if c not in r["missed_concepts"]]
    out = []
    for sec in syllabus.sections:
        r = rows.get(sec.name, {"section": sec.name, "correct": 0, "total": 0, "missed_concepts": []})
        r["score"] = round(r["correct"] / r["total"], 3) if r["total"] else None   # None = not assessed
        out.append(r)
    return out
