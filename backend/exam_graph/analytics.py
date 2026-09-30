"""Performance analytics: response log, item statistics, online IRT calibration.

Every answered item is logged (session, question, correct, response time, candidate's
final ability once the exam ends). From that:

* item statistics - p-value (share correct), point-biserial discrimination (correlation of
  item correctness with final ability), mean response time;
* item health flags - too easy / too hard, low or negative discrimination (a bad key or an
  ambiguous question), and a LEAK signal: an item answered correctly much more often than
  its IRT difficulty predicts for the candidates who saw it (observed - expected > 0.25
  over >= 20 answers), which is how a leaked question shows up;
* online calibration - re-estimates difficulty b (discrimination kept) by maximum a posteriori
  from the final abilities once an item has >= min_n answers, so the generated questions'
  difficulty priors are replaced by data. Items where everyone answered the same, or that
  are flagged as leaked, are not recalibrated (that would hide the problem).

The log holds no personal data: session ids are the CAT's pseudonymous ids.
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from app.irt.engine import p_correct


class ResponseLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, session_id: str, question_id: str, correct: bool, seconds: Optional[float] = None) -> None:
        rec = {"ts": time.time(), "session": session_id, "question": question_id, "correct": bool(correct),
               "seconds": None if seconds is None else round(seconds, 2)}
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def finish(self, session_id: str, theta: float) -> None:
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "session": session_id, "final_theta": round(theta, 4)}) + "\n")

    def rows(self) -> tuple[list[dict], dict[str, float]]:
        if not self.path.exists():
            return [], {}
        answers, finals = [], {}
        with self.lock:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:          # line cut off by a crash mid-write
                continue
            if not isinstance(r, dict) or "session" not in r:
                continue
            if "final_theta" in r:
                finals[r["session"]] = r["final_theta"]
            else:
                answers.append(r)
        return answers, finals


def _pbis(xs: list[float], ys: list[int]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(my * (1 - my))
    if sx < 1e-9 or sy < 1e-9:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n * sx * sy)


def calibrate_b(thetas: list[float], ys: list[int], a: float, b0: float, iters: int = 25,
                prior_sd: float = 1.0) -> float:
    """MAP estimate of difficulty b for fixed discrimination a: the likelihood plus a normal
    prior centred on the current value (keeps all-correct / all-wrong items finite)."""
    b = b0
    for _ in range(iters):
        g = -(b - b0) / prior_sd ** 2
        h = -1.0 / prior_sd ** 2
        for th, y in zip(thetas, ys):
            p = p_correct(th, a, b)
            g += -a * (y - p)                # d logL / db
            h += -a * a * p * (1 - p)        # d2 logL / db2
        if abs(h) < 1e-9:
            break
        step = g / h
        b = max(-4.0, min(4.0, b - step))
        if abs(step) < 1e-4:
            break
    return b


def item_stats(log: ResponseLog, bank, min_n: int = 30) -> dict:
    answers, finals = log.rows()
    by_q = defaultdict(list)
    for r in answers:
        by_q[r["question"]].append(r)
    items, calibrated = [], {}
    for qid, rs in by_q.items():
        q = bank.get(qid) or {}
        n = len(rs)
        correct = sum(r["correct"] for r in rs)
        with_theta = [(finals[r["session"]], int(r["correct"])) for r in rs if r["session"] in finals]
        thetas, ys = [t for t, _ in with_theta], [y for _, y in with_theta]
        pb = _pbis(thetas, ys)
        times = [r["seconds"] for r in rs if r.get("seconds")]
        expected = (sum(p_correct(t, q.get("a", 1.0), q.get("b", 0.0)) for t in thetas) / len(thetas)) if thetas else None
        observed = sum(ys) / len(ys) if ys else None
        flags = []
        if n >= 10 and correct / n > 0.95:
            flags.append("too easy")
        if n >= 10 and correct / n < 0.05:
            flags.append("too hard")
        if pb is not None and len(thetas) >= 10 and pb < 0.1:
            flags.append("low discrimination - check the key / ambiguity")
        if expected is not None and len(thetas) >= 20 and observed - expected > 0.25:
            flags.append("possible leak - far more correct answers than its difficulty predicts")
        row = {"id": qid, "section": q.get("section"), "topic": q.get("topic_name"), "n": n,
               "p_value": round(correct / n, 3), "point_biserial": None if pb is None else round(pb, 3),
               "mean_seconds": round(sum(times) / len(times), 1) if times else None,
               "expected_p": None if expected is None else round(expected, 3), "b": q.get("b"), "flags": flags}
        uniform = bool(ys) and (sum(ys) == 0 or sum(ys) == len(ys))
        if len(thetas) >= min_n and not uniform and not any("leak" in f for f in flags):
            nb = round(calibrate_b(thetas, ys, q.get("a", 1.0), q.get("b", 0.0)), 3)
            row["b_calibrated"] = nb
            calibrated[qid] = {"b": nb, "calibrated_n": len(thetas)}
        items.append(row)
    items.sort(key=lambda r: (-len(r["flags"]), -r["n"]))
    perf = {r["id"]: {"n": r["n"], "correct": round(r["p_value"] * r["n"])} for r in items}
    return {"items": items, "calibrated": calibrated, "performance": perf,
            "responses": len(answers), "sessions": len(finals)}
