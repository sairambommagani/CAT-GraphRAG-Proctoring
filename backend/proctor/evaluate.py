"""Measure accuracy on a labelled set of recorded clips: precision, recall, false-alarm rate.

    python -m proctor.evaluate eval_clips/                    # offline judge (detector accuracy)
    python -m proctor.evaluate eval_clips/ --judge nim        # full system incl. NVIDIA judge chain

eval_clips/labels.csv (one row per recording):

    file,label,expect,notes
    clean_01.mp4,clean,,normal answering, quiet room
    phone_01.mp4,cheat,PROHIBITED_OBJECT,phone held up at 10 s
    whisper_01.mp4,cheat,VOICE_DETECTED,friend whispers answers off-camera
    tv_01.mp4,clean,,TV on in the next room (should NOT alert)

* label  - ground truth for the whole clip: cheat | clean
* expect - optional flag type(s) that should fire, separated by |

Each recording should start with ~5 s of looking at the screen in silence
(calibration), like a real exam. The report is computed at two levels:

* Detector  - did any flag fire? (what the cheap local models catch)
* System    - did the candidate get a popup? (after the AI judge)

The gap between the two shows how many false flags the AI judge filtered out
(its value) and how many real cases it wrongly cleared (its cost). Results go to
<folder>/eval_report.md and eval_report.json, so runs can be compared over time
(e.g. after changing a threshold, a model, or a rule).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, predicted: bool, actual: bool) -> None:
        if predicted and actual:
            self.tp += 1
        elif predicted:
            self.fp += 1
        elif actual:
            self.fn += 1
        else:
            self.tn += 1

    def metrics(self) -> dict:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else None
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else None
        f1 = 2 * p * r / (p + r) if p and r else None
        far = self.fp / (self.fp + self.tn) if self.fp + self.tn else None
        n = self.tp + self.fp + self.fn + self.tn
        acc = (self.tp + self.tn) / n if n else None
        rnd = lambda x: None if x is None else round(x, 3)
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn, "precision": rnd(p), "recall": rnd(r),
                "f1": rnd(f1), "false_alarm_rate": rnd(far), "accuracy": rnd(acc)}


def read_labels(folder: Path) -> list[dict]:
    path = folder / "labels.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - see `python -m proctor.evaluate --help` for the format")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = (r.get("label") or "").strip().lower()
            if label not in ("cheat", "clean"):
                raise SystemExit(f"labels.csv: label must be cheat or clean (got {label!r} for {r.get('file')})")
            rows.append({"file": r["file"].strip(), "cheat": label == "cheat",
                         "expect": [x.strip() for x in (r.get("expect") or "").split("|") if x.strip()],
                         "notes": (r.get("notes") or "").strip()})
    return rows


def evaluate(folder: Path, judge_kind: str = "mock", replay=None, use_audio: bool = True) -> dict:
    from .judge import MockJudge, NIMJudge
    from .knowledge import ProctorKnowledge
    if replay is None:
        from .replay import replay_file as replay
    rows = read_labels(folder)
    detector, system = Counts(), Counts()
    per_type: dict[str, dict] = {}
    clips, latencies = [], []
    judge = NIMJudge() if judge_kind == "nim" else MockJudge()
    filtered = wrongly_cleared = 0
    t_start = time.time()
    for r in rows:
        kg = ProctorKnowledge(tempfile.mkdtemp(prefix="eval-kg-"))      # no leakage between clips
        res = replay(str(folder / r["file"]), judge=judge, knowledge=kg, verbose=False, use_audio=use_audio)
        flagged = bool(res.flags)
        alerted = bool(res.alerts)
        detector.add(flagged, r["cheat"])
        system.add(alerted, r["cheat"])
        fired = sorted({f["type"] for f in res.flags})
        for t in r["expect"]:
            d = per_type.setdefault(t, {"expected": 0, "flagged": 0, "alerted": 0})
            d["expected"] += 1
            d["flagged"] += t in fired
            d["alerted"] += any(a.get("event_type") == t for a in res.alerts)
        for j in res.judged:
            v = j["verdict"] or {}
            if v.get("latency_s"):
                latencies.append(v["latency_s"])
        if flagged and not alerted:
            if r["cheat"]:
                wrongly_cleared += 1
            else:
                filtered += 1
        clips.append({"file": r["file"], "label": "cheat" if r["cheat"] else "clean", "expect": r["expect"],
                      "flags": fired, "alerts": [a.get("event_type") for a in res.alerts],
                      "verdicts": [f"{j['event']['type']}:{(j['verdict'] or {}).get('verdict')}" for j in res.judged],
                      "calibrated": res.calibrated, "audio": res.has_audio,
                      "result": ("OK" if alerted == r["cheat"] else ("MISSED" if r["cheat"] else "FALSE ALARM")),
                      "notes": r["notes"]})
        print(f"  {clips[-1]['result']:11s} {r['file']}: flags={fired or '-'} alerts={clips[-1]['alerts'] or '-'}")
    report = {
        "folder": str(folder), "judge": judge_kind, "judge_models": list(getattr(judge.cfg, "models", ())) or ["mock"],
        "clips": len(rows), "cheat_clips": sum(r["cheat"] for r in rows), "run_seconds": round(time.time() - t_start, 1),
        "detector": detector.metrics(), "system": system.metrics(),
        "judge_filtered_false_flags": filtered, "judge_wrongly_cleared": wrongly_cleared,
        "judge_latency_s": {"median": round(statistics.median(latencies), 2) if latencies else None,
                            "max": round(max(latencies), 2) if latencies else None},
        "per_flag_type_recall": {t: {**d, "recall": round(d["flagged"] / d["expected"], 3)} for t, d in per_type.items()},
        "per_clip": clips,
    }
    return report


def to_markdown(rep: dict) -> str:
    def row(name, m):
        f = lambda x: "–" if x is None else f"{x:.0%}" if isinstance(x, float) else str(x)
        return (f"| {name} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {f(m['false_alarm_rate'])} | "
                f"{m['tp']} / {m['fp']} / {m['fn']} / {m['tn']} |")
    lines = [f"# Proctoring evaluation", "",
             f"{rep['clips']} clips ({rep['cheat_clips']} cheating, {rep['clips'] - rep['cheat_clips']} clean) · "
             f"judge: {', '.join(rep['judge_models'])}", "",
             "| Level | Precision | Recall | F1 | False-alarm rate | TP / FP / FN / TN |",
             "|---|---|---|---|---|---|", row("Detector (any flag)", rep["detector"]),
             row("System (popup after AI judge)", rep["system"]), "",
             f"AI judge filtered **{rep['judge_filtered_false_flags']}** false flags on clean clips and wrongly "
             f"cleared **{rep['judge_wrongly_cleared']}** cheating clips. Median judge latency: "
             f"{rep['judge_latency_s']['median'] or '–'} s.", ""]
    if rep["per_flag_type_recall"]:
        lines += ["| Expected flag | Clips | Detected | Recall |", "|---|---|---|---|"]
        for t, d in sorted(rep["per_flag_type_recall"].items()):
            lines.append(f"| {t} | {d['expected']} | {d['flagged']} | {d['recall']:.0%} |")
        lines.append("")
    lines += ["| Clip | Label | Result | Flags | Verdicts | Notes |", "|---|---|---|---|---|---|"]
    for c in rep["per_clip"]:
        lines.append(f"| {c['file']} | {c['label']} | {c['result']} | {', '.join(c['flags']) or '–'} | "
                     f"{', '.join(c['verdicts']) or '–'} | {c['notes']} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--judge", choices=["mock", "nim"], default="mock")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    print(f"evaluating {folder} with judge={args.judge}")
    rep = evaluate(folder, args.judge, use_audio=not args.no_audio)
    (folder / "eval_report.json").write_text(json.dumps(rep, indent=1))
    md = to_markdown(rep)
    (folder / "eval_report.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"written: {folder / 'eval_report.md'} and eval_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
