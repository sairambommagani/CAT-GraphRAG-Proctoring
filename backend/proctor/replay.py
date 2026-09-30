"""Replay a recorded webcam video (with its microphone track) through the exact proctoring pipeline.

    python -m proctor.replay my_test.mp4                 # offline judge, no API calls
    python -m proctor.replay my_test.mp4 --judge nim     # also ask the NVIDIA judge chain (uses quota)

Record the video with the Windows Camera app (or any webcam recorder), sitting as
in the exam: look at the screen and stay quiet for the first 5 seconds (that is the
calibration: gaze baseline + room noise floor), then do the behaviours you want to
test. The report shows, second by second, what each detector saw (gaze state, faces,
objects with their best raw score even below threshold, speech), and every flag that
fired - so a miss can be traced to its cause instead of guessed.

The same engine powers `python -m proctor.evaluate` (precision/recall on a labelled set).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import cv2

from .audio import SAMPLE_RATE, Transcriber, load_media_audio
from .judge import MockJudge, NIMJudge
from .objects import CLASS_THRESHOLDS, PROHIBITED, SceneAnalyzer
from .retention import EvidenceStore
from .service import ProctorSession
from .tracker import FaceTracker

FPS = 8.0
CALIB = (0.5, 3.5)          # seconds of the video used as the "look at the dot" baseline


@dataclass
class ReplayResult:
    video: str
    duration_s: float = 0.0
    calibrated: bool = False
    mode: str = ""
    has_audio: bool = False
    flags: list = field(default_factory=list)          # detector events (dicts)
    judged: list = field(default_factory=list)         # {event, verdict}
    alerts: list = field(default_factory=list)         # popups shown (judge said fraud)
    seconds: dict = field(default_factory=dict)        # per-second detector summary
    log: list = field(default_factory=list)


def replay_file(video: str, judge=None, knowledge=None, transcriber=None, object_detection: bool = True,
                verbose: bool = True, use_audio: bool = True) -> ReplayResult:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / FPS))
    audio = load_media_audio(video) if use_audio else None

    scene = SceneAnalyzer() if object_detection else None
    judge = judge or MockJudge()
    store = EvidenceStore(tempfile.mkdtemp(prefix="replay-"))
    s = ProctorSession("replay", FaceTracker(), judge, store, scene_analyzer=scene,
                       transcriber=transcriber, knowledge=knowledge)
    res = ReplayResult(video, has_audio=audio is not None)
    say = print if verbose else (lambda *a, **k: None)
    say(f"video {video}: {src_fps:.0f} fps, analysing {FPS:.0f} fps | objects "
        f"{scene.model_name if scene else 'off'} | audio {'yes' if audio is not None else 'none'} | "
        f"judge {getattr(judge.cfg, 'model', 'mock')}")

    last_scene = {"s": None}
    if scene is not None:
        def wrapped(frame, ts, faces):
            out = scene(frame, ts, faces)
            if out is not None:
                last_scene["s"] = out
            return out
        s.scene_analyzer = wrapped

    async def notify(msg):
        if msg["type"] == "alert":
            res.alerts.append(msg)

    audio_pos = 0

    async def handle(events, ready):
        for ev in events:
            res.flags.append(ev.to_dict())
            say(f"  >>> FLAG {ev.type}{' (' + ev.direction + ')' if ev.direction else ''} at {ev.t_trigger:.1f}s")
        for e in ready:
            rec = await s.process_evidence(e, notify)
            v = rec.get("verdict", {})
            res.judged.append({"event": e.event.to_dict(), "verdict": v})
            say(f"  >>> JUDGE {e.event.type}: {v.get('verdict')} {v.get('confidence', 0):.2f} "
                f"[{v.get('rule') or '-'}] {v.get('model', '')} | {v.get('reason', '')}")

    async def run():
        nonlocal audio_pos
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if (n - 1) % step:
                continue
            t = (n - 1) / src_fps
            res.duration_s = t
            if not res.calibrated and CALIB[0] <= t < CALIB[1] and s.collector._current is None:
                s.calib_target("center")
            if not res.calibrated and t >= CALIB[1]:
                s.calib_target_end()
                cal = s.calib_finish()
                res.calibrated, res.mode = cal["ok"], cal["mode"]
                say(f"calibration at {t:.1f}s: ok={cal['ok']} mode={cal['mode']} {cal.get('error') or ''}"
                    f" | mic noise floor {cal['mic']['noise_floor_db']} dB")
            if audio is not None:                     # feed the audio up to this frame's time
                end = min(len(audio), int(t * SAMPLE_RATE))
                if end > audio_pos:
                    await handle(s.ingest_audio(end / SAMPLE_RATE, audio[audio_pos:end]), [])
                    audio_pos = end
            jpeg = cv2.imencode(".jpg", cv2.resize(frame, (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            status, events, ready = s.ingest(t, jpeg)
            row = s.buffer._frames[-1].row if len(s.buffer) else {}
            best = {}
            for b in (last_scene["s"].raw if last_scene["s"] is not None else []):
                if b.label in PROHIBITED:
                    best[b.label] = max(best.get(b.label, 0), round(b.score, 2))
            entry = {"t": round(t, 2), "state": s.detector.state, "faces": row.get("face_count"),
                     "gaze": [row.get("gaze_x"), row.get("gaze_y")], "objects": row.get("objects", []),
                     "foreign_hands": row.get("foreign_hands", 0), "persons": row.get("persons", 0),
                     "speaking": s.speech.speaking, "mouth": row.get("mouth"), "best_scores": best}
            res.log.append(entry)
            r = res.seconds.setdefault(int(t), {"states": set(), "faces": set(), "objects": set(), "hands": 0,
                                                "best": {}, "speech": False})
            r["states"].add(entry["state"]); r["faces"].add(entry["faces"]); r["objects"].update(entry["objects"])
            r["hands"] = max(r["hands"], entry["foreign_hands"])
            r["speech"] = r["speech"] or entry["speaking"]
            for k, v in best.items():
                r["best"][k] = max(r["best"].get(k, 0), v)
            await handle(events, ready)
        for e in s.clips.flush():
            await handle([], [e])

    asyncio.run(run())
    if scene is not None:
        scene.close()
    return res


def print_report(res: ReplayResult) -> None:
    print("\nsecond | state          | faces | voice | counted objects | best raw scores (threshold)")
    for sec in sorted(res.seconds):
        r = res.seconds[sec]
        best = ", ".join(f"{k} {v:.2f} ({CLASS_THRESHOLDS.get(k, 0.35):.2f})" for k, v in sorted(r["best"].items())) or "-"
        faces = "/".join(str(f) for f in sorted(x for x in r["faces"] if x is not None))
        print(f"{sec:6d} | {'/'.join(sorted(r['states'])):14s} | {faces:5s} | {'yes' if r['speech'] else '-':5s} | "
              f"{', '.join(sorted(r['objects'])) or '-':15s} | {best}{' | foreign hand' if r['hands'] else ''}")
    print(f"\nflags: {len(res.flags)}  popups (judge said fraud): {len(res.alerts)}")
    for ev in res.flags:
        print(f"  {ev['t_trigger']:6.1f}s  {ev['type']}{' (' + ev['direction'] + ')' if ev['direction'] else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--judge", choices=["mock", "nim"], default="mock")
    ap.add_argument("--no-audio", action="store_true", help="ignore the video's microphone track")
    ap.add_argument("--transcribe", action="store_true", help="run speech-to-text on flagged clips")
    ap.add_argument("--json", help="also write the per-frame log here")
    args = ap.parse_args()
    from .knowledge import ProctorKnowledge
    kg = ProctorKnowledge(tempfile.mkdtemp(prefix="replay-kg-"))
    res = replay_file(args.video, NIMJudge() if args.judge == "nim" else MockJudge(), knowledge=kg,
                      transcriber=Transcriber() if args.transcribe else None, use_audio=not args.no_audio)
    print_report(res)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res.log, f)
        print(f"per-frame log written to {args.json}")


if __name__ == "__main__":
    main()
