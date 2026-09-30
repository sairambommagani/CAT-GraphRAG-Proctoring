"""Frame loop: capture -> landmarks -> features -> overlay + CSV log."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from .features import EMA, EYE_A, EYE_B, FrameFeatures, extract_features, provisional_direction

CSV_FIELDS = ["t", "face_count", "yaw", "pitch", "roll", "iris_h", "iris_v", "ear", "blink",
              "yaw_s", "pitch_s", "iris_h_s", "iris_v_s", "direction", "label"]

LandmarkFn = Callable[[np.ndarray, int], list]


@dataclass
class Smoother:
    yaw: EMA = field(default_factory=lambda: EMA(0.3))
    pitch: EMA = field(default_factory=lambda: EMA(0.3))
    iris_h: EMA = field(default_factory=lambda: EMA(0.4))
    iris_v: EMA = field(default_factory=lambda: EMA(0.4))
    ear: EMA = field(default_factory=lambda: EMA(0.35))
    blink_threshold: Optional[float] = None   # set from the candidate's baseline eye openness

    def update(self, f: FrameFeatures) -> None:
        if f.face_count == 0:
            for e in (self.yaw, self.pitch, self.iris_h, self.iris_v, self.ear):
                e.reset()
        # Don't let blinks drag the iris estimate around. Once the candidate's own
        # baseline is known, "blink" means relative to their eye openness.
        if self.blink_threshold is not None and f.ear is not None:
            f.blink = f.ear < self.blink_threshold
        skip_iris = bool(f.blink)
        f.extra["yaw_s"] = self.yaw.update(f.yaw)
        f.extra["pitch_s"] = self.pitch.update(f.pitch)
        f.extra["iris_h_s"] = self.iris_h.value if skip_iris else self.iris_h.update(f.iris_h)
        f.extra["iris_v_s"] = self.iris_v.value if skip_iris else self.iris_v.update(f.iris_v)
        f.extra["ear_s"] = self.ear.update(f.ear)


def draw_overlay(frame: np.ndarray, faces: list, f: FrameFeatures) -> np.ndarray:
    out = frame.copy()
    for i, lm in enumerate(faces):
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        for eye in (EYE_A, EYE_B):
            for idx in (eye["corner_l"], eye["corner_r"]):
                cv2.circle(out, tuple(int(v) for v in lm[idx, :2]), 2, color, -1)
            cv2.circle(out, tuple(int(v) for v in lm[eye["iris"], :2]), 3, (255, 0, 255), -1)
        nose = lm[1, :2]
        yaw, pitch = f.extra.get("yaw_s"), f.extra.get("pitch_s")
        if i == 0 and yaw is not None and pitch is not None:
            tip = nose + np.array([np.sin(np.radians(yaw)), -np.sin(np.radians(pitch))]) * 120
            cv2.arrowedLine(out, tuple(int(v) for v in nose), tuple(int(v) for v in tip),
                            (0, 200, 255), 2, tipLength=0.2)

    lines = [f"faces: {f.face_count}   dir: {f.extra.get('direction', '')}"]
    if f.yaw is not None:
        lines.append(f"yaw {f.extra['yaw_s']:+.1f}  pitch {f.extra['pitch_s']:+.1f}  roll {f.roll:+.1f}")
        ih, iv = f.extra.get("iris_h_s"), f.extra.get("iris_v_s")
        if ih is not None:
            lines.append(f"iris h {ih:.2f}  v {iv:+.2f}  EAR {f.ear:.2f}{'  BLINK' if f.blink else ''}")
    if f.extra.get("label"):
        lines.append(f"label: {f.extra['label']}")
    alert = f.face_count != 1
    for k, text in enumerate(lines):
        cv2.putText(out, text, (10, 25 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if alert and k == 0 else (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run(frames: Iterable[tuple[np.ndarray, float]], landmarker: LandmarkFn,
        csv_path: Optional[str] = None, video_out: Optional[str] = None,
        show: bool = False, fps_hint: float = 30.0,
        label_fn: Optional[Callable[[], str]] = None) -> list[FrameFeatures]:
    """Process a stream of (frame, t_seconds). Returns all FrameFeatures.

    `landmarker(frame, ts_ms)` must return a list of (478,3) pixel-space arrays;
    injecting it keeps this loop testable without MediaPipe.
    `label_fn` returns the current ground-truth label (hotkeys in the CLI) so
    recordings double as labelled data for Phase 2/3 evaluation.
    """
    smoother = Smoother()
    history: list[FrameFeatures] = []
    writer = None
    csv_file = open(csv_path, "w", newline="") if csv_path else None
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore") if csv_file else None
    if csv_writer:
        csv_writer.writeheader()
    try:
        for frame, t in frames:
            h, w = frame.shape[:2]
            faces = landmarker(frame, int(t * 1000))
            f = extract_features(faces, t, w, h)
            smoother.update(f)
            f.extra["direction"] = provisional_direction(
                f.extra.get("yaw_s"), f.extra.get("pitch_s"), f.extra.get("iris_h_s"))
            f.extra["label"] = label_fn() if label_fn else ""
            history.append(f)
            if csv_writer:
                csv_writer.writerow({k: _fmt(v) for k, v in f.to_row().items()})

            if video_out or show:
                vis = draw_overlay(frame, faces, f)
                if video_out:
                    if writer is None:
                        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps_hint, (w, h))
                    writer.write(vis)
                if show:
                    cv2.imshow("proctor - gaze (q to quit)", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if label_fn and hasattr(label_fn, "on_key"):
                        label_fn.on_key(key)
    finally:
        if writer is not None:
            writer.release()
        if csv_file:
            csv_file.close()
        if show:
            cv2.destroyAllWindows()
    return history


def _fmt(v):
    if isinstance(v, float):
        return "" if np.isnan(v) else round(v, 4)
    return "" if v is None else v


def camera_frames(source, max_seconds: Optional[float] = None):
    """Yield (frame, t) from a webcam index or a video file path."""
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}")
    is_file = not isinstance(src, int)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, n = time.monotonic(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # For files use the file's clock; for webcams use wall-clock time.
            t = n / fps if is_file else time.monotonic() - start
            n += 1
            if max_seconds is not None and t > max_seconds:
                break
            yield frame, t
    finally:
        cap.release()


def summarize(history: list[FrameFeatures]) -> dict:
    n = len(history)
    if n == 0:
        return {"frames": 0}
    dur = history[-1].t - history[0].t
    face = [f for f in history if f.face_count >= 1]
    dirs: dict[str, int] = {}
    for f in history:
        d = f.extra.get("direction", "")
        dirs[d] = dirs.get(d, 0) + 1
    return {
        "frames": n,
        "duration_s": round(dur, 1),
        "fps": round((n - 1) / dur, 1) if dur > 0 else None,
        "no_face_pct": round(100 * sum(f.face_count == 0 for f in history) / n, 1),
        "multi_face_pct": round(100 * sum(f.face_count > 1 for f in history) / n, 1),
        "blink_pct": round(100 * sum(bool(f.blink) for f in face) / max(len(face), 1), 1),
        "direction_pct": {k: round(100 * v / n, 1) for k, v in sorted(dirs.items(), key=lambda x: -x[1])},
    }
