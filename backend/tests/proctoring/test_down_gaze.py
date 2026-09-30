"""Regression tests from the first real-webcam run (2026-09-25 recording):
looking down at a lap/phone was NOT flagged, while sideways looks were.
Cause: drooping eyelids read as a 'blink', freezing the iris signal, and the
calibration model leaned on iris for the vertical axis."""
import numpy as np
import pytest

from proctor.calibration import TARGETS, CalibrationCollector
from proctor.detector import AnomalyDetector
from proctor.features import extract_features
from proctor.pipeline import Smoother
from proctor.service import DownCue
from synth import H, W, looking_at, looking_down_at_lap

FPS = 8


def rows_for(fn, seconds, t0, rng, sm):
    out = []
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        f = extract_features(fn(), t, W, H)
        sm.update(f)
        out.append((t, f.to_row()))
    return out


def calibrate(rng, head_gain):
    sm, c, t = Smoother(), CalibrationCollector(), 0.0
    for name, (sx, sy) in TARGETS.items():
        c.start_target(name, t)
        for tt, row in rows_for(lambda: [looking_at(sx, sy, rng, 1.0, head_gain=head_gain)], 2.0, t, rng, sm):
            c.add(tt, row)
        c.end_target()
        t += 2.0
    return c.fit(), sm, t


def run_session(cal, sm, t, rng, segments):
    det, cue, events, states = AnomalyDetector(), DownCue(cal), [], []
    for fn, secs in segments:
        for tt, row in rows_for(fn, secs, t, rng, sm):
            gaze = cue.apply(tt, row, cal.predict(row))
            events += det.update(tt, row["face_count"], gaze)
            states.append(det.state)
        t += secs
    return events, states


@pytest.mark.parametrize("head_gain", [1.0, 0.3])
def test_looking_down_at_lap_is_flagged(head_gain):
    rng = np.random.default_rng(0)
    cal, sm, t = calibrate(rng, head_gain)
    assert cal.ok
    events, _ = run_session(cal, sm, t, rng, [
        (lambda: [looking_at(0, 0, rng, 1.0, head_gain=head_gain)], 3),
        (lambda: [looking_down_at_lap(rng, 1.0)], 5),
        (lambda: [looking_at(0, 0, rng, 1.0, head_gain=head_gain)], 3),
    ])
    assert [(e.type, e.direction) for e in events] == [("OFFSCREEN_SUSTAINED", "down")]


def test_eyes_only_glance_down_without_head_tilt_is_flagged():
    """Pure eye movement down with drooping lids (phone on the desk)."""
    rng = np.random.default_rng(1)
    cal, sm, t = calibrate(rng, 0.3)
    events, _ = run_session(cal, sm, t, rng, [
        (lambda: [looking_at(0, 0, rng, 1.0, head_gain=0.3)], 2),
        (lambda: [looking_down_at_lap(rng, 1.0, pitch_drop=4.0, eye_open=0.4)], 5),
        (lambda: [looking_at(0, 0, rng, 1.0, head_gain=0.3)], 2),
    ])
    assert any(e.type == "OFFSCREEN_SUSTAINED" and e.direction == "down" for e in events)


def test_normal_blinks_do_not_trigger():
    rng = np.random.default_rng(2)
    cal, sm, t = calibrate(rng, 1.0)
    segs = []
    for _ in range(12):              # a blink (~0.25 s) every ~2.5 s for 30 s
        segs += [(lambda: [looking_at(0.2, 0.1, rng, 1.0)], 2.25),
                 (lambda: [looking_at(0.2, 0.1, rng, 1.0, eye_open=0.08)], 0.25)]
    events, states = run_session(cal, sm, t, rng, segs)
    assert events == []
    assert states.count("off_screen") <= 2


def test_reading_bottom_of_screen_is_not_flagged():
    """Looking at the lowest answer option must stay on-screen."""
    rng = np.random.default_rng(3)
    cal, sm, t = calibrate(rng, 1.0)
    events, states = run_session(cal, sm, t, rng, [
        (lambda: [looking_at(0.0, 0.85, rng, 1.0, eye_open=0.8)], 10)])
    assert events == [] and "off_screen" not in states
