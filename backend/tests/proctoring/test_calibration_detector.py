import numpy as np
import pytest

from proctor.calibration import TARGETS, CalibrationCollector, GazeCalibration
from proctor.detector import AnomalyDetector, DetectorConfig
from proctor.features import extract_features
from proctor.pipeline import Smoother
from synth import H, W, looking_at

FPS = 10


def feature_rows(sx, sy, t0, seconds, rng, noise=1.0):
    sm = Smoother()
    rows = []
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        f = extract_features([looking_at(sx, sy, rng, noise)], t, W, H)
        sm.update(f)
        rows.append((t, f.to_row()))
    return rows


def calibrate(rng, noise=1.0) -> GazeCalibration:
    c = CalibrationCollector()
    t = 0.0
    for name, (sx, sy) in TARGETS.items():
        c.start_target(name, t)
        for tt, row in feature_rows(sx, sy, t, 2.0, rng, noise):
            c.add(tt, row)
        c.end_target()
        t += 2.0
    return c.fit()


def test_calibration_fits_and_predicts_targets():
    cal = calibrate(np.random.default_rng(0))
    assert cal.ok and cal.rmse_x < 0.2 and cal.rmse_y < 0.2
    for name, err in cal.per_target_error.items():
        assert err < 0.3, name


def test_calibration_extrapolates_offscreen_directions():
    rng = np.random.default_rng(1)
    cal = calibrate(rng)
    for (sx, sy), expect in [((2.2, 0), "right"), ((-2.2, 0), "left"), ((0, 2.2), "down")]:
        row = feature_rows(sx, sy, 0, 1.0, rng)[-1][1]
        x, y = cal.predict(row)
        assert max(abs(x), abs(y)) > 1.4
        from proctor.detector import gaze_direction
        assert gaze_direction(x, y) == expect


def test_calibration_skips_settling_period_and_rejects_missing_targets():
    c = CalibrationCollector()
    c.start_target("center", 0.0)
    row = {"face_count": 1, "blink": False, "yaw_s": 0, "pitch_s": 0, "iris_h_s": .5, "iris_v_s": 0}
    c.add(0.2, row)                          # inside settle window -> ignored
    assert c.samples["center"] == []
    with pytest.raises(ValueError):
        c.fit()


def test_calibration_roundtrip_serialisation():
    cal = calibrate(np.random.default_rng(2))
    cal2 = GazeCalibration.from_dict(cal.to_dict())
    row = feature_rows(0.5, -0.5, 0, 1, np.random.default_rng(3))[-1][1]
    assert cal.predict(row) == pytest.approx(cal2.predict(row))


# ---- detector (driven directly with gaze points) ---------------------------------

def run(det, segments):
    """segments: list of (seconds, face_count, gaze or None)."""
    t, events = 0.0, []
    for secs, fc, g in segments:
        for _ in range(int(secs * FPS)):
            events += det.update(t, fc, g)
            t += 1 / FPS
    return events


def test_sustained_offscreen_fires_once():
    ev = run(AnomalyDetector(), [(2, 1, (0, 0)), (6, 1, (0, 1.8)), (2, 1, (0, 0))])
    assert [e.type for e in ev] == ["OFFSCREEN_SUSTAINED"]
    assert ev[0].direction == "down" and ev[0].details["duration_s"] >= 3.0
    assert ev[0].t_start == pytest.approx(2.0, abs=0.15)


def test_repeated_glances_same_direction():
    segs = [(2, 1, (0, 0))]
    for _ in range(3):
        segs += [(1.0, 1, (-1.7, 0.1)), (3, 1, (0, 0))]
    ev = run(AnomalyDetector(), segs)
    assert [e.type for e in ev] == ["REPEATED_GLANCES"]
    assert ev[0].direction == "left" and ev[0].details["glances"] == 3


def test_micro_glances_and_mixed_directions_ignored():
    segs = [(2, 1, (0, 0))]
    for _ in range(6):                              # 0.2 s saccades: below min_glance_s
        segs += [(0.2, 1, (0, 1.8)), (1, 1, (0, 0))]
    for g in [(1.8, 0), (-1.8, 0), (0, -1.8)]:     # one glance each way
        segs += [(1, 1, g), (2, 1, (0, 0))]
    assert run(AnomalyDetector(), segs) == []


def test_hysteresis_keeps_borderline_gaze_in_one_glance():
    # hovers between exit (1.10) and enter (1.25): should be ONE continuous excursion
    segs = [(1, 1, (0, 0)), (0.5, 1, (1.4, 0))] + [(0.3, 1, (1.15, 0)), (0.3, 1, (1.4, 0))] * 5 + [(2, 1, (0, 0))]
    ev = run(AnomalyDetector(), segs)
    assert [e.type for e in ev] == ["OFFSCREEN_SUSTAINED"]


def test_no_face_and_dropout_tolerance():
    det = AnomalyDetector()
    # 0.3 s detection dropouts must not trigger anything
    ev = run(det, [(1, 1, (0, 0)), (0.3, 0, None), (1, 1, (0, 0)), (0.3, 0, None), (1, 1, (0, 0))])
    assert ev == []
    ev = run(AnomalyDetector(), [(1, 1, (0, 0)), (4, 0, None)])
    assert [e.type for e in ev] == ["NO_FACE"]


def test_multiple_faces():
    ev = run(AnomalyDetector(), [(1, 1, (0, 0)), (1.5, 2, (0, 0))])
    assert [e.type for e in ev] == ["MULTIPLE_FACES"]


def test_cooldown_limits_repeat_events():
    det = AnomalyDetector(DetectorConfig(cooldown_s=20))
    segs = []
    for _ in range(3):                               # three 4-second look-aways, 5 s apart
        segs += [(4, 1, (0, 1.8)), (1, 1, (0, 0))]
    ev = run(det, segs)
    assert [e.type for e in ev].count("OFFSCREEN_SUSTAINED") == 1


# ---- regressions from the second real-webcam run (2026-09-25 15:43 recording) ----

def test_excursion_direction_is_dominant_not_first():
    """Glanced sideways first, then looked down for longer, all in one excursion.
    The event must say 'down' (where the gaze actually spent its time)."""
    ev = run(AnomalyDetector(), [(2, 1, (0, 0)), (1.0, 1, (1.6, 0.2)), (4, 1, (0.1, 1.8)), (2, 1, (0, 0))])
    assert [(e.type, e.direction) for e in ev] == [("OFFSCREEN_SUSTAINED", "down")]


def test_cooldown_is_per_direction():
    """A sideways look-away must not suppress a later phone-in-lap look-away."""
    ev = run(AnomalyDetector(), [(1, 1, (0, 0)), (4, 1, (1.7, 0)), (3, 1, (0, 0)),
                                 (5, 1, (0, 1.8)), (2, 1, (0, 0))])
    assert [(e.type, e.direction) for e in ev] == [("OFFSCREEN_SUSTAINED", "right"),
                                                  ("OFFSCREEN_SUSTAINED", "down")]
    # ...but the same direction repeating quickly is still rate-limited
    ev = run(AnomalyDetector(), [(4, 1, (0, 1.8)), (2, 1, (0, 0)), (4, 1, (0, 1.8)), (2, 1, (0, 0))])
    assert [e.direction for e in ev] == ["down"]
