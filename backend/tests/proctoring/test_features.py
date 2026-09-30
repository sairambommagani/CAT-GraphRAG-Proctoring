import csv

import cv2
import numpy as np
import pytest

from proctor.features import (EMA, EYE_A, EYE_B, FACE_MODEL_3D, POSE_IDX, camera_matrix,
                              extract_features, eye_aspect_ratio, head_pose, iris_position,
                              provisional_direction, ypr_to_rotation)
from proctor.pipeline import run, summarize

from synth import W, H, synthetic_face  # noqa: E402


# ---- iris ------------------------------------------------------------------

@pytest.mark.parametrize("h", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("v", [-0.1, 0.0, 0.15])
def test_iris_position_recovers_placement(h, v):
    lm = synthetic_face(iris_h=h, iris_v=v)
    for eye in (EYE_A, EYE_B):
        rh, rv = iris_position(lm, eye)
        assert rh == pytest.approx(h, abs=1e-6)
        assert rv == pytest.approx(v, abs=1e-6)


def test_iris_vertical_positive_means_below():
    lm = synthetic_face(iris_v=0.1)
    iris_y = lm[EYE_A["iris"], 1]
    corner_y = (lm[EYE_A["corner_l"], 1] + lm[EYE_A["corner_r"], 1]) / 2
    assert iris_y > corner_y   # image y grows downward


def test_iris_ratio_invariant_to_scale_and_translation():
    a = iris_position(synthetic_face(iris_h=0.7), EYE_A)
    b = iris_position(synthetic_face(iris_h=0.7, scale=0.6, shift=(90, -40)), EYE_A)
    assert a == pytest.approx(b, abs=1e-6)


# ---- EAR -------------------------------------------------------------------

def test_ear_drops_when_eyes_close():
    open_ear = eye_aspect_ratio(synthetic_face(eye_open=1.0), EYE_A["ear"])
    closed_ear = eye_aspect_ratio(synthetic_face(eye_open=0.1), EYE_A["ear"])
    assert open_ear > 0.2 > closed_ear


# ---- head pose -------------------------------------------------------------

def test_frontal_face_is_zero_pose():
    yaw, pitch, roll = head_pose(synthetic_face(), W, H)
    assert abs(yaw) < 1 and abs(pitch) < 1 and abs(roll) < 1


@pytest.mark.parametrize("yaw,pitch,roll", [(20, 0, 0), (-30, 0, 0), (0, 15, 0),
                                            (0, -20, 0), (0, 0, 10), (15, -10, 5)])
def test_head_pose_recovers_known_rotation(yaw, pitch, roll):
    est = head_pose(synthetic_face(yaw, pitch, roll), W, H)
    assert est == pytest.approx((yaw, pitch, roll), abs=1.5)


def test_head_pose_sign_semantics():
    # +yaw must mean the face normal (pointing at the camera, -z) swings toward +x.
    R = ypr_to_rotation(-25, 0, 0)          # raw rotation used for +25 output yaw
    assert (R @ np.array([0, 0, -1]))[0] > 0
    # +pitch must mean the face normal swings upward (-y in image coords).
    R = ypr_to_rotation(0, -20, 0)
    assert (R @ np.array([0, 0, -1]))[1] < 0


def test_head_pose_noise_robustness():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(50):
        lm = synthetic_face(yaw=20, pitch=-10)
        lm[:, :2] += rng.normal(0, 1.5, size=(478, 2))   # ~1.5px landmark jitter
        errs.append(np.abs(np.array(head_pose(lm, W, H)[:2]) - [20, -10]).max())
    assert np.median(errs) < 4


# ---- aggregate + direction ---------------------------------------------------

def test_extract_features_no_face():
    f = extract_features([], 1.0, W, H)
    assert f.face_count == 0 and f.yaw is None


def test_extract_features_picks_largest_face():
    near = synthetic_face(yaw=0, scale=1.0)
    far = synthetic_face(yaw=30, scale=0.4, shift=(200, 0))
    f = extract_features([far, near], 0.0, W, H)
    assert f.face_count == 2 and abs(f.yaw) < 2


def test_provisional_direction():
    assert provisional_direction(0, 0, 0.5) == "CENTER"
    assert provisional_direction(30, 0, 0.5) == "IMG-RIGHT"
    assert provisional_direction(0, 0, 0.3) == "IMG-LEFT"
    assert provisional_direction(0, -25, 0.5) == "DOWN"
    assert provisional_direction(None, None, None) == "NO FACE"


def test_ema_smooths_and_ignores_missing():
    e = EMA(0.5)
    assert e.update(10.0) == 10.0
    assert e.update(None) == 10.0
    assert e.update(float("nan")) == 10.0
    assert e.update(20.0) == 15.0


# ---- end-to-end loop with a fake landmarker ----------------------------------

def test_pipeline_end_to_end(tmp_path):
    script = ([[synthetic_face()]] * 30 + [[synthetic_face(yaw=35)]] * 30
              + [[]] * 10 + [[synthetic_face(), synthetic_face(scale=0.5, shift=(150, 0))]] * 10)
    frames = [(np.zeros((H, W, 3), np.uint8), i / 30) for i in range(len(script))]
    calls = iter(script)
    csv_path, vid_path = tmp_path / "f.csv", tmp_path / "f.mp4"

    hist = run(frames, lambda fr, ts: next(calls), csv_path=str(csv_path),
               video_out=str(vid_path), label_fn=lambda: "normal")

    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == len(script) == len(hist)
    assert rows[0]["direction"] == "CENTER"
    assert rows[59]["direction"] == "IMG-RIGHT"        # EMA has converged by then
    assert rows[65]["direction"] == "NO FACE" and rows[65]["yaw"] == ""
    assert rows[-1]["face_count"] == "2"
    assert all(r["label"] == "normal" for r in rows)

    s = summarize(hist)
    assert s["no_face_pct"] == pytest.approx(12.5) and s["multi_face_pct"] == pytest.approx(12.5)
    cap = cv2.VideoCapture(str(vid_path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(script)
