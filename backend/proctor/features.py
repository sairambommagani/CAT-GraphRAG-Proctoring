"""Per-frame gaze / face features computed from MediaPipe Face Mesh landmarks.

Everything here is pure NumPy/OpenCV math on a (478, 3) landmark array in
pixel coordinates, so it can be unit-tested without a camera or model.

Conventions
-----------
* "image-left" / "image-right" refer to the raw (un-mirrored) camera image.
* Iris horizontal ratio: 0.0 = iris at image-left eye corner, 1.0 = image-right
  corner, ~0.5 = centred. Both eyes use the same direction so they can be
  averaged.
* Iris vertical offset: signed distance of the iris centre from the line
  joining the two eye corners, divided by eye width. Positive = below the line
  (image y grows downward). Independent of eyelids, so blinks don't move it.
* Head pose (degrees): 0/0/0 = facing the camera. +yaw = face turned toward
  image-right, +pitch = looking up, roll = in-plane tilt.

Absolute values vary per person, camera and glasses; Phase 2 calibration makes
all thresholds relative to each candidate's own baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import cv2
import numpy as np

# ---- MediaPipe Face Mesh (478-point, refine_landmarks) indices ------------
# Eye on the image-left side of the frame (the subject's right eye)
EYE_A = dict(corner_l=33, corner_r=133, iris=468,
             ear=(33, 160, 158, 133, 153, 144))
# Eye on the image-right side of the frame (the subject's left eye)
EYE_B = dict(corner_l=362, corner_r=263, iris=473,
             ear=(362, 385, 387, 263, 373, 380))

# Landmarks used for head pose, and a generic 3D face model in a camera-aligned
# frame (x right, y down, z away from camera; nose tip at the origin; units ~0.1 mm).
POSE_IDX = (1, 152, 33, 263, 61, 291)  # nose tip, chin, eye corners, mouth corners
FACE_MODEL_3D = np.array([
    [0.0,     0.0,    0.0],   # nose tip
    [0.0,   330.0,   65.0],   # chin
    [-225.0, -170.0, 135.0],  # image-left eye outer corner
    [225.0, -170.0,  135.0],  # image-right eye outer corner
    [-150.0,  150.0, 125.0],  # image-left mouth corner
    [150.0,   150.0, 125.0],  # image-right mouth corner
], dtype=np.float64)

BLINK_EAR_THRESHOLD = 0.20

# Inner-lip landmarks: upper lip centre, lower lip centre, left/right inner corners.
MOUTH = dict(upper=13, lower=14, left=78, right=308)


@dataclass
class FrameFeatures:
    t: float                       # seconds since start
    face_count: int
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    iris_h: Optional[float] = None  # 0..1, mean of both eyes
    iris_v: Optional[float] = None  # signed, fraction of eye width
    ear: Optional[float] = None     # eye aspect ratio, mean of both eyes
    blink: Optional[bool] = None
    mouth: Optional[float] = None   # mouth aspect ratio (lip gap / mouth width); talking makes it oscillate
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop("extra")
        row.update(self.extra)
        return row


# ---- individual features ---------------------------------------------------

def iris_position(lm: np.ndarray, eye: dict) -> tuple[float, float]:
    """Return (horizontal ratio, vertical offset) of the iris within one eye."""
    c1 = lm[eye["corner_l"], :2]
    c2 = lm[eye["corner_r"], :2]
    iris = lm[eye["iris"], :2]
    axis = c2 - c1
    width2 = float(axis @ axis)
    if width2 < 1e-9:
        return 0.5, 0.0
    width = np.sqrt(width2)
    rel = iris - c1
    h = float(rel @ axis) / width2
    # 2D cross product: sign tells which side of the corner line the iris is on.
    v = float(axis[0] * rel[1] - axis[1] * rel[0]) / (width * width)
    return h, v


def eye_aspect_ratio(lm: np.ndarray, idx: tuple[int, ...]) -> float:
    """Soukupová & Čech (2016) EAR: (|p2-p6| + |p3-p5|) / (2|p1-p4|)."""
    p = lm[list(idx), :2]
    horiz = np.linalg.norm(p[0] - p[3])
    if horiz < 1e-9:
        return 0.0
    vert = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return float(vert / (2.0 * horiz))


def mouth_aspect_ratio(lm: np.ndarray) -> float:
    """Inner-lip gap divided by mouth width. ~0 closed, ~0.3-0.6 open; speech makes it
    oscillate a few times a second, which is what the audio module fuses with the mic."""
    top, bot = lm[MOUTH["upper"], :2], lm[MOUTH["lower"], :2]
    width = np.linalg.norm(lm[MOUTH["left"], :2] - lm[MOUTH["right"], :2])
    if width < 1e-9:
        return 0.0
    return float(np.linalg.norm(top - bot) / width)


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole approximation: focal length ~ image width, centre = image centre."""
    f = float(width)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)


def head_pose(lm: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    """Estimate (yaw, pitch, roll) in degrees with solvePnP on 6 stable landmarks."""
    img_pts = lm[list(POSE_IDX), :2].astype(np.float64)
    K = camera_matrix(width, height)
    ok, rvec, _ = cv2.solvePnP(FACE_MODEL_3D, img_pts, K, np.zeros(4),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return float("nan"), float("nan"), float("nan")
    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = rotation_to_ypr(R)
    # In the camera frame (y down, z away) a positive raw yaw turns the face
    # normal toward image-left and a positive raw pitch tilts it down. Flip both
    # so the outputs read naturally: +yaw = face turned toward image-right,
    # +pitch = looking up.
    return -yaw, -pitch, roll


def rotation_to_ypr(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose R = Rz(roll) @ Ry(yaw) @ Rx(pitch) into degrees."""
    yaw = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(yaw), float(pitch), float(roll)


def ypr_to_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Inverse of rotation_to_ypr (used by tests and synthetic data)."""
    y, p, r = np.radians([yaw, pitch, roll])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ---- aggregate -------------------------------------------------------------

def extract_features(faces: list[np.ndarray], t: float, width: int, height: int) -> FrameFeatures:
    """Compute FrameFeatures for the primary (largest) face in the frame."""
    feats = FrameFeatures(t=t, face_count=len(faces))
    if not faces:
        return feats
    # Primary face = largest bounding box (closest to camera).
    def face_area(f: np.ndarray) -> float:
        p = f[list(POSE_IDX), :2]
        return float(np.ptp(p[:, 0]) * np.ptp(p[:, 1]))
    lm = max(faces, key=face_area)

    feats.yaw, feats.pitch, feats.roll = head_pose(lm, width, height)
    ha, va = iris_position(lm, EYE_A)
    hb, vb = iris_position(lm, EYE_B)
    feats.iris_h = (ha + hb) / 2.0
    feats.iris_v = (va + vb) / 2.0
    feats.ear = (eye_aspect_ratio(lm, EYE_A["ear"]) + eye_aspect_ratio(lm, EYE_B["ear"])) / 2.0
    feats.blink = feats.ear < BLINK_EAR_THRESHOLD
    if lm.shape[0] > max(MOUTH.values()):
        feats.mouth = mouth_aspect_ratio(lm)
    return feats


class EMA:
    """Exponential moving average for noisy per-frame signals (NaN/None-safe)."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: Optional[float]) -> Optional[float]:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return self.value
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


def provisional_direction(yaw: Optional[float], pitch: Optional[float],
                          iris_h: Optional[float], yaw_thr: float = 20.0,
                          pitch_thr: float = 15.0, iris_thr: float = 0.12) -> str:
    """Rough uncalibrated gaze label for the live overlay only.

    Phase 2 replaces these global thresholds with a per-candidate envelope.
    Labels are in raw-image terms (not mirrored).
    """
    if yaw is None:
        return "NO FACE"
    parts = []
    if pitch is not None and pitch > pitch_thr:
        parts.append("UP")
    elif pitch is not None and pitch < -pitch_thr:
        parts.append("DOWN")
    if yaw > yaw_thr or (iris_h is not None and iris_h > 0.5 + iris_thr):
        parts.append("IMG-RIGHT")
    elif yaw < -yaw_thr or (iris_h is not None and iris_h < 0.5 - iris_thr):
        parts.append("IMG-LEFT")
    return "-".join(parts) if parts else "CENTER"
