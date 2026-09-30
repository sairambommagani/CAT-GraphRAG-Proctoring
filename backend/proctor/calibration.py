"""Per-candidate gaze calibration.

At exam start the candidate looks at 5 on-screen targets (centre + 4 corners).
We fit a small ridge regression per axis that maps smoothed features to
normalised screen coordinates:

    x_screen ~ yaw_s + iris_h_s      (-1 = left edge, +1 = right edge)
    y_screen ~ pitch_s + iris_v_s    (-1 = top edge,  +1 = bottom edge)

Why a learned mapping rather than fixed thresholds:
* Camera placement shifts every signal (a laptop cam below the screen makes
  "looking at the screen" read as "looking up").
* Head-vs-eye contribution differs by person; the regression learns the mix.
* Sign conventions (mirroring, which way the camera faces) are learned too.

Afterwards, |x| or |y| well beyond 1 means the gaze left the screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Target positions in normalised screen coordinates (edges = +-1).
TARGETS: dict[str, tuple[float, float]] = {
    "center": (0.0, 0.0),
    "top_left": (-0.9, -0.9),
    "top_right": (0.9, -0.9),
    "bottom_right": (0.9, 0.9),
    "bottom_left": (-0.9, 0.9),
}

X_FEATURES = ("yaw_s", "iris_h_s")
Y_FEATURES = ("pitch_s", "iris_v_s")


@dataclass
class AxisModel:
    features: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, sample: dict) -> Optional[float]:
        vals = [sample.get(f) for f in self.features]
        if any(v is None for v in vals):
            return None
        z = (np.asarray(vals, float) - self.mean) / self.std
        return float(z @ self.coef + self.intercept)

    def to_dict(self) -> dict:
        return {"features": list(self.features), "mean": self.mean.tolist(), "std": self.std.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept}

    @classmethod
    def from_dict(cls, d: dict) -> "AxisModel":
        return cls(tuple(d["features"]), np.array(d["mean"]), np.array(d["std"]),
                   np.array(d["coef"]), float(d["intercept"]))


def _fit_ridge(X: np.ndarray, y: np.ndarray, features: tuple[str, ...], lam: float) -> AxisModel:
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    Z = (X - mean) / std
    yc = y - y.mean()
    coef = np.linalg.solve(Z.T @ Z + lam * len(y) * np.eye(Z.shape[1]), Z.T @ yc)
    return AxisModel(features, mean, std, coef, float(y.mean()))


@dataclass
class GazeCalibration:
    x_model: AxisModel
    y_model: AxisModel
    rmse_x: float
    rmse_y: float
    n_samples: int
    per_target_error: dict = field(default_factory=dict)
    # Per-candidate baselines used by the down-gaze cue (see service.DownCue):
    ear_base: Optional[float] = None       # median eye openness while looking at the screen
    pitch_lo: Optional[float] = None       # lowest head pitch used for the bottom targets
    pitch_hi: Optional[float] = None

    # Calibration is only trusted if targets are separable and fit error is low.
    MAX_RMSE = 0.35

    @property
    def ok(self) -> bool:
        return self.rmse_x <= self.MAX_RMSE and self.rmse_y <= self.MAX_RMSE

    def predict(self, sample: dict) -> Optional[tuple[float, float]]:
        x, y = self.x_model.predict(sample), self.y_model.predict(sample)
        if x is None or y is None:
            return None
        return x, y

    def to_dict(self) -> dict:
        return {"x_model": self.x_model.to_dict(), "y_model": self.y_model.to_dict(),
                "rmse_x": self.rmse_x, "rmse_y": self.rmse_y, "n_samples": self.n_samples,
                "per_target_error": self.per_target_error, "ok": self.ok,
                "ear_base": self.ear_base, "pitch_lo": self.pitch_lo, "pitch_hi": self.pitch_hi}

    @classmethod
    def from_dict(cls, d: dict) -> "GazeCalibration":
        return cls(AxisModel.from_dict(d["x_model"]), AxisModel.from_dict(d["y_model"]),
                   d["rmse_x"], d["rmse_y"], d["n_samples"], d.get("per_target_error", {}),
                   d.get("ear_base"), d.get("pitch_lo"), d.get("pitch_hi"))


class CalibrationCollector:
    """Accumulates feature samples while the candidate fixates each target."""

    SETTLE_S = 0.5          # drop the saccade at the start of each target
    MIN_SAMPLES_PER_TARGET = 5

    def __init__(self, lam: float = 0.05):
        self.lam = lam
        self.samples: dict[str, list[dict]] = {k: [] for k in TARGETS}
        self._current: Optional[str] = None
        self._t0: Optional[float] = None

    def start_target(self, name: str, t: float) -> None:
        if name not in TARGETS:
            raise ValueError(f"unknown calibration target {name!r}")
        self._current, self._t0 = name, t

    def end_target(self) -> None:
        self._current, self._t0 = None, None

    def add(self, t: float, row: dict) -> None:
        if self._current is None or t - self._t0 < self.SETTLE_S:
            return
        if row.get("face_count") != 1 or row.get("blink"):
            return
        if any(row.get(f) is None for f in X_FEATURES + Y_FEATURES):
            return
        self.samples[self._current].append(row)

    def missing_targets(self) -> list[str]:
        return [k for k, v in self.samples.items() if len(v) < self.MIN_SAMPLES_PER_TARGET]

    def fit(self) -> GazeCalibration:
        missing = self.missing_targets()
        if missing:
            raise ValueError(f"not enough calibration samples for: {', '.join(missing)}")
        rows, tx, ty, names = [], [], [], []
        for name, samples in self.samples.items():
            for r in samples:
                rows.append(r)
                tx.append(TARGETS[name][0])
                ty.append(TARGETS[name][1])
                names.append(name)
        Xx = np.array([[r[f] for f in X_FEATURES] for r in rows], float)
        Xy = np.array([[r[f] for f in Y_FEATURES] for r in rows], float)
        tx, ty = np.array(tx), np.array(ty)
        mx = _fit_ridge(Xx, tx, X_FEATURES, self.lam)
        my = _fit_ridge(Xy, ty, Y_FEATURES, self.lam)
        px = np.array([mx.predict(r) for r in rows])
        py = np.array([my.predict(r) for r in rows])
        per_target = {}
        names_arr = np.array(names)
        for name in TARGETS:
            m = names_arr == name
            per_target[name] = round(float(np.hypot(px[m] - tx[m], py[m] - ty[m]).mean()), 3)
        ears = [r["ear_s"] for r in rows if r.get("ear_s") is not None]
        target_pitch = [float(np.median([r["pitch_s"] for r in self.samples[n]])) for n in TARGETS]
        return GazeCalibration(mx, my,
                               rmse_x=float(np.sqrt(np.mean((px - tx) ** 2))),
                               rmse_y=float(np.sqrt(np.mean((py - ty) ** 2))),
                               n_samples=len(rows), per_target_error=per_target,
                               ear_base=float(np.median(ears)) if ears else None,
                               pitch_lo=min(target_pitch), pitch_hi=max(target_pitch))


# =============================================================================
# Baseline calibration (used by the live service)
# =============================================================================
# The 5-point regression above is kept for offline experiments, but on real
# laptop webcams it proved fragile: people whose open-eye aspect ratio sits
# below the fixed 0.20 "blink" threshold lost every sample and calibration
# failed, silently dropping the session to presence-only mode.
#
# The live service instead records a neutral baseline while the candidate looks
# at a centre dot for ~3 s (medians, so blinks don't matter), then scores every
# frame as a deviation from that baseline in head-pose degrees and iris offset.
# Output uses the same normalised convention as GazeCalibration.predict:
# (x, y) with +x = candidate's right, +y = down, and |value| >= ~1.25 = off-screen.


@dataclass
class BaselineCalibration:
    yaw0: float
    pitch0: float
    iris_h0: float
    iris_v0: float
    ear0: float
    n_samples: int
    # Deviation that maps to 1.0. The detector's enter threshold is 1.25, so the
    # effective triggers are 1.25x these: ~20 deg head turn, ~15 deg tilt,
    # ~0.10 / ~0.07 iris shift (eyes-only looks).
    yaw_scale: float = 16.0
    pitch_scale: float = 12.0
    iris_h_scale: float = 0.08
    iris_v_scale: float = 0.055

    MIN_SAMPLES = 8
    ok = True               # a baseline is always usable once it has samples

    @property
    def ear_base(self) -> float:
        return self.ear0

    @property
    def pitch_lo(self) -> float:
        return self.pitch0

    @property
    def pitch_hi(self) -> float:
        return self.pitch0

    def predict(self, sample: dict) -> Optional[tuple[float, float]]:
        yaw, pitch = sample.get("yaw_s"), sample.get("pitch_s")
        ih, iv = sample.get("iris_h_s"), sample.get("iris_v_s")
        if yaw is None or pitch is None:
            return None
        # +yaw = face turned toward image-right = the candidate's own left.
        x_head = -(yaw - self.yaw0) / self.yaw_scale
        x_eye = -(ih - self.iris_h0) / self.iris_h_scale if ih is not None else 0.0
        # +pitch = up, so a drop in pitch = looking down (+y).
        y_head = (self.pitch0 - pitch) / self.pitch_scale
        y_eye = (iv - self.iris_v0) / self.iris_v_scale if iv is not None else 0.0
        x = x_head if abs(x_head) >= abs(x_eye) else x_eye
        y = y_head if abs(y_head) >= abs(y_eye) else y_eye
        return float(x), float(y)

    def blink_threshold(self) -> float:
        return 0.6 * self.ear0

    def to_dict(self) -> dict:
        return {"type": "baseline", "yaw0": self.yaw0, "pitch0": self.pitch0, "iris_h0": self.iris_h0,
                "iris_v0": self.iris_v0, "ear0": self.ear0, "n_samples": self.n_samples, "ok": True}


class BaselineCollector:
    """Collects neutral-pose samples while the candidate looks at the centre dot.

    Accepts the same control calls as CalibrationCollector. Only frames taken
    while the 'center' target is shown are used; blinks are removed afterwards
    relative to the person's own median eye openness, not a fixed threshold.
    """

    SETTLE_S = 0.4

    def __init__(self):
        self.rows: list[dict] = []
        self._current: Optional[str] = None
        self._t0: Optional[float] = None

    def start_target(self, name: str, t: float) -> None:
        if name not in TARGETS:
            raise ValueError(f"unknown calibration target {name!r}")
        self._current, self._t0 = name, t

    def end_target(self) -> None:
        self._current, self._t0 = None, None

    def add(self, t: float, row: dict) -> None:
        if self._current != "center" or t - self._t0 < self.SETTLE_S:
            return
        if row.get("face_count") != 1 or row.get("yaw_s") is None or row.get("ear") is None:
            return
        self.rows.append(row)

    def missing_targets(self) -> list[str]:
        return [] if len(self.rows) >= BaselineCalibration.MIN_SAMPLES else ["center"]

    def fit(self) -> BaselineCalibration:
        if len(self.rows) < BaselineCalibration.MIN_SAMPLES:
            raise ValueError("face not visible long enough while looking at the centre dot")
        ear0 = float(np.median([r["ear"] for r in self.rows]))
        steady = [r for r in self.rows if r["ear"] >= 0.6 * ear0] or self.rows   # drop blinks
        def med(k: str, default: float) -> float:
            vals = [r[k] for r in steady if r.get(k) is not None]
            return float(np.median(vals)) if vals else default
        # Raw per-frame iris values: the smoothed ones may still be frozen by the
        # generic blink threshold before this candidate's own threshold exists.
        return BaselineCalibration(yaw0=med("yaw_s", 0.0), pitch0=med("pitch_s", 0.0),
                                   iris_h0=med("iris_h", 0.5), iris_v0=med("iris_v", 0.0),
                                   ear0=ear0, n_samples=len(steady))
