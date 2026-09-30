"""Scene analysis beyond the face: prohibited objects, extra people, foreign hands.

Gaze tracking can't see a phone held in front of the camera, notes on the
desk, or someone signalling from the side. Two MediaPipe models cover that:

* ObjectDetector (EfficientDet-Lite0, COCO classes): 'cell phone', 'book',
  'laptop', 'person' boxes with scores.
* HandLandmarker: hand boxes; a hand counts as *foreign* when it is far to
  the side of the candidate's face at head height (a candidate using a mouse
  doesn't hold a hand up beside their head), or when there are 3+ hands.

Both run every `every_n` frames to keep CPU cost low; the detector applies
persistence so single-frame false detections don't raise events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .features import POSE_IDX
from .tracker import ensure_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OBJECT_MODELS = {
    # Lite0 is fastest but misses many phones (a lit screen facing the camera, in the
    # 2026-09-25 17:51 recording). Lite2 is markedly more accurate and still real-time
    # on a laptop CPU at the 4 fps this analyser runs. Override with PROCTOR_OBJECT_MODEL.
    "lite0": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
    "lite2": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite",
}
# Per-class minimum scores. Phones score low when the screen is lit or seen edge-on;
# persistence in the detector (>= 0.8 s) filters one-frame mistakes.
# Low on purpose: window voting in the detector filters flicker, and the AI judge verifies.
CLASS_THRESHOLDS = {"cell phone": 0.20, "remote": 0.25, "book": 0.30, "laptop": 0.40, "tv": 0.45}
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")

# COCO often labels a phone seen edge-on or from the back as "remote": report it as a phone.
PROHIBITED = {"cell phone": "phone", "book": "book", "laptop": "second device",
              "remote": "phone", "tv": "second screen"}


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0
    label: str = ""

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class Scene:
    objects: list[Box] = field(default_factory=list)      # prohibited items only
    persons: int = 0
    hands: list[Box] = field(default_factory=list)
    foreign_hands: int = 0
    face: Optional[Box] = None                           # candidate's face box, if visible
    raw: list[Box] = field(default_factory=list)         # every detection (debug viewer)

    @property
    def prohibited(self) -> list[str]:
        return sorted({PROHIBITED[b.label] for b in self.objects})

    def summary(self) -> dict:
        return {"objects": self.prohibited, "persons": self.persons,
                "hands": len(self.hands), "foreign_hands": self.foreign_hands}


def face_box(face: np.ndarray) -> Box:
    p = face[list(POSE_IDX), :2]
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return Box(float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def count_foreign_hands(hands: list[Box], face: Optional[Box], frame_w: int) -> int:
    """Hands that are unlikely to be the candidate's own.

    Rule of thumb for a mouse-operated exam: the candidate's hands are either
    near the face (resting chin, scratching) or low in the frame. A hand at
    head height whose centre is more than ~1.8 face-widths to the side, or any
    hand beyond the second, is treated as foreign; the AI judge then looks at
    the frames and decides.
    """
    if not hands:
        return 0
    extra = max(0, len(hands) - 2)
    if face is None or face.w <= 0:
        return extra
    fw = max(face.w, 0.08 * frame_w)
    side = 0
    for h in hands:
        far = abs(h.cx - face.cx) > 1.8 * fw
        head_height = h.cy < face.y + face.h * 1.2
        if far and head_height:
            side += 1
    return max(extra, side)


def iou(a: Box, b: Box) -> float:
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _touches(a: Box, b: Box, pad: float = 0.0) -> bool:
    return not (a.x + a.w < b.x - pad or b.x + b.w < a.x - pad or
                a.y + a.h < b.y - pad or b.y + b.h < a.y - pad)


def relevant_objects(objects: list[Box], hands: list[Box], face: Optional[Box],
                     background: list[Box], bg_iou: float = 0.3) -> list[Box]:
    """Keep only prohibited items that plausibly belong to the candidate's activity.

    * Anything already in view during calibration (shelves, posters, a TV on the
      wall) is background: dropped if it overlaps a background box (IoU >= bg_iou).
    * A 'book' only counts when it is held (touches a hand) or lies below the
      candidate's chin (desk area). Books on a shelf behind the head don't count;
      with no face visible, only a held book counts (absence is its own event).
    * Phones and devices count anywhere they aren't background - a phone held
      up next to the face is the classic case.
    """
    kept = []
    for o in objects:
        if any(iou(o, b) >= bg_iou for b in background):
            continue
        if o.label == "book":
            held = any(_touches(o, h, pad=10) for h in hands)
            on_desk = face is not None and o.cy > face.y + face.h * 1.15
            if not (held or on_desk):
                continue
        kept.append(o)
    return kept


class SceneAnalyzer:
    """Runs the object + hand models on a frame (VIDEO mode, monotonic timestamps)."""

    def __init__(self, object_threshold: float = 0.35, max_hands: int = 4, every_n: int = 2,
                 model: Optional[str] = None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.object_threshold = object_threshold
        self.every_n = every_n
        self._n = 0
        self._last_ts = -1
        import os
        name = (model or os.environ.get("PROCTOR_OBJECT_MODEL", "lite2")).lower()
        if name not in OBJECT_MODELS:
            name = "lite2"
        try:
            obj_path = ensure_model(MODELS_DIR / f"efficientdet_{name}.tflite", OBJECT_MODELS[name])
        except Exception as e:                   # download blocked/unavailable: use the small model
            print(f"[proctor] could not get efficientdet_{name} ({e}); falling back to lite0")
            name = "lite0"
            obj_path = ensure_model(MODELS_DIR / "efficientdet_lite0.tflite", OBJECT_MODELS["lite0"])
        self.model_name = f"efficientdet_{name}"
        hand_path = ensure_model(MODELS_DIR / "hand_landmarker.task", HAND_MODEL_URL)
        self._objects = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(obj_path)),
            running_mode=vision.RunningMode.VIDEO, score_threshold=0.2, max_results=15))
        self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=max_hands,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5))

    def __call__(self, frame_bgr: np.ndarray, timestamp_ms: int, faces: list) -> Optional[Scene]:
        """Returns a Scene every `every_n` frames, otherwise None (not analysed)."""
        self._n += 1
        if self._n % self.every_n:
            return None
        import cv2

        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        h, w = frame_bgr.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        scene = Scene()
        for det in self._objects.detect_for_video(image, ts).detections:
            cat = det.categories[0]
            bb = det.bounding_box
            box = Box(bb.origin_x, bb.origin_y, bb.width, bb.height, cat.score, cat.category_name)
            scene.raw.append(box)
            if cat.category_name == "person" and cat.score >= 0.5:
                scene.persons += 1
            elif cat.category_name in PROHIBITED and \
                    cat.score >= CLASS_THRESHOLDS.get(cat.category_name, self.object_threshold):
                scene.objects.append(box)
        res = self._hands.detect_for_video(image, ts)
        for lm in res.hand_landmarks:
            xs = np.array([p.x for p in lm]) * w
            ys = np.array([p.y for p in lm]) * h
            scene.hands.append(Box(float(xs.min()), float(ys.min()), float(np.ptp(xs)), float(np.ptp(ys)), 1.0, "hand"))
        primary = face_box(max(faces, key=lambda f: face_box(f).w)) if faces else None
        scene.face = primary
        scene.foreign_hands = count_foreign_hands(scene.hands, primary, w)
        return scene

    def close(self) -> None:
        self._objects.close()
        self._hands.close()
