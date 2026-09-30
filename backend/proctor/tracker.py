"""Thin wrapper around the MediaPipe Tasks FaceLandmarker (478 landmarks incl. iris)."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def ensure_model(path: Path = DEFAULT_MODEL, url: str = MODEL_URL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model {path.name} -> {path}")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)                      # no half-written model after an interrupted download
    return path


class FaceTracker:
    """Returns a list of (478, 3) landmark arrays in pixel coordinates per frame.

    num_faces=2 so a second person in view is detected (a proctoring signal).
    Runs in VIDEO mode, which uses temporal tracking between frames and is
    faster/smoother than per-image detection.
    """

    def __init__(self, model_path: str | os.PathLike | None = None, num_faces: int = 2):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        path = ensure_model(Path(model_path) if model_path else DEFAULT_MODEL)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def __call__(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[np.ndarray]:
        import cv2

        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        faces = []
        for face in result.face_landmarks:
            # x, y are normalised to [0,1]; z is roughly in the same scale as x.
            arr = np.array([[p.x * w, p.y * h, p.z * w] for p in face], dtype=np.float64)
            faces.append(arr)
        return faces

    def close(self) -> None:
        self._landmarker.close()
