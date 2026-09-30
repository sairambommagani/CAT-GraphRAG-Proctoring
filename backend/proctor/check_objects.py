"""Live viewer for the object / hand detectors - see what they recognise and how sure they are.

    python -m proctor.check_objects            # webcam 0, EfficientDet-Lite2
    python -m proctor.check_objects lite0      # compare with the small model

Red boxes   = prohibited items that would count (score above that class's threshold)
Yellow      = detected but below threshold (would NOT count)
Grey        = other COCO classes (person, chair, ...)
Cyan        = hands (magenta = counted as another person's hand)
Press q to quit, s to save a snapshot (for tuning).
"""
import sys
import time

import cv2

from .objects import CLASS_THRESHOLDS, PROHIBITED, SceneAnalyzer
from .tracker import FaceTracker


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else None
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    faces_model = FaceTracker()
    scene_model = SceneAnalyzer(every_n=1, model=model)
    print(f"object model: {scene_model.model_name}. Hold a phone / book in view. q = quit, s = snapshot")
    t0 = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("no camera frame")
            break
        ts = int((time.monotonic() - t0) * 1000)
        faces = faces_model(frame, ts)
        scene = scene_model(frame, ts, faces)
        vis = frame.copy()
        for b in scene.raw:
            if b.label in PROHIBITED:
                color = (0, 0, 255) if b.score >= CLASS_THRESHOLDS.get(b.label, 0.35) else (0, 220, 255)
            else:
                color = (150, 150, 150)
            p0, p1 = (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h))
            cv2.rectangle(vis, p0, p1, color, 2)
            cv2.putText(vis, f"{b.label} {b.score:.2f}", (p0[0], max(15, p0[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        for i, h in enumerate(scene.hands):
            color = (255, 0, 255) if i < scene.foreign_hands else (255, 255, 0)
            cv2.rectangle(vis, (int(h.x), int(h.y)), (int(h.x + h.w), int(h.y + h.h)), color, 2)
        cv2.putText(vis, f"{scene_model.model_name}  faces={len(faces)} persons={scene.persons} "
                         f"hands={len(scene.hands)} foreign={scene.foreign_hands}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("proctor - object detector check (q quits)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(name, vis)
            print("saved", name)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
