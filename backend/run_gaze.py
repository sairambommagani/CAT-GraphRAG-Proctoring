"""Phase 1 CLI: live webcam (or video file) gaze tracking with overlay + CSV log.

Examples
    python run_gaze.py                          # webcam 0, live window
    python run_gaze.py --csv logs/me.csv --out logs/me.mp4
    python run_gaze.py --source clip.mp4 --no-show --csv logs/clip.csv

Labelling hotkeys (in the video window) tag each frame with the behaviour you
are acting out, so recordings become labelled data for later phases:
    0 normal   1 phone   2 second-screen   3 notes-down
    4 thinking-up   5 talking/other-person   6 away   (q quits)
"""
import argparse
import json
import os

from proctor.pipeline import camera_frames, run, summarize

LABELS = {ord("0"): "normal", ord("1"): "phone", ord("2"): "second_screen",
          ord("3"): "notes_down", ord("4"): "thinking_up", ord("5"): "other_person",
          ord("6"): "away"}


class HotkeyLabel:
    def __init__(self):
        self.current = "normal"

    def __call__(self) -> str:
        return self.current

    def on_key(self, key: int) -> None:
        if key in LABELS and LABELS[key] != self.current:
            self.current = LABELS[key]
            print(f"label -> {self.current}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index or video file path")
    ap.add_argument("--csv", help="write per-frame features to this CSV")
    ap.add_argument("--out", help="write annotated video (mp4) here")
    ap.add_argument("--no-show", action="store_true", help="don't open a preview window")
    ap.add_argument("--seconds", type=float, help="stop after N seconds")
    ap.add_argument("--model", help="path to face_landmarker.task (auto-downloaded if absent)")
    args = ap.parse_args()

    for p in (args.csv, args.out):
        if p and os.path.dirname(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)

    from proctor.tracker import FaceTracker
    tracker = FaceTracker(args.model)
    try:
        history = run(camera_frames(args.source, args.seconds), tracker,
                      csv_path=args.csv, video_out=args.out, show=not args.no_show,
                      label_fn=HotkeyLabel())
    finally:
        tracker.close()
    print(json.dumps(summarize(history), indent=2))


if __name__ == "__main__":
    main()
