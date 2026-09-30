"""Replay + evaluation harness: precision / recall on a labelled folder of recordings."""
import numpy as np
import cv2
import pytest

import proctor.replay as replay_mod
from proctor.evaluate import Counts, evaluate, main as eval_main, to_markdown
from synth import ScriptedLandmarker, looking_at, room_noise, voice

FPS = 15


def write_video(path, seconds=14):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    for i in range(seconds * FPS):
        vw.write(np.full((240, 320, 3), 80 + i % 50, np.uint8))
    vw.release()


@pytest.fixture
def fake_media(monkeypatch):
    """Recordings whose name contains 'voice' have another person talking at 7-10 s;
    'phonedown' clips have the candidate staring down (at a phone) at 7-11 s."""
    rng = np.random.default_rng(0)
    current = {"name": ""}

    def audio_for(path):
        current["name"] = str(path)
        a = room_noise(14, rng)
        if "voice" in str(path):
            a[7 * 16000:10 * 16000] = voice(3, rng)
        return a

    def face(t):
        if "phonedown" in current["name"] and 7 <= t < 11:
            return [looking_at(0.0, 2.6, rng, 1.0)]
        return [looking_at(0.05, 0.1, rng, 1.0, mouth_open=0.03)]

    monkeypatch.setattr(replay_mod, "load_media_audio", audio_for)
    monkeypatch.setattr(replay_mod, "FaceTracker", lambda: ScriptedLandmarker(face))
    monkeypatch.setattr(replay_mod, "SceneAnalyzer", None)


def test_replay_file_runs_video_and_audio(tmp_path, fake_media):
    p = tmp_path / "voice_01.mp4"
    write_video(p)
    res = replay_mod.replay_file(str(p), object_detection=False, verbose=False)
    assert res.calibrated and res.has_audio
    assert [f["type"] for f in res.flags] == ["VOICE_DETECTED"] and res.flags[0]["direction"] == "another person"
    assert len(res.alerts) == 1 and res.judged[0]["verdict"]["verdict"] == "fraud"
    assert any(r["speech"] for r in res.seconds.values())


def test_evaluate_reports_precision_recall(tmp_path, fake_media):
    for name in ("voice_01.mp4", "phonedown_01.mp4", "clean_01.mp4", "clean_02.mp4"):
        write_video(tmp_path / name)
    (tmp_path / "labels.csv").write_text(
        "file,label,expect,notes\n"
        "voice_01.mp4,cheat,VOICE_DETECTED,friend whispering\n"
        "phonedown_01.mp4,cheat,OFFSCREEN_SUSTAINED,reading phone in lap\n"
        "clean_01.mp4,clean,,quiet\n"
        "clean_02.mp4,clean,,quiet\n")
    rep = evaluate(tmp_path, replay=lambda *a, **k: replay_mod.replay_file(*a, object_detection=False, **k))
    assert rep["system"]["precision"] == 1.0 and rep["system"]["recall"] == 1.0
    assert rep["system"]["false_alarm_rate"] == 0.0 and rep["detector"]["tp"] == 2
    assert rep["per_flag_type_recall"]["VOICE_DETECTED"]["recall"] == 1.0
    assert rep["per_flag_type_recall"]["OFFSCREEN_SUSTAINED"]["recall"] == 1.0
    md = to_markdown(rep)
    assert "| System (popup after AI judge) | 100% | 100% |" in md and "friend whispering" in md


def test_counts_metrics():
    c = Counts()
    for pred, act in [(1, 1), (1, 1), (1, 0), (0, 1), (0, 0), (0, 0), (0, 0)]:
        c.add(bool(pred), bool(act))
    m = c.metrics()
    assert m["precision"] == pytest.approx(0.667, abs=1e-3) and m["recall"] == pytest.approx(0.667, abs=1e-3)
    assert m["false_alarm_rate"] == 0.25 and m["accuracy"] == pytest.approx(5 / 7, abs=1e-3)


def test_bad_labels_are_rejected(tmp_path):
    (tmp_path / "labels.csv").write_text("file,label\nx.mp4,maybe\n")
    with pytest.raises(SystemExit):
        eval_main([str(tmp_path)])
