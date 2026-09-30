"""Phone / notes / extra person / foreign hand detection (regressions from the
2026-09-25 16:18 recording: a phone held up and a hand signalling from the
side were not detected)."""
import asyncio

import cv2
import numpy as np
import pytest

from proctor.judge import MockJudge
from proctor.objects import Box, Scene, count_foreign_hands, face_box
from proctor.retention import EvidenceStore
from proctor.service import ProctorSession
from synth import ScriptedLandmarker, looking_at

FPS = 8
JPEG = cv2.imencode(".jpg", np.full((360, 480, 3), 90, np.uint8))[1].tobytes()
PHONE = Box(300, 120, 60, 110, 0.7, "cell phone")
BOOK = Box(40, 280, 120, 70, 0.6, "book")


class FakeScene:
    """Stands in for SceneAnalyzer; runs every other frame like the real one."""

    def __init__(self):
        self.next = Scene()
        self.n = 0

    def __call__(self, frame, ts, faces):
        self.n += 1
        if self.n % 2:
            return None
        return self.next


class H:
    def __init__(self, tmp_path):
        self.scene = FakeScene()
        self.faces = lambda: [looking_at(0, 0)]
        self.s = ProctorSession("s", ScriptedLandmarker(lambda t: self.faces()), MockJudge(),
                                EvidenceStore(tmp_path / "st"), scene_analyzer=self.scene)
        self.t, self.events, self.alerts, self.statuses = 0.0, [], [], []

    async def feed(self, seconds, scene=None, faces=None):
        if scene is not None:
            self.scene.next = scene
        if faces is not None:
            self.faces = faces
        for _ in range(int(round(seconds * FPS))):
            status, ev, ready = self.s.ingest(self.t, JPEG)
            self.events += ev
            if status:
                self.statuses.append(status)
            for e in ready:
                await self.s.process_evidence(e, self._n)
            self.t += 1 / FPS

    async def _n(self, m):
        if m["type"] == "alert":
            self.alerts.append(m)

    async def start(self):
        self.s.calib_target("center")
        await self.feed(3)
        self.s.calib_target_end()
        assert self.s.calib_finish()["ok"]


def run(c):
    asyncio.run(c)


def test_phone_in_view_flags_and_pops_up(tmp_path):
    h = H(tmp_path)
    async def go():
        await h.start()
        await h.feed(2)
        await h.feed(2, Scene(objects=[PHONE]))
        await h.feed(5, Scene())
    run(go())
    assert [(e.type, e.direction) for e in h.events] == [("PROHIBITED_OBJECT", "phone")]
    assert h.alerts and "phone" in h.alerts[0]["message"]
    assert any(st["objects"] == ["phone"] for st in h.statuses)          # badge shows it live
    ev = h.s.store.list_evidence("s")[0]
    assert any(line.startswith("phone visible") for line in ev["timeline"])


def test_phone_covering_face_is_still_flagged(tmp_path):
    """In the recording the phone hid the face: 'Face not visible' instead of 'phone'."""
    h = H(tmp_path)
    async def go():
        await h.start()
        await h.feed(1.5, Scene(objects=[PHONE]), faces=lambda: [])
        await h.feed(5, Scene(), faces=lambda: [looking_at(0, 0)])
    run(go())
    assert "PROHIBITED_OBJECT" in [e.type for e in h.events] and h.alerts


def test_single_frame_flicker_ignored(tmp_path):
    h = H(tmp_path)
    async def go():
        await h.start()
        for _ in range(6):                         # phone "seen" for one analysed frame every 2 s
            await h.feed(0.25, Scene(objects=[PHONE]))
            await h.feed(1.75, Scene())
    run(go())
    assert h.events == []


def test_notes_book_flagged(tmp_path):
    h = H(tmp_path)
    face = face_box(looking_at(0, 0))
    desk_book = Box(face.x, face.y + face.h * 1.6, 120, 60, 0.6, "book")
    async def go():
        await h.start()
        await h.feed(2, Scene(objects=[desk_book], face=face))
        await h.feed(5, Scene())
    run(go())
    assert [(e.type, e.direction) for e in h.events] == [("PROHIBITED_OBJECT", "book")]


def test_extra_person_without_visible_face(tmp_path):
    h = H(tmp_path)
    async def go():
        await h.start()
        await h.feed(2, Scene(persons=2))
        await h.feed(5, Scene(persons=1))
    run(go())
    assert [e.type for e in h.events] == ["EXTRA_PERSON"] and h.alerts


def test_foreign_hand_signalling_from_the_side(tmp_path):
    h = H(tmp_path)
    face = face_box(looking_at(0, 0))
    side_hand = Box(face.x - 3.2 * face.w, face.y - 10, 50, 60, 1.0, "hand")
    async def go():
        await h.start()
        await h.feed(2, Scene(hands=[side_hand], foreign_hands=count_foreign_hands([side_hand], face, 480)))
        await h.feed(5, Scene())
    run(go())
    assert [e.type for e in h.events] == ["FOREIGN_HAND"] and h.alerts


# ---- the foreign-hand rule itself -------------------------------------------

def _face():
    return Box(200, 100, 80, 90)


def test_own_hand_near_face_is_not_foreign():
    chin_rest = Box(215, 170, 50, 60)
    scratching_head = Box(260, 60, 50, 50)
    assert count_foreign_hands([chin_rest, scratching_head], _face(), 480) == 0


def test_own_hand_low_in_frame_is_not_foreign():
    on_desk = Box(20, 300, 60, 50)                   # far to the side but low (desk / mouse)
    assert count_foreign_hands([on_desk], _face(), 480) == 0


def test_hand_at_head_height_far_to_the_side_is_foreign():
    assert count_foreign_hands([Box(10, 90, 50, 60)], _face(), 480) == 1


def test_more_than_two_hands_is_foreign():
    near = [Box(215, 170, 40, 40), Box(250, 170, 40, 40), Box(230, 200, 40, 40)]
    assert count_foreign_hands(near, _face(), 480) == 1


def test_scene_prohibited_labels():
    s = Scene(objects=[PHONE, BOOK, Box(0, 0, 1, 1, 0.9, "cell phone")])
    assert s.prohibited == ["book", "phone"]


# ---- background filtering (regression: bookshelf flagged as "book", 2026-09-25 17:36) ----

from proctor.objects import relevant_objects

FACE = Box(200, 100, 80, 90)
SHELF_BOOKS = Box(380, 40, 90, 120, 0.5, "book")          # behind the candidate, head height


def test_bookshelf_behind_candidate_ignored():
    assert relevant_objects([SHELF_BOOKS], [], FACE, background=[]) == []
    assert relevant_objects([SHELF_BOOKS], [], None, background=[]) == []   # seat empty


def test_book_on_desk_or_in_hand_kept():
    desk = Box(180, 260, 120, 60, 0.6, "book")
    held = Box(380, 60, 60, 80, 0.6, "book")
    hand = Box(420, 120, 40, 40)
    assert relevant_objects([desk], [], FACE, []) == [desk]
    assert relevant_objects([held], [hand], FACE, []) == [held]


def test_objects_seen_during_calibration_are_background():
    tv = Box(10, 10, 100, 60, 0.6, "tv")
    moved = Box(14, 12, 100, 60, 0.6, "tv")                 # same TV, detector jitter
    phone = Box(300, 120, 60, 110, 0.7, "cell phone")
    assert relevant_objects([moved, phone], [], FACE, background=[tv]) == [phone]


def test_session_learns_room_during_calibration(tmp_path):
    """A shelf 'book' visible at calibration never raises an event later, even when
    the candidate leans away; a phone still does."""
    h = H(tmp_path)
    face = face_box(looking_at(0, 0))
    shelf = Box(face.x + 3 * face.w, face.y + face.h * 1.5, 90, 120, 0.5, "book")   # low shelf: would pass the desk rule
    async def go():
        h.scene.next = Scene(objects=[shelf], face=face)
        await h.start()
        await h.feed(3, Scene(objects=[shelf], face=face))
        await h.feed(2, Scene(objects=[shelf, PHONE], face=face))
        await h.feed(5, Scene(objects=[shelf], face=face))
    run(go())
    assert [(e.type, e.direction) for e in h.events] == [("PROHIBITED_OBJECT", "phone")]


def test_flickering_phone_is_still_flagged(tmp_path):
    """2026-09-28 recording: the phone was detected only intermittently (score flickering
    around the threshold) and the old 'continuous for 0.8 s' rule never fired."""
    h = H(tmp_path)
    async def go():
        await h.start()
        await h.feed(1, Scene())
        for _ in range(5):                        # seen in 1 of every 3 analysed frames
            await h.feed(0.25, Scene(objects=[PHONE]))
            await h.feed(0.5, Scene())
        await h.feed(4, Scene())
    run(go())
    assert [(e.type, e.direction) for e in h.events] == [("PROHIBITED_OBJECT", "phone")]
    assert h.alerts


def test_phone_seen_as_remote_counts_as_phone():
    s = Scene(objects=[Box(300, 120, 60, 110, 0.3, "remote")])
    assert s.prohibited == ["phone"]
