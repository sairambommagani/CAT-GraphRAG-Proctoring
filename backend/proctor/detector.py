"""Temporal anomaly detector: calibrated gaze stream -> proctoring events.

Per-frame gaze is noisy, so the detector never fires on a single frame. It
uses persistence (a condition must hold for a minimum time), hysteresis
(different enter/exit thresholds so the state doesn't flicker), and a
per-type cooldown (one event per behaviour per window, which also caps LLM
judge calls).

Events
------
OFFSCREEN_SUSTAINED  gaze off-screen continuously for >= sustained_s
REPEATED_GLANCES     >= glance_count off-screen glances in the same direction within glance_window_s
NO_FACE              no face for >= no_face_s
MULTIPLE_FACES       2+ faces for >= multi_face_s
PROHIBITED_OBJECT    phone / book / second device visible for >= object_s   (objects.py)
EXTRA_PERSON         2+ people (bodies, even without a visible face) >= person_s
FOREIGN_HAND         a hand that isn't the candidate's (side, head height) >= hand_s
VOICE_DETECTED       speech on the microphone; direction = who (audio.py lip-sync attribution)
"""
from __future__ import annotations

import itertools
from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from typing import Optional

_ids = itertools.count(1)


@dataclass
class DetectorConfig:
    enter_score: float = 1.25    # |gaze| beyond this = off-screen (1.0 = screen edge)
    exit_score: float = 1.10     # must come back inside this to count as on-screen
    min_glance_s: float = 0.35   # shorter excursions are saccade noise
    exit_hold_s: float = 0.30    # must stay back on-screen this long to end a glance
    sustained_s: float = 3.0
    glance_count: int = 3
    glance_window_s: float = 30.0
    no_face_s: float = 3.0
    multi_face_s: float = 1.0
    dropout_tolerance_s: float = 0.4   # brief detection dropouts don't reset timers
    cooldown_s: float = 10.0            # same flag + same direction can't repeat within this
    object_s: float = 0.8
    person_s: float = 1.0
    hand_s: float = 0.8
    scene_tolerance_s: float = 0.6     # scene models run every other frame and flicker


@dataclass
class Event:
    type: str
    t_start: float
    t_trigger: float
    direction: Optional[str] = None
    score: float = 0.0
    details: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"ev{next(_ids)}")

    def to_dict(self) -> dict:
        return asdict(self)


def gaze_direction(x: float, y: float) -> str:
    """Screen-space direction label for an off-screen gaze point."""
    if abs(x) >= abs(y):
        return "right" if x > 0 else "left"
    return "down" if y > 0 else "up"


class _Persist:
    """Tracks how long a boolean condition has held, tolerating short dropouts."""

    def __init__(self, tolerance: float):
        self.tolerance = tolerance
        self.since: Optional[float] = None
        self.last_true: Optional[float] = None
        self.fired = False

    def update(self, t: float, cond: bool) -> float:
        if cond:
            if self.since is None or (self.last_true is not None and t - self.last_true > self.tolerance):
                self.since, self.fired = t, False
            self.last_true = t
            return t - self.since
        if self.last_true is not None and t - self.last_true > self.tolerance:
            self.since, self.last_true, self.fired = None, None, False
        return 0.0 if self.since is None else self.last_true - self.since


class _Vote:
    """Fires when a flickering condition was true in enough recent checks.

    Object/hand detectors flicker (a phone scores 0.3, 0.1, 0.4, 0.2 ... frame to
    frame). "Continuously true for 0.8 s" resets on every miss, so a phone held up
    for 10 s could go undetected. Voting over a window - at least `min_hits` checks
    and `min_ratio` of the checks in the last `window` seconds - is robust to that
    while still ignoring a single-frame false detection.
    """

    def __init__(self, window: float, min_hits: int, min_ratio: float):
        self.window, self.min_hits, self.min_ratio = window, min_hits, min_ratio
        self.checks: deque = deque()          # (t, hit)
        self.fired = False

    def update(self, t: float, hit: bool) -> bool:
        self.checks.append((t, hit))
        while self.checks and t - self.checks[0][0] > self.window:
            self.checks.popleft()
        hits = sum(1 for _, h in self.checks if h)
        if hits == 0:
            self.fired = False                 # cleared: the next appearance can fire again
            return False
        return hits >= self.min_hits and hits / len(self.checks) >= self.min_ratio

    @property
    def first_hit(self) -> Optional[float]:
        return next((tt for tt, h in self.checks if h), None)


class AnomalyDetector:
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()
        self.off_since: Optional[float] = None
        self.off_dir: Optional[str] = None
        self.off_peak = 0.0
        self.off_dirs: Counter = Counter()      # frames per direction in the current excursion
        self.on_since: Optional[float] = None
        self.sustained_fired = False
        self.glances: deque = deque()           # (t_start, t_end, direction)
        self.no_face = _Persist(self.cfg.dropout_tolerance_s)
        self.multi_face = _Persist(self.cfg.dropout_tolerance_s)
        self.last_fired: dict = {}
        # ~4 scene checks a second (every other frame): 2+ hits in the last 1.5 s
        self.obj_vote = _Vote(1.5, 2, 0.3)
        self.person_vote = _Vote(1.5, 3, 0.5)
        self.hand_vote = _Vote(1.5, 2, 0.3)
        self._objects_seen: set = set()
        self.state = "on_screen"

    # -- helpers -----------------------------------------------------------
    def _emit(self, ev: Event, out: list) -> None:
        # Cooldown per (type, direction): a benign sideways look must not
        # hide a later look down at a phone.
        key = (ev.type, ev.direction)
        last = self.last_fired.get(key)
        if last is not None and ev.t_trigger - last < self.cfg.cooldown_s:
            return
        self.last_fired[key] = ev.t_trigger
        out.append(ev)

    def emit_external(self, ev: Event) -> list[Event]:
        """Events from other detectors (audio) share this detector's cooldowns."""
        out: list[Event] = []
        self._emit(ev, out)
        return out

    def _dominant_dir(self) -> Optional[str]:
        return self.off_dirs.most_common(1)[0][0] if self.off_dirs else self.off_dir

    def _end_glance(self, t_end: float, out: list) -> None:
        dur = t_end - self.off_since
        self.off_dir = self._dominant_dir()
        if dur >= self.cfg.min_glance_s:
            self.glances.append((self.off_since, t_end, self.off_dir))
            while self.glances and t_end - self.glances[0][1] > self.cfg.glance_window_s:
                self.glances.popleft()
            same = [g for g in self.glances if g[2] == self.off_dir]
            if len(same) >= self.cfg.glance_count:
                self._emit(Event("REPEATED_GLANCES", same[0][0], t_end, self.off_dir,
                                 score=float(len(same)),
                                 details={"glances": len(same),
                                          "total_off_s": round(sum(g[1] - g[0] for g in same), 2)}), out)
        self.off_since, self.off_dir, self.off_peak = None, None, 0.0
        self.off_dirs = Counter()
        self.sustained_fired = False

    # -- main update --------------------------------------------------------
    def update(self, t: float, face_count: int, gaze: Optional[tuple[float, float]]) -> list[Event]:
        out: list[Event] = []
        cfg = self.cfg

        d = self.no_face.update(t, face_count == 0)
        if d >= cfg.no_face_s and not self.no_face.fired:
            self.no_face.fired = True
            self._emit(Event("NO_FACE", self.no_face.since, t, score=round(d, 2)), out)

        d = self.multi_face.update(t, face_count >= 2)
        if d >= cfg.multi_face_s and not self.multi_face.fired:
            self.multi_face.fired = True
            self._emit(Event("MULTIPLE_FACES", self.multi_face.since, t, score=round(d, 2),
                             details={"faces": face_count}), out)

        if face_count == 0 or gaze is None:
            # Face gone: close any open glance; NO_FACE covers this period.
            if self.off_since is not None:
                self._end_glance(t, out)
            self.state = "no_face" if face_count == 0 else self.state
            return out

        x, y = gaze
        score = max(abs(x), abs(y))
        if self.off_since is None:
            if score > cfg.enter_score:
                self.off_since, self.off_dir, self.off_peak = t, gaze_direction(x, y), score
                self.off_dirs = Counter({self.off_dir: 1})
                self.on_since = None
            self.state = "on_screen" if self.off_since is None else "off_screen"
        else:
            self.off_peak = max(self.off_peak, score)
            if score < cfg.exit_score:
                if self.on_since is None:
                    self.on_since = t
                if t - self.on_since >= cfg.exit_hold_s:
                    self._end_glance(self.on_since, out)
                    self.state = "on_screen"
            else:
                self.on_since = None
                self.off_dirs[gaze_direction(x, y)] += 1
                dur = t - self.off_since
                if dur >= cfg.sustained_s and not self.sustained_fired:
                    self.sustained_fired = True
                    self.off_dir = self._dominant_dir()
                    self._emit(Event("OFFSCREEN_SUSTAINED", self.off_since, t, self.off_dir,
                                     score=round(self.off_peak, 2),
                                     details={"duration_s": round(dur, 2)}), out)
                self.state = "off_screen"
        return out

    # -- scene (objects / people / hands) --------------------------------------
    def update_scene(self, t: float, scene) -> list[Event]:
        """`scene` is an objects.Scene; call only on frames where it was analysed."""
        out: list[Event] = []
        objs = scene.prohibited
        if objs:
            self._objects_seen.update(objs)
        if self.obj_vote.update(t, bool(objs)) and not self.obj_vote.fired:
            self.obj_vote.fired = True
            what = ", ".join(sorted(self._objects_seen))
            start = self.obj_vote.first_hit or t
            self._emit(Event("PROHIBITED_OBJECT", start, t, what, score=round(t - start, 2),
                             details={"objects": what}), out)
        if not any(h for _, h in self.obj_vote.checks):
            self._objects_seen = set()

        if self.person_vote.update(t, scene.persons >= 2) and not self.person_vote.fired:
            self.person_vote.fired = True
            start = self.person_vote.first_hit or t
            self._emit(Event("EXTRA_PERSON", start, t, score=round(t - start, 2),
                             details={"persons": scene.persons}), out)

        if self.hand_vote.update(t, scene.foreign_hands >= 1) and not self.hand_vote.fired:
            self.hand_vote.fired = True
            start = self.hand_vote.first_hit or t
            self._emit(Event("FOREIGN_HAND", start, t, score=round(t - start, 2),
                             details={"hands_in_view": len(scene.hands)}), out)
        return out
