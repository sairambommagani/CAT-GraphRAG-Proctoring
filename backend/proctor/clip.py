"""RAM-only frame ring buffer and evidence-clip builder.

Retention by design: raw frames exist only in this in-memory buffer, which
holds the last `seconds` of video and drops older frames as new ones arrive.
Nothing reaches disk unless the detector flags an event, and then only the
trimmed window around that event does.
"""
from __future__ import annotations

import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detector import Event


@dataclass
class BufferedFrame:
    t: float
    jpeg: bytes
    row: dict                # features + calibrated gaze for this frame


class FrameRingBuffer:
    def __init__(self, seconds: float = 25.0):
        self.seconds = seconds
        self._frames: deque[BufferedFrame] = deque()

    def add(self, frame: BufferedFrame) -> None:
        self._frames.append(frame)
        while self._frames and frame.t - self._frames[0].t > self.seconds:
            self._frames.popleft()

    def window(self, t0: float, t1: float) -> list[BufferedFrame]:
        return [f for f in self._frames if t0 <= f.t <= t1]

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def latest_t(self) -> Optional[float]:
        return self._frames[-1].t if self._frames else None


@dataclass
class Evidence:
    event: Event
    frames: list[BufferedFrame]          # selected key frames, downscaled JPEGs (stored as evidence)
    t0: float                            # clip start (session seconds)
    t1: float
    timeline: list[str] = field(default_factory=list)
    video_frames: list[BufferedFrame] = field(default_factory=list)   # denser sample for video judges
    audio: Optional[np.ndarray] = None   # 16 kHz int16 PCM of [t0, t1] (None: no mic)
    speech_s: float = 0.0                # voiced seconds inside the clip
    transcript: dict = field(default_factory=dict)                    # filled by the transcriber
    knowledge: dict = field(default_factory=dict)                     # rules / past cases (GraphRAG)

    @property
    def has_speech(self) -> bool:
        return self.audio is not None and self.speech_s >= 0.5

    @property
    def rel_times(self) -> list[float]:
        return [round(f.t - self.t0, 1) for f in self.frames]


# Speech needs a longer post-roll than a glance: the flag fires ~1.2 s into the talking, and the
# judge and the transcript need the words that follow.
POST_S_BY_TYPE = {"VOICE_DETECTED": 4.0}


class ClipBuilder:
    """Waits `post_s` after an event, then cuts [t_start - pre_s, t_trigger + post_s]."""

    def __init__(self, buffer: FrameRingBuffer, pre_s: float = 3.0, post_s: float = 1.5,
                 max_len_s: float = 20.0, n_frames: int = 8, max_side: int = 560,
                 audio=None, speech=None, video_frames: int = 16, video_side: int = 448):
        self.buffer = buffer
        self.audio, self.speech = audio, speech          # audio.AudioRingBuffer / audio.SpeechMonitor
        self.video_n, self.video_side = video_frames, video_side
        self.pre_s, self.post_s, self.max_len_s = pre_s, post_s, max_len_s
        self.n_frames, self.max_side = n_frames, max_side
        self.pending: list[Event] = []

    def add_event(self, ev: Event) -> None:
        self.pending.append(ev)

    def post_for(self, ev: Event) -> float:
        return max(self.post_s, POST_S_BY_TYPE.get(ev.type, 0.0))

    def poll(self, now: float) -> list[Evidence]:
        ready = [e for e in self.pending if now >= e.t_trigger + self.post_for(e)]
        self.pending = [e for e in self.pending if e not in ready]
        return [self.build(e) for e in ready]

    def flush(self) -> list[Evidence]:
        ready, self.pending = self.pending, []
        return [self.build(e) for e in ready]

    def build(self, ev: Event) -> Evidence:
        t1 = ev.t_trigger + self.post_for(ev)
        t0 = max(ev.t_start - self.pre_s, t1 - self.max_len_s)
        window = self.buffer.window(t0, t1)
        chosen = select_frames(window, self.n_frames)
        small = [BufferedFrame(f.t, downscale_jpeg(f.jpeg, self.max_side), f.row) for f in chosen]
        timeline = summarize_timeline(window, t0)
        # Uniform, denser sample for models that watch the clip as a video (Cosmos, Omni):
        # motion between frames is what physical reasoning needs.
        if len(window) > self.video_n:
            idx = np.linspace(0, len(window) - 1, self.video_n).round().astype(int)
            vsel = [window[i] for i in sorted(set(idx.tolist()))]
        else:
            vsel = list(window)
        video = [BufferedFrame(f.t, downscale_jpeg(f.jpeg, self.video_side), f.row) for f in vsel]
        out = Evidence(ev, small, t0, t1, timeline=timeline, video_frames=video)
        if self.audio is not None:
            a_t0, pcm = self.audio.window(t0, t1)
            if len(pcm):
                out.audio = pcm
                if self.speech is not None:
                    out.speech_s = round(self.speech.speech_seconds(t0, t1), 1)
                    speaker = ev.details.get("speaker") if ev.type == "VOICE_DETECTED" else None
                    for a, b in self.speech.segments(t0, t1)[:4]:
                        who = f" ({speaker})" if speaker else ""
                        out.timeline.append(f"voice{who} {a - t0:.1f}-{b - t0:.1f}s")
        return out


def select_frames(window: list[BufferedFrame], k: int) -> list[BufferedFrame]:
    """Pick k frames: half spread over the whole clip for context, half from
    the anomalous frames (off-screen / no face / extra face), so a few
    images cover what actually matters."""
    if len(window) <= k:
        return list(window)
    idx_all = np.linspace(0, len(window) - 1, num=max(k // 2, 1)).round().astype(int).tolist()
    anomalous = [i for i, f in enumerate(window) if _is_anomalous(f.row)]
    picks = set(idx_all)
    if anomalous:
        need = k - len(picks)
        for i in np.linspace(0, len(anomalous) - 1, num=max(need, 1)).round().astype(int):
            picks.add(anomalous[i])
    # top up uniformly if duplicates left us short
    for i in np.linspace(0, len(window) - 1, num=k).round().astype(int):
        if len(picks) >= k:
            break
        picks.add(int(i))
    return [window[i] for i in sorted(picks)[:k]]


def _is_anomalous(row: dict) -> bool:
    if row.get("face_count", 1) != 1 or row.get("objects") or row.get("foreign_hands") or row.get("persons", 0) >= 2:
        return True
    g = row.get("gaze_score")
    return g is not None and g > 1.1


def downscale_jpeg(jpeg: bytes, max_side: int, quality: int = 80) -> bytes:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jpeg
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else jpeg


def summarize_timeline(window: list[BufferedFrame], t0: float, max_segments: int = 8) -> list[str]:
    """Compress per-frame signals into a few segment lines for the judge prompt,
    e.g. 'off-screen right 2.1-5.6s (peak 1.8)'. Segment lines are far cheaper
    in tokens than per-frame numbers."""
    segs: list[list] = []   # [label, start, end, peak]
    for f in window:
        r = f.row
        fc = r.get("face_count", 1)
        if r.get("objects"):
            label, val = f"{'/'.join(r['objects'])} visible", 0.0
        elif r.get("foreign_hands"):
            label, val = "another person's hand", 0.0
        elif r.get("persons", 0) >= 2:
            label, val = f"{r['persons']} people", 0.0
        elif fc == 0:
            label, val = "no face", 0.0
        elif fc >= 2:
            label, val = f"{fc} faces", float(fc)
        elif r.get("gaze_score") is not None and r["gaze_score"] > 1.1:
            label, val = f"off-screen {r.get('gaze_dir', '?')}", r["gaze_score"]
        else:
            label, val = "on-screen", r.get("gaze_score") or 0.0
        t = f.t - t0
        if segs and segs[-1][0] == label and t - segs[-1][2] < 0.6:
            segs[-1][2] = t
            segs[-1][3] = max(segs[-1][3], val)
        else:
            segs.append([label, t, t, val])
    segs = [s for s in segs if s[0] != "on-screen" or s[2] - s[1] >= 0.5]
    out = []
    for label, a, b, peak in segs[:max_segments]:
        extra = f" (peak {peak:.1f})" if label.startswith("off-screen") else ""
        out.append(f"{label} {a:.1f}-{b:.1f}s{extra}")
    return out


def frames_to_mp4(frames: list[BufferedFrame], fps: float = 2.0) -> bytes:
    """Pack selected frames into a small MP4 (for models that take video input)."""
    imgs = [cv2.imdecode(np.frombuffer(f.jpeg, np.uint8), cv2.IMREAD_COLOR) for f in frames]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return b""
    h, w = imgs[0].shape[:2]
    with tempfile.TemporaryDirectory() as d:     # deleted immediately after encoding
        p = Path(d) / "clip.mp4"
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for im in imgs:
            vw.write(cv2.resize(im, (w, h)) if im.shape[:2] != (h, w) else im)
        vw.release()
        return p.read_bytes()


def frames_to_grid(frames: list[BufferedFrame], rel_times: list[float], cols: int = 2,
                   tile_w: int = 560, quality: int = 85) -> bytes:
    """Tile the selected frames into ONE labelled image (time order: left->right,
    top->bottom). Works with models that accept a single image per prompt and
    costs fewer vision tokens than 8 separate images."""
    imgs = [cv2.imdecode(np.frombuffer(f.jpeg, np.uint8), cv2.IMREAD_COLOR) for f in frames]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return b""
    h0, w0 = imgs[0].shape[:2]
    tile_h = int(round(tile_w * h0 / w0))
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), np.uint8)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        tile = cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        label = f"#{k + 1}  t={rel_times[k]:.1f}s" if k < len(rel_times) else f"#{k + 1}"
        cv2.rectangle(tile, (0, 0), (190, 30), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""
