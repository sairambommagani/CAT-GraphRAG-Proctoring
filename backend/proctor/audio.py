"""Microphone proctoring: voice activity, lip-sync attribution, transcription.

Pipeline
--------
browser mic (16 kHz mono PCM, 250 ms chunks over the same WebSocket)
  -> AudioRingBuffer          RAM only, same rolling window as the video frames
  -> VoiceActivity            30 ms frames: WebRTC VAD + an energy gate above the
                              room's own noise floor (measured during calibration)
  -> SpeechMonitor            groups voiced frames into speech episodes and asks the
                              video which mouth is moving:
                                 lips moving with the voice  -> "candidate" talking
                                 face visible, lips still    -> "another person"
                                 face not visible            -> "unattributed"
  -> Event("VOICE_DETECTED", direction=<speaker>)  -> clip -> AI judge (with audio)
  -> Transcriber              faster-whisper on the flagged clip only, so the judge
                              and the examiner can read what was said

Lip-sync attribution is the key accuracy step: a voice alone can be the
candidate reading a question aloud or a TV next door; a voice while the
candidate's lips are still is someone else in the room, which is the case
that matters most. Mouth movement comes from the face landmarks we already
track (features.mouth_aspect_ratio), so it costs nothing extra.

Retention: raw audio lives only in the RAM ring buffer. Only the flagged
clip's audio is kept (encrypted, same TTLs as frames). The transcript is
text in the evidence metadata and is deleted with the frames.
"""
from __future__ import annotations

import io
import os
import struct
import threading
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
AUDIO_MAGIC = b"AUD1"            # WebSocket binary message prefix for audio chunks
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000   # 480 samples

try:                                           # optional C extension (pip install webrtcvad-wheels)
    import webrtcvad as _webrtcvad
except Exception:                              # pragma: no cover - energy VAD fallback
    _webrtcvad = None


def pack_audio(t_ms: float, pcm: np.ndarray) -> bytes:
    """Client wire format (used by tests; the browser builds the same bytes)."""
    return AUDIO_MAGIC + struct.pack("<d", t_ms) + pcm.astype("<i2").tobytes()


def unpack_audio(data: bytes) -> Optional[tuple[float, np.ndarray]]:
    """-> (capture time of the chunk's LAST sample in seconds, int16 samples) or None."""
    if len(data) < 14 or data[:4] != AUDIO_MAGIC:
        return None
    (t_ms,) = struct.unpack("<d", data[4:12])
    pcm = np.frombuffer(data[12:12 + (len(data) - 12) // 2 * 2], dtype="<i2").astype(np.int16)
    return t_ms / 1000.0, pcm


def to_wav(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


def dbfs(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) if len(frame) else 0.0
    return 20 * np.log10(max(rms, 1.0) / 32768.0)


# ---- buffer -----------------------------------------------------------------------

class AudioRingBuffer:
    """Last `seconds` of audio as (t_start, samples) chunks. RAM only."""

    def __init__(self, seconds: float = 25.0):
        self.seconds = seconds
        self._chunks: deque[tuple[float, np.ndarray]] = deque()
        self.lock = threading.Lock()

    def add(self, t_end: float, pcm: np.ndarray) -> float:
        t_start = t_end - len(pcm) / SAMPLE_RATE
        with self.lock:
            if self._chunks:                            # keep chunks contiguous and ordered
                prev_s, prev = self._chunks[-1]
                prev_end = prev_s + len(prev) / SAMPLE_RATE
                if t_start < prev_end:
                    t_start = prev_end
            self._chunks.append((t_start, pcm))
            while self._chunks and t_end - (self._chunks[0][0] + len(self._chunks[0][1]) / SAMPLE_RATE) > self.seconds:
                self._chunks.popleft()
        return t_start

    def window(self, t0: float, t1: float) -> tuple[float, np.ndarray]:
        """Samples covering [t0, t1] (gaps are not filled). -> (actual start time, samples)."""
        parts, start = [], None
        with self.lock:
            for s, pcm in self._chunks:
                e = s + len(pcm) / SAMPLE_RATE
                if e < t0 or s > t1:
                    continue
                a = max(0, int((t0 - s) * SAMPLE_RATE))
                b = min(len(pcm), int((t1 - s) * SAMPLE_RATE))
                if b > a:
                    parts.append(pcm[a:b])
                    start = s + a / SAMPLE_RATE if start is None else start
        if not parts:
            return t0, np.zeros(0, np.int16)
        return start, np.concatenate(parts)

    def clear(self) -> None:
        with self.lock:
            self._chunks.clear()

    @property
    def seconds_held(self) -> float:
        with self.lock:
            return sum(len(p) for _, p in self._chunks) / SAMPLE_RATE


# ---- voice activity -----------------------------------------------------------------

@dataclass
class VADFrame:
    t: float            # frame end time (session seconds)
    speech: bool
    db: float
    pitched: Optional[bool] = None   # periodic like a voice (vowels); None = not measured


def periodicity(frame: np.ndarray, rate: int = SAMPLE_RATE) -> float:
    """Peak normalised autocorrelation at human pitch lags (70-400 Hz). Vowels score
    0.5-0.9; keyboard clicks, fans and other noise stay low."""
    x = frame.astype(np.float64)
    x = x - x.mean()
    e = float(np.dot(x, x))
    if e <= 1e-6:
        return 0.0
    lo, hi = rate // 400, min(rate // 70, len(x) - 40)
    n = 1 << (2 * len(x) - 1).bit_length()
    f = np.fft.rfft(x, n)
    ac = np.fft.irfft(f * np.conj(f), n)[:hi + 1]
    # unbiased: normalise each lag by the energy of the overlapping parts
    c = np.cumsum(x * x)
    k = np.arange(lo, hi + 1)
    denom = np.sqrt(c[len(x) - 1 - k] * (c[-1] - c[k - 1]))
    r = ac[lo:hi + 1] / np.maximum(denom, 1e-9)
    # a real pitch is a peak between the lag limits; low-frequency noise (fans, rumble)
    # just decays from the shortest lag, so a maximum at the edge doesn't count
    i = int(np.argmax(r))
    if i == 0 or i == len(r) - 1:
        inner = r[1:-1]
        peaks = [j for j in range(1, len(r) - 1) if r[j] >= r[j - 1] and r[j] >= r[j + 1]]
        if not peaks or not len(inner):
            return 0.0
        i = max(peaks, key=lambda j: r[j])
    # the peak must stand out above the dip before it (periodic, not just smooth)
    dip = float(np.min(r[:i + 1]))
    return float(r[i]) if r[i] - dip >= 0.2 else 0.0


class VoiceActivity:
    """30 ms frames -> speech yes/no.

    A frame is speech when WebRTC VAD says so AND it is louder than the room's
    noise floor by `margin_db`. The floor is learnt during calibration (the
    candidate is asked to sit quietly) and then tracks slowly toward quieter
    levels only, so a long conversation can't raise it and hide itself.
    Without webrtcvad the energy gate alone decides (slightly more false
    positives on noise; the judge filters those).
    """

    def __init__(self, aggressiveness: int = 2, margin_db: float = 6.0, default_floor_db: float = -55.0):
        self.vad = _webrtcvad.Vad(aggressiveness) if _webrtcvad else None
        self.margin_db = margin_db
        self.pitch_min = 0.45
        self.floor_db = default_floor_db
        self._calib: list[float] = []
        self._rest = np.zeros(0, np.int16)
        self._rest_t0: Optional[float] = None

    @property
    def backend(self) -> str:
        return "webrtcvad+energy" if self.vad else "energy"

    def calibrate_sample(self, db: float) -> None:
        self._calib.append(db)

    def finish_calibration(self) -> float:
        if len(self._calib) >= 10:
            # 30th percentile: robust to a cough or a word during calibration
            self.floor_db = float(np.clip(np.percentile(self._calib, 30), -80.0, -30.0))
        self._calib = []
        return self.floor_db

    def process(self, t_start: float, pcm: np.ndarray, calibrating: bool = False) -> list[VADFrame]:
        if self._rest_t0 is not None and len(self._rest):
            expected = self._rest_t0 + len(self._rest) / SAMPLE_RATE
            if abs(t_start - expected) > 0.05:          # gap/dropout: re-anchor, drop the partial frame
                self._rest = np.zeros(0, np.int16)
        if self._rest_t0 is None or len(self._rest) == 0:
            self._rest_t0 = t_start
        buf = np.concatenate([self._rest, pcm]) if len(self._rest) else pcm
        t0 = self._rest_t0
        out = []
        n = len(buf) // FRAME_LEN
        for i in range(n):
            fr = buf[i * FRAME_LEN:(i + 1) * FRAME_LEN]
            db = dbfs(fr)
            t_end = t0 + (i + 1) * FRAME_LEN / SAMPLE_RATE
            # WebRTC VAD sees EVERY frame, not just the loud ones: it keeps a running model
            # of the room's noise and needs the quiet frames to learn it. Fed only loud
            # frames it misses most normal-volume speech from a laptop mic.
            vad_says = None
            if self.vad is not None:
                try:
                    vad_says = self.vad.is_speech(fr.astype("<i2").tobytes(), SAMPLE_RATE)
                except Exception:
                    vad_says = None
            if calibrating:
                self.calibrate_sample(db)
                out.append(VADFrame(t_end, False, db))
                continue
            loud = db > self.floor_db + self.margin_db
            voiced = loud and (vad_says if vad_says is not None else True)
            pitched = voiced and periodicity(fr) >= self.pitch_min
            if not voiced and db < self.floor_db:
                self.floor_db += 0.02 * (db - self.floor_db)          # drift down only
            out.append(VADFrame(t_end, bool(voiced), db, bool(pitched)))
        self._rest = buf[n * FRAME_LEN:]
        self._rest_t0 = t0 + n * FRAME_LEN / SAMPLE_RATE
        return out


# ---- speech episodes + lip-sync attribution ---------------------------------------------

@dataclass
class SpeechConfig:
    min_speech_s: float = 1.0        # voiced time inside an episode before it counts
    min_ratio: float = 0.3           # voiced share of the episode (speech has pauses)
    gap_s: float = 1.2               # silence that ends an episode
    fill_s: float = 0.3              # pauses between syllables shorter than this count as speech
    min_runs: int = 1                # voiced stretches of >= 60 ms (a lone click/keystroke is 1 frame)
    min_modulation_db: float = 4.0   # loudness must rise and fall like syllables (a fan or hum is flat)
    min_pitched_s: float = 0.3       # voiced (vowel) sound with a pitch; typing and clicks have none
    min_pitched_share: float = 0.2   # speech is mostly vowels; noise only now and then looks pitched
    mouth_std_min: float = 0.025     # lip movement that counts as talking (MAR std)
    mouth_range_min: float = 0.08    # ... or p90 - p10 of MAR
    face_share_min: float = 0.6      # need the face in most frames to call it "another person"


@dataclass
class SpeechEpisode:
    t_start: float
    t_last_voiced: float
    voiced_s: float = 0.0
    total_s: float = 0.0
    peak_db: float = -120.0
    fired: bool = False
    runs: int = 0                    # voiced stretches of >= 2 frames
    pitched_s: float = 0.0           # voiced frames with a voice pitch
    run_len: int = 0                 # length of the current stretch (frames)
    dbs: list = field(default_factory=list)

    @property
    def modulation_db(self) -> float:
        """Syllable-rate loudness swing: p90 - p10 of frame loudness over the episode."""
        if len(self.dbs) < 5:
            return 0.0
        return float(np.percentile(self.dbs, 90) - np.percentile(self.dbs, 10))

    def summary(self) -> dict:
        return {"speech_s": round(self.voiced_s, 1), "episode_s": round(self.total_s, 1), "runs": self.runs, "pitched_s": round(self.pitched_s, 2),
                "modulation_db": round(self.modulation_db, 1), "peak_db": round(self.peak_db, 1)}


@dataclass
class Attribution:
    speaker: str                    # candidate | another person | unattributed
    mouth_std: Optional[float]
    mouth_range: Optional[float]
    face_share: float
    detail: str

    def to_dict(self) -> dict:
        return {"speaker": self.speaker, "mouth_std": None if self.mouth_std is None else round(self.mouth_std, 3),
                "mouth_range": None if self.mouth_range is None else round(self.mouth_range, 3),
                "face_share": round(self.face_share, 2)}


def attribute_speaker(rows: list[dict], cfg: SpeechConfig, mouth_baseline_std: Optional[float] = None) -> Attribution:
    """Decide who is talking from the video rows that overlap a speech episode."""
    if not rows:
        return Attribution("unattributed", None, None, 0.0, "no video during speech")
    faces = [r for r in rows if r.get("face_count") == 1 and r.get("mouth") is not None]
    face_share = len(faces) / len(rows)
    if len(faces) < 3:
        return Attribution("unattributed", None, None, face_share, "face not visible while the voice was heard")
    m = np.array([r["mouth"] for r in faces], float)
    std = float(np.std(m))
    rng = float(np.percentile(m, 90) - np.percentile(m, 10))
    thr_std = max(cfg.mouth_std_min, 3.0 * (mouth_baseline_std or 0.0))
    if std >= thr_std or rng >= cfg.mouth_range_min:
        return Attribution("candidate", std, rng, face_share, "candidate's lips moved with the voice")
    if face_share >= cfg.face_share_min:
        return Attribution("another person", std, rng, face_share,
                           "voice heard while the candidate's lips were still")
    return Attribution("unattributed", std, rng, face_share, "face only partly visible while the voice was heard")


class SpeechMonitor:
    """Voiced frames -> speech episodes -> VOICE_DETECTED events (with speaker)."""

    def __init__(self, config: Optional[SpeechConfig] = None):
        self.cfg = config or SpeechConfig()
        self.episode: Optional[SpeechEpisode] = None
        self.speaking = False                    # live status for the badge
        self.recent: deque[VADFrame] = deque(maxlen=int(30 * 1000 / FRAME_MS))   # 30 s of VAD decisions
        self.mouth_baseline_std: Optional[float] = None   # lip jitter while silent, from calibration
        self.last_rejected: Optional[dict] = None          # last sound that ended without a flag (for the log)

    def update(self, frames: list[VADFrame], rows_between) -> list:
        """`rows_between(t0, t1)` returns video rows (dicts) in that window.
        Returns detector Events (not yet cooldown-filtered)."""
        from .detector import Event
        out = []
        dt = FRAME_MS / 1000.0
        for f in frames:
            self.recent.append(f)
            ep = self.episode
            if f.speech:
                if ep is None:
                    ep = self.episode = SpeechEpisode(f.t - dt, f.t)
                else:
                    pause = f.t - dt - ep.t_last_voiced
                    if 0 < pause <= self.cfg.fill_s + 1e-6:
                        ep.voiced_s += pause         # short pause between syllables
                ep.voiced_s += dt
                ep.t_last_voiced = f.t
                ep.peak_db = max(ep.peak_db, f.db)
                if f.pitched is not False:
                    ep.pitched_s += dt
                ep.run_len += 1
                if ep.run_len == 2:
                    ep.runs += 1
            elif ep is not None:
                ep.run_len = 0
            if ep is None:
                continue
            ep.dbs.append(f.db)
            ep.total_s = f.t - ep.t_start
            if f.t - ep.t_last_voiced > self.cfg.gap_s:
                self.episode = None                  # silence: episode over
                if not ep.fired and ep.voiced_s >= 0.4:
                    self.last_rejected = ep.summary()     # heard something, not enough to flag
                continue
            ratio = ep.voiced_s / max(ep.total_s, dt)
            if (not ep.fired and ep.voiced_s >= self.cfg.min_speech_s and ratio >= self.cfg.min_ratio
                    and ep.runs >= self.cfg.min_runs and ep.modulation_db >= self.cfg.min_modulation_db
                    and ep.pitched_s >= self.cfg.min_pitched_s
                    and ep.pitched_s >= self.cfg.min_pitched_share * ep.voiced_s
                    and f.speech):                  # decide while the sound is on, not on its tail
                ep.fired = True
                att = attribute_speaker(rows_between(ep.t_start, f.t), self.cfg, self.mouth_baseline_std)
                out.append(Event("VOICE_DETECTED", ep.t_start, f.t, att.speaker,
                                 score=round(ep.voiced_s, 2),
                                 details={"speech_s": round(ep.voiced_s, 1), "speaker": att.speaker,
                                          "lip_sync": att.detail, **att.to_dict(),
                                          "loudness_db": round(ep.peak_db, 1)}))
        self.speaking = self.episode is not None and (frames[-1].t - self.episode.t_last_voiced < 0.5 if frames else False)
        return out

    def speech_seconds(self, t0: float, t1: float) -> float:
        return sum(FRAME_MS / 1000.0 for f in self.recent if t0 <= f.t <= t1 and f.speech)

    def segments(self, t0: float, t1: float) -> list[tuple[float, float]]:
        """Voiced segments inside [t0, t1], merged across gaps < 0.3 s (for the timeline)."""
        segs: list[list[float]] = []
        dt = FRAME_MS / 1000.0
        for f in self.recent:
            if not (t0 <= f.t <= t1) or not f.speech:
                continue
            if segs and f.t - dt - segs[-1][1] < 0.3:
                segs[-1][1] = f.t
            else:
                segs.append([f.t - dt, f.t])
        return [(a, b) for a, b in segs if b - a >= 0.2]


def load_media_audio(path: str) -> Optional[np.ndarray]:
    """Audio track of a recorded video/audio file as 16 kHz mono int16 (for replay/evaluation).
    Uses PyAV, which faster-whisper already installs. None if there's no audio track."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(path) as c:
            if not c.streams.audio:
                return None
            res = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            parts = []
            for frame in c.decode(audio=0):
                for f in res.resample(frame):
                    parts.append(f.to_ndarray().reshape(-1))
            for f in res.resample(None):
                parts.append(f.to_ndarray().reshape(-1))
        return np.concatenate(parts).astype(np.int16) if parts else None
    except Exception:
        return None


# ---- transcription --------------------------------------------------------------------

class Transcriber:
    """Speech-to-text for flagged clips only (never the whole exam).

    Uses faster-whisper (CTranslate2, CPU int8, no PyTorch). The model is
    downloaded once on first use to backend/models/. PROCTOR_ASR_MODEL picks the
    size: tiny | base (default) | small | medium. Multilingual by default, so
    Hindi/Telugu/English speech all come out as text; set PROCTOR_ASR_LANGUAGE
    to force one. If faster-whisper isn't installed, transcription is simply
    skipped and the judge still gets the audio signal and lip-sync result.
    """

    def __init__(self, model: Optional[str] = None, language: Optional[str] = None):
        self.model_name = model or os.environ.get("PROCTOR_ASR_MODEL", "base")
        self.language = language or os.environ.get("PROCTOR_ASR_LANGUAGE") or None
        self._model = None
        self._lock = threading.Lock()
        self.error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return os.environ.get("PROCTOR_ASR", "1") != "0"

    def _load(self):
        if self._model is None and self.error is None:
            try:
                from faster_whisper import WhisperModel
                root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "whisper")
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8", download_root=root)
            except Exception as e:                       # not installed / offline: skip ASR
                self.error = f"{type(e).__name__}: {e}"[:200]
                print(f"[proctor] speech-to-text disabled ({self.error})", flush=True)
        return self._model

    def warm_up(self) -> None:
        with self._lock:
            if self._load() is not None:
                print(f"[proctor] speech-to-text ready (faster-whisper {self.model_name})", flush=True)

    def transcribe(self, pcm: np.ndarray) -> dict:
        if not self.enabled or len(pcm) < SAMPLE_RATE // 2:
            return {"text": "", "language": None}
        with self._lock:                                 # one transcription at a time on CPU
            model = self._load()
            if model is None:
                return {"text": "", "language": None, "error": self.error}
            audio = pcm.astype(np.float32) / 32768.0
            segments, info = model.transcribe(audio, language=self.language, beam_size=1,
                                              vad_filter=True, condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in segments).strip()
        return {"text": text[:500], "language": getattr(info, "language", None)}


class NullTranscriber:
    enabled = False

    def transcribe(self, pcm) -> dict:
        return {"text": "", "language": None}
