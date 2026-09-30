"""Synthetic face landmarks for tests (projects a generic 3D face)."""
import numpy as np

from proctor.features import EYE_A, EYE_B, FACE_MODEL_3D, POSE_IDX, camera_matrix, ypr_to_rotation

W, H = 640, 480

# Extra 3D model points (same camera-aligned frame as FACE_MODEL_3D) so a
# synthetic face has every landmark the feature code reads.
EXTRA_3D = {
    133: [-80, -170, 140], 362: [80, -170, 140],          # inner eye corners
    160: [-185, -192, 138], 158: [-120, -192, 139],      # eye A upper lid
    153: [-120, -148, 139], 144: [-185, -148, 138],      # eye A lower lid
    385: [120, -192, 139], 387: [185, -192, 138],        # eye B upper lid
    373: [185, -148, 138], 380: [120, -148, 139],        # eye B lower lid
}


MOUTH_3D = {78: [-110, 150, 125], 308: [110, 150, 125]}      # inner lip corners


def synthetic_face(yaw=0.0, pitch=0.0, roll=0.0, iris_h=0.5, iris_v=0.0, eye_open=1.0,
                   dist=1500.0, shift=(0.0, 0.0), scale=1.0, mouth_open=0.0):
    """Project a generic 3D face into a 478x3 landmark array.

    yaw/pitch use the *output* convention of head_pose (+yaw = image-right,
    +pitch = up), so the raw rotation is built with the signs flipped.
    """
    pts3d = {i: FACE_MODEL_3D[k] for k, i in enumerate(POSE_IDX)}
    for i, p in EXTRA_3D.items():
        p = np.array(p, float)
        if i in (160, 158, 385, 387, 153, 144, 373, 380):   # open/close lids
            p[1] = -170 + (p[1] + 170) * eye_open
        pts3d[i] = p
    # inner lips: gap of mouth_open * 120 units -> mouth aspect ratio ~ 0.55 * mouth_open
    pts3d.update({i: np.array(p, float) for i, p in MOUTH_3D.items()})
    gap = 120.0 * mouth_open
    pts3d[13] = np.array([0.0, 150.0 - gap / 2, 120.0])
    pts3d[14] = np.array([0.0, 150.0 + gap / 2, 120.0])
    R = ypr_to_rotation(-yaw, -pitch, roll)
    K = camera_matrix(W, H)
    lm = np.zeros((478, 3))
    for i, p in pts3d.items():
        c = R @ (np.asarray(p, float) * scale) + np.array([0, 0, dist])
        uv = K @ c
        lm[i, :2] = uv[:2] / uv[2] + np.array(shift)
    # Iris placed in 2D relative to the projected corners.
    for eye in (EYE_A, EYE_B):
        c1, c2 = lm[eye["corner_l"], :2], lm[eye["corner_r"], :2]
        axis = c2 - c1
        normal = np.array([-axis[1], axis[0]])
        lm[eye["iris"], :2] = c1 + iris_h * axis + iris_v * normal
    return lm




def looking_at(sx, sy, rng=None, noise=0.0, head_gain=1.0, eye_open=1.0, mouth_open=0.0):
    """Landmarks of a candidate looking at normalised screen point (sx, sy).

    Mimics a laptop camera below the screen: looking at the screen centre
    already reads as ~10 deg pitch up. Gaze is split between head and eyes;
    head_gain < 1 = an "eyes-only" person whose head barely moves on-screen.
    """
    j = (lambda s: rng.normal(0, s)) if (rng is not None and noise) else (lambda s: 0.0)
    return synthetic_face(yaw=-9.0 * head_gain * sx + j(noise * 2), pitch=10.0 - 7.0 * head_gain * sy + j(noise * 2),
                          iris_h=0.5 - 0.07 * sx + j(noise * 0.01), iris_v=0.04 * sy + j(noise * 0.005),
                          eye_open=eye_open, mouth_open=max(0.0, mouth_open + j(noise * 0.01)))


def looking_down_at_lap(rng=None, noise=0.0, pitch_drop=18.0, eye_open=0.45):
    """What the recording showed: head tilts down and the eyelids droop so much
    that the eye aspect ratio falls into 'blink' range for as long as it lasts."""
    j = (lambda s: rng.normal(0, s)) if (rng is not None and noise) else (lambda s: 0.0)
    return synthetic_face(yaw=j(noise * 2), pitch=10.0 - pitch_drop + j(noise * 2),
                          iris_h=0.5 + j(noise * 0.01), iris_v=0.02 + j(noise * 0.005), eye_open=eye_open)


class ScriptedLandmarker:
    """Fake landmarker: `script(t_seconds)` -> list of landmark arrays."""

    def __init__(self, script):
        self.script = script
        self.closed = False

    def __call__(self, frame, ts_ms):
        return self.script(ts_ms / 1000.0)

    def close(self):
        self.closed = True


def voice(seconds, rng, f0=130.0, amp=6000, rate=16000, t0=0.0):
    """Speech-like audio: harmonic voice with a ~4 Hz syllable envelope (passes WebRTC VAD).
    `t0` continues the envelope across consecutive short chunks (see voice_stream)."""
    n = int(seconds * rate)
    t = t0 + np.arange(n) / rate
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 25))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4 * t)
    return (sig / np.max(np.abs(sig)) * amp * env + rng.normal(0, 50, n)).astype(np.int16)


def voice_stream(rng, **kw):
    """Callable seconds -> next chunk of one continuous voice (syllable rhythm kept across chunks)."""
    state = {"t": 0.0}

    def nxt(seconds):
        out = voice(seconds, rng, t0=state["t"], **kw)
        state["t"] += len(out) / kw.get("rate", 16000)
        return out
    return nxt


def room_noise(seconds, rng, amp=60, rate=16000):
    return rng.normal(0, amp, int(seconds * rate)).astype(np.int16)


def _resonate(x, freq, bw, rate=16000):
    from scipy.signal import lfilter
    r = np.exp(-np.pi * bw / rate)
    return lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * freq / rate), r * r], x)


def syllabic_speech(seconds, rng, amp=2000, rate=16000):
    """More realistic speech: 120-250 ms voiced syllables (glottal pulse train through three
    formant resonators) separated by short pauses and longer word gaps."""
    parts, total = [], 0.0
    while total < seconds:
        for _ in range(int(rng.integers(1, 4))):
            n = int(rng.uniform(0.12, 0.25) * rate)
            t = np.arange(n) / rate
            f0 = rng.uniform(100, 150) * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 3) * t))
            ph = 2 * np.pi * np.cumsum(f0) / rate
            src = sum(np.sin(k * ph) / k for k in range(1, 30))
            syl = sum(_resonate(src, f, bw, rate) for f, bw in
                      ((rng.uniform(300, 800), 80), (rng.uniform(900, 2200), 120), (2600, 200)))
            parts += [syl * np.sin(np.pi * np.arange(n) / n) ** 0.7, np.zeros(int(rng.uniform(0.03, 0.12) * rate))]
        parts.append(np.zeros(int(rng.uniform(0.15, 0.45) * rate)))
        total = sum(len(p) for p in parts) / rate
    x = np.concatenate(parts)
    return x / np.max(np.abs(x)) * amp


def pink_noise(seconds, rng, amp, rate=16000):
    from scipy.signal import lfilter
    p = lfilter([1], [1, -0.95], rng.normal(0, 1, int(seconds * rate)))
    return p / np.std(p) * amp


def typing_clicks(seconds, rng, amp, rate=16000):
    x, k = np.zeros(int(seconds * rate)), 0
    while k < len(x) - 400:
        x[k:k + 300] += rng.normal(0, amp, 300) * np.exp(-np.arange(300) / 40)
        k += 1200 + int(rng.integers(0, 1600))
    return x
