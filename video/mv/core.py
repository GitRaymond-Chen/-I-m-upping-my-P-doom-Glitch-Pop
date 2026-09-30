"""Constants, palette, easing and the song timeline (lyrics + audio features)."""
import json
import math
import os

import numpy as np

W, H = 1920, 1080
FPS = 30
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FONTS = os.path.join(ROOT, "fonts")

# --- palette -------------------------------------------------------------------
BG = (11, 10, 18)
INK = (16, 14, 24)
CLAWD = (217, 119, 87)       # Claude orange
CLAWD_DARK = (168, 82, 58)
CREAM = (245, 240, 230)
CYAN = (61, 245, 255)
PINK = (255, 61, 165)
LIME = (198, 255, 61)
RED = (255, 42, 42)
PURPLE = (139, 92, 255)
GREEN = (57, 255, 136)
YELLOW = (255, 214, 10)
BLUE = (40, 90, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# --- math helpers ----------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def inv_lerp(a, b, x):
    return clamp((x - a) / (b - a)) if b != a else 0.0


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0)) if e1 != e0 else float(x >= e1)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in(t):
    t = clamp(t)
    return t * t * t


def ease_in_out(t):
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_back(t, s=1.70158):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def ease_out_elastic(t):
    t = clamp(t)
    if t in (0.0, 1.0):
        return t
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi) / 3) + 1


def mix(c1, c2, t):
    t = clamp(t)
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def scale_color(c, k):
    return tuple(int(clamp(v * k, 0, 255)) for v in c)


def hsv(h, s=1.0, v=1.0):
    h = (h % 1.0) * 6
    i = int(h)
    f = h - i
    p, q, r = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    rgb = [(v, r, p), (q, v, p), (p, v, r), (p, q, v), (r, p, v), (v, p, q)][i % 6]
    return tuple(int(255 * x) for x in rgb)


def hash01(*args):
    """Deterministic pseudo-random float in [0, 1) from integers/floats."""
    x = 0x9E3779B9
    for a in args:
        x ^= (int(a * 1000003) if isinstance(a, float) else int(a)) & 0xFFFFFFFF
        x = (x * 0x85EBCA6B) & 0xFFFFFFFF
        x ^= x >> 13
        x = (x * 0xC2B2AE35) & 0xFFFFFFFF
        x ^= x >> 16
    return (x & 0xFFFFFF) / float(0x1000000)


def rng_for(*args):
    return np.random.default_rng(int(hash01(*args) * 2**31))


# --- timeline ----------------------------------------------------------------------
class Timeline:
    def __init__(self):
        tl = json.load(open(os.path.join(DATA, "timeline.json")))
        self.lines = tl["lines"]
        for i, line in enumerate(self.lines):
            line["idx"] = i
            nxt = self.lines[i + 1]["start"] if i + 1 < len(self.lines) else None
            line["next_start"] = nxt
        beats = json.load(open(os.path.join(DATA, "beats.json")))
        self.duration = beats["duration"]
        self.beats = np.array(beats["beats"])
        self.downbeats = np.array(beats["downbeats"])
        self.kicks = np.array(beats["kicks"])
        self.snares = np.array(beats["snares"])
        self.onsets = np.array(beats["onsets"])
        f = np.load(os.path.join(DATA, "audio_features.npz"))
        self.feat_fps = float(f["fps"])
        self.features = {k: f[k] for k in ("rms", "onset", "bass", "low_mid", "high_mid", "treble")}

    def feat(self, name, t):
        arr = self.features[name]
        x = t * self.feat_fps
        i = int(x)
        if i < 0:
            return float(arr[0])
        if i >= len(arr) - 1:
            return float(arr[-1])
        fr = x - i
        return float(arr[i] * (1 - fr) + arr[i + 1] * fr)

    def smooth_feat(self, name, t, width=0.15):
        return sum(self.feat(name, t - width * k / 4) for k in range(5)) / 5

    def since(self, times, t):
        """Seconds since the most recent event in `times` (inf if none yet)."""
        i = np.searchsorted(times, t, side="right") - 1
        return t - times[i] if i >= 0 else math.inf

    def pulse(self, times, t, decay=0.12):
        dt = self.since(times, t)
        return math.exp(-dt / decay) if dt != math.inf else 0.0

    def beat_info(self, t):
        """(beat index, phase in [0,1)) using the tracked beats."""
        b = self.beats
        i = np.searchsorted(b, t, side="right") - 1
        if i < 0:
            period = b[1] - b[0]
            return -1, (t - (b[0] - period)) / period % 1.0
        if i >= len(b) - 1:
            period = b[-1] - b[-2]
            return i, ((t - b[-1]) / period) % 1.0
        return int(i), (t - b[i]) / (b[i + 1] - b[i])

    def bar_info(self, t):
        d = self.downbeats
        i = np.searchsorted(d, t, side="right") - 1
        if i < 0 or i >= len(d) - 1:
            return max(i, 0), 0.0
        return int(i), (t - d[i]) / (d[i + 1] - d[i])

    def line_at(self, t):
        for line in self.lines:
            end = line["next_start"] if line["next_start"] else line["end"] + 2
            if line["start"] - 0.05 <= t < end - 0.05:
                return line
        return None

    @staticmethod
    def word_state(line, t, lead=0.04):
        """Index of the word being sung at t (-1 before the first word)."""
        idx = -1
        for i, w in enumerate(line["words"]):
            if t + lead >= w["start"]:
                idx = i
        return idx
