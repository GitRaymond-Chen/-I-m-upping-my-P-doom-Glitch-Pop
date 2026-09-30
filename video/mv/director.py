"""Scene schedule, transitions, the P(doom) HUD and the per-frame FX pipeline."""
import os
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from . import fx
from .core import (BG, CREAM, FPS, GREEN, H, RED, W, YELLOW, clamp, ease_out, hash01, lerp,
                   mix, rng_for)
from .gfx import font, text_layer, paste, to_np


@dataclass
class Scene:
    key: str
    start: float
    end: float
    line: dict = None
    trans: str = "glitch"
    tdur: float = 0.2
    lines: list = field(default_factory=list)


class Ctx:
    """Everything a scene function needs for one frame."""

    def __init__(self, tl, scene, t, frame):
        self.tl = tl
        self.scene = scene
        self.t = t
        self.frame = frame
        self.lt = t - scene.start
        self.dur = scene.end - scene.start
        self.u = clamp(self.lt / self.dur) if self.dur > 0 else 0.0
        self.line = scene.line
        self.img = Image.new("RGB", (W, H), BG)
        self.fx = {}
        self.hud = True

    # word helpers --------------------------------------------------------------
    def wt(self, i, line=None):
        line = line or self.line
        return line["words"][i]["start"]

    def since(self, i, line=None):
        return self.t - self.wt(i, line)

    def widx(self, line=None):
        return self.tl.word_state(line or self.line, self.t)

    def word_named(self, name, line=None):
        line = line or self.line
        for i, w in enumerate(line["words"]):
            if w["w"].lower().strip(",.?!\"'") == name.lower():
                return i
        raise KeyError(name)

    def at(self, name, line=None):
        return self.wt(self.word_named(name, line), line)

    # audio helpers ---------------------------------------------------------------
    def kick(self, decay=0.12):
        return self.tl.pulse(self.tl.kicks, self.t, decay)

    def snare(self, decay=0.1):
        return self.tl.pulse(self.tl.snares, self.t, decay)

    def beat(self, decay=0.15):
        return self.tl.pulse(self.tl.beats, self.t, decay)

    def downbeat(self, decay=0.2):
        return self.tl.pulse(self.tl.downbeats, self.t, decay)

    def beat_phase(self):
        return self.tl.beat_info(self.t)

    def bass(self):
        return self.tl.feat("bass", self.t)

    def rms(self):
        return self.tl.feat("rms", self.t)

    def bounce(self, amp=1.0):
        """0..1 hop synced to the beat (peaks mid-beat)."""
        _, ph = self.tl.beat_info(self.t)
        return math.sin(ph * math.pi) * amp

    def rng(self, *k):
        return rng_for(self.frame, *k)


# --- P(doom) meter --------------------------------------------------------------------
PDOOM_EVENTS = [
    (0.0, 0.01), (10.87, 0.02), (17.68, 0.04), (22.97, 0.07), (28.0, 0.10),
    (31.50, 0.25), (33.26, 0.33), (38.0, 0.36), (42.87, 0.40),
    (50.41, 0.44), (55.59, 0.48), (59.20, 0.52), (66.08, 0.55),
    (68.10, 0.66), (70.03, 0.70), (72.13, 0.74), (74.55, 0.78), (76.38, 0.50), (77.33, 0.79),
    (90.28, 0.80), (93.41, 0.82), (94.68, 0.85), (101.09, 0.86),
    (104.50, 0.92), (105.80, 0.94), (108.69, 0.95), (112.02, 0.96),
    (124.00, 0.97), (129.27, 0.975), (132.76, 0.98),
    (135.22, 0.99), (139.67, 0.995), (142.48, 0.999), (149.89, 0.9999), (153.0, 1.0),
]
CHAOS = (77.8, 90.2)  # interlude: the meter glitches out


def pdoom_value(t):
    prev = PDOOM_EVENTS[0][1]
    for (te, v) in PDOOM_EVENTS:
        if t < te:
            break
        k = ease_out((t - te) / 0.45)
        prev = lerp(prev, v, k) if k < 1 else v
    return prev


def pdoom_jump(t):
    """Seconds since the last meter change (for flash/pulse)."""
    last = -1e9
    for (te, v) in PDOOM_EVENTS:
        if te <= t:
            last = te
    return t - last


def pdoom_color(v):
    if v < 0.3:
        return mix(GREEN, YELLOW, v / 0.3)
    if v < 0.7:
        return mix(YELLOW, (255, 120, 40), (v - 0.3) / 0.4)
    return mix((255, 120, 40), RED, clamp((v - 0.7) / 0.25))


def format_p(v):
    if v >= 1.0:
        return "1.00"
    if v >= 0.999:
        return f"{v:.4f}"
    if v >= 0.99:
        return f"{v:.3f}"
    return f"{v:.2f}"


def hud_composite(arr, t, frame, alpha=1.0):
    """Draw the P(doom) meter straight into the numpy frame (top-right corner)."""
    if alpha <= 0.01:
        return arr
    lw, lh = 640, 110
    img = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    draw_hud(img, t, frame, 1.0, x1=lw - 50, y0=24)
    a = np.asarray(img, dtype=np.float32)
    x0, y0 = W - lw - 8, 6
    region = arr[y0:y0 + lh, x0:x0 + lw].astype(np.float32)
    al = a[..., 3:4] / 255.0 * alpha
    arr = arr.copy() if not arr.flags.writeable else arr
    arr[y0:y0 + lh, x0:x0 + lw] = (region * (1 - al) + a[..., :3] * al).astype(np.uint8)
    return arr


def draw_hud(img, t, frame, alpha=1.0, x1=W - 40, y0=34):
    if alpha <= 0:
        return
    v = pdoom_value(t)
    since = pdoom_jump(t)
    pulse = math.exp(-since / 0.25) if since < 2 else 0.0
    chaos = CHAOS[0] <= t < CHAOS[1]
    overflow = t >= 156.0
    label = "P(doom)"
    if chaos:
        glyphs = ["NaN", "0.??", "-0.3", "4.20", "∞", "0.5?", "404", "1e30", "????"]
        val = glyphs[int(hash01(frame // 3, 77) * len(glyphs))]
        v_bar = hash01(frame // 2, 5)
        col = mix(pdoom_color(v_bar), (255, 255, 255), 0.3 * (frame % 2))
    elif overflow:
        k = t - 156.0
        val = ["1.01", "1.1", "1.337", "2.0", "9.99", "ERR!", "OVERFLOW"][min(6, int(k / 0.6))]
        v_bar = 1.0
        col = RED if frame % 4 < 2 else (255, 255, 255)
    else:
        val = format_p(v)
        v_bar = v
        col = pdoom_color(v)
    fnt = "mono"
    size = 25
    segs = 12
    seg_w, seg_h, gap = 10, 20, 3
    val_w = max(80, font(fnt, size).getlength(val) + 4)
    lab_w = font(fnt, size).getlength(label)
    bar_w = segs * (seg_w + gap)
    total = lab_w + 16 + bar_w + 14 + val_w
    x0 = x1 - total
    sc = 1 + 0.14 * pulse
    # panel (drawn on a layer so it can pulse-scale)
    pw, ph = int(total + 32), 52
    layer = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((0, 0, pw - 1, ph - 1), radius=10, fill=(12, 10, 20, 215),
                        outline=(*col, 255), width=3)
    base_y = 35  # text baseline inside the panel
    li, ox, oy = text_layer(label, fnt, size, CREAM)
    layer.paste(li, (int(16 - ox), int(base_y - oy)), li)
    bx = 16 + lab_w + 16
    lit = int(round(clamp(v_bar) * segs))
    for s in range(segs):
        c = col if s < lit else (45, 42, 60)
        d.rectangle((bx + s * (seg_w + gap), 16, bx + s * (seg_w + gap) + seg_w, 16 + seg_h),
                    fill=(*c, 255))
    vi, vox, voy = text_layer(val, fnt, size, col)
    layer.paste(vi, (int(bx + bar_w + 14 - vox), int(base_y - voy)), vi)
    paste(img, layer, x0 - 16 + pw / 2, y0 + ph / 2, scale=sc, alpha=alpha)
    return pulse


# --- schedule ---------------------------------------------------------------------------
# Transition into each scene: (type, duration)
TRANSITIONS = {
    "L01": ("crt_on", 0.5), "L02": ("zoom", 0.35), "L03": ("slices", 0.3), "L04": ("glitch", 0.2),
    "L05": ("flash", 0.3), "L06": ("black", 0.25), "L07": ("whip_up", 0.3), "L08": ("dissolve", 0.3),
    "L09": ("glitch", 0.25), "L10": ("crt_on", 0.45), "L11": ("diagonal", 0.35),
    "L12": ("black", 0.3), "L13": ("flash", 0.3), "L14": ("dissolve", 0.35),
    "L15": ("flash", 0.25), "L16": ("whip", 0.3), "L17": ("slices_v", 0.3), "L18": ("glitch", 0.2),
    "interlude": ("glitch", 0.3), "L26": ("glitch", 0.3), "L27": ("checker", 0.35),
    "L28": ("glitch", 0.2), "L29": ("whip", 0.3), "L30": ("dissolve", 0.35), "L31": ("flash", 0.3),
    "L32": ("slices", 0.3), "L33": ("glitch", 0.2), "L34": ("flash_blue", 0.4),
    "L35": ("crt_on", 0.45), "L36": ("glitch", 0.25), "L37": ("slices_v", 0.3),
    "L38": ("flash", 0.3), "L39": ("clawd", 0.45), "L40": ("black", 0.4),
    "L41": ("dissolve", 0.4), "outro": ("crt_off", 0.5),
}


def build_schedule(tl):
    beats = tl.beats

    def snap(t, latest):
        # snap scene boundaries to a nearby beat so cuts land on the groove,
        # but never after the line's first sung word
        i = int(np.argmin(np.abs(beats - t)))
        if abs(beats[i] - t) < 0.16 and beats[i] <= latest:
            return float(beats[i])
        return t

    lines = tl.lines
    starts = {}
    for i, line in enumerate(lines):
        starts[i] = snap(line["start"] - 0.12, line["start"] - 0.04)
    scenes = [Scene("intro", 0.0, starts[0], None, "none", 0)]
    i = 0
    while i < len(lines):
        sec = lines[i]["section"]
        if sec == "Interlude":
            j = i
            while j < len(lines) and lines[j]["section"] == "Interlude":
                j += 1
            scenes.append(Scene("interlude", starts[i], starts[j], lines[i],
                                lines=lines[i:j]))
            i = j
            continue
        key = f"L{i + 1:02d}"
        end = starts[i + 1] if i + 1 < len(lines) else 152.2
        scenes.append(Scene(key, starts[i], end, lines[i]))
        i += 1
    scenes.append(Scene("outro", scenes[-1].end, tl.duration + 0.5))
    for s in scenes:
        if s.key in TRANSITIONS:
            s.trans, s.tdur = TRANSITIONS[s.key]
    return scenes


def scene_at(schedule, t):
    for k, s in enumerate(schedule):
        if s.start <= t < s.end:
            return k
    return len(schedule) - 1


# --- rendering ---------------------------------------------------------------------------
def run_scene(tl, scene, t, frame, fns):
    ctx = Ctx(tl, scene, t, frame)
    fns[scene.key](ctx)
    arr = to_np(ctx.img)
    arr = apply_scene_fx(arr, ctx)
    return arr, ctx


def apply_scene_fx(arr, ctx):
    p = ctx.fx
    if p.get("hue"):
        arr = fx.hue_shift(arr, p["hue"])
    if p.get("swirl"):
        arr = fx.swirl(arr, p["swirl"], p.get("swirl_r", 800), *p.get("swirl_c", (W / 2, H / 2)))
    if p.get("wave"):
        amp, freq, ph = p["wave"]
        arr = fx.wave(arr, amp, freq, ph)
    if p.get("vwave"):
        amp, freq, ph = p["vwave"]
        arr = fx.wave(arr, amp, freq, ph, vertical=True)
    if p.get("skew"):
        arr = fx.skew(arr, *p["skew"])
    zoom = p.get("zoom", 1.0)
    ang = p.get("angle", 0.0)
    dx, dy = p.get("shake", (0, 0))
    if zoom != 1.0 or ang or dx or dy:
        arr = fx.warp(arr, zoom, ang, dx, dy, *p.get("center", (W / 2, H / 2)))
    if p.get("pixelate"):
        arr = fx.pixelate(arr, p["pixelate"])
    if p.get("bloom"):
        from .gfx import glow_layer
        arr = glow_layer(arr, p.get("bloom_r", 30), p["bloom"], p.get("bloom_th", 150))
    if p.get("invert"):
        arr = fx.invert(arr, p["invert"])
    if p.get("tint"):
        arr = fx.tint(arr, *p["tint"])
    if p.get("flash"):
        arr = fx.flash(arr, p.get("flash_color", (255, 255, 255)), p["flash"])
    return arr


def transition(kind, A, B, p, frame, t):
    if kind in ("none",):
        return B if p >= 0.5 else A
    if kind == "glitch":
        base = B if p >= 0.5 else A
        return fx.block_glitch(fx.rgb_split(base, 18 * (1 - abs(p - 0.5) * 2)),
                               1 - abs(p - 0.5) * 2, frame)
    if kind == "slices":
        return fx.mix(A, B, fx.mask_slices(p, 14))
    if kind == "slices_v":
        return fx.mix(A, B, fx.mask_slices(p, 16, horizontal=False))
    if kind == "dissolve":
        return fx.mix(A, B, fx.mask_dissolve(p, 48))
    if kind == "checker":
        return fx.mix(A, B, fx.mask_checker(p, 6))
    if kind == "diagonal":
        return fx.mix(A, B, fx.mask_diagonal(p))
    if kind == "circle":
        return fx.mix(A, B, fx.mask_circle(p))
    if kind == "clawd":
        from .clawd import clawd_mask
        m = clawd_mask(40)
        sc = 0.02 + ease_in_expo(p) * 14
        return fx.mix(A, B, fx.mask_from_image(m, W / 2, H / 2 + 60 * sc, sc))
    if kind in ("flash", "flash_blue"):
        col = (255, 255, 255) if kind == "flash" else (90, 140, 255)
        amt = 1 - abs(p - 0.5) * 2
        base = B if p >= 0.5 else A
        return fx.flash(base, col, amt ** 0.7)
    if kind == "black":
        amt = 1 - abs(p - 0.5) * 2
        base = B if p >= 0.5 else A
        return fx.flash(base, (0, 0, 0), clamp(amt * 1.4))
    if kind == "zoom":
        if p < 0.5:
            return fx.flash(fx.warp(A, 1 + 4 * ease_in_expo(p * 2)), (255, 255, 255), p * 1.4)
        return fx.flash(fx.warp(B, 0.8 + 0.2 * ease_out((p - 0.5) * 2)), (255, 255, 255),
                        (1 - p) * 1.4)
    if kind in ("whip", "whip_up"):
        e = ease_in_out_cubic(p)
        if kind == "whip":
            a = fx.warp(A, 1, 0, -e * W, 0)
            b = fx.warp(B, 1, 0, (1 - e) * W, 0)
            m = np.zeros((H, W), np.float32)
            m[:, int((1 - e) * W):] = 1
        else:
            a = fx.warp(A, 1, 0, 0, e * H)
            b = fx.warp(B, 1, 0, 0, -(1 - e) * H)
            m = np.zeros((H, W), np.float32)
            m[: int(e * H), :] = 1
        out = fx.mix(a, b, m)
        k = 1 - abs(p - 0.5) * 2
        return fx.rgb_split(out, 30 * k)
    if kind == "crt_on":
        if p < 0.5:
            return fx.flash(A, (0, 0, 0), p * 2)
        k = ease_out((p - 0.5) * 2)
        out = np.zeros_like(B)
        hh = max(2, int(H * k))
        ww = W if k > 0.15 else int(W * k / 0.15)
        src = fx.warp(B, 1, 0, 0, 0)
        y0 = (H - hh) // 2
        x0 = (W - ww) // 2
        import cv2
        squashed = cv2.resize(src, (max(1, ww), hh), interpolation=cv2.INTER_AREA)
        out[y0:y0 + hh, x0:x0 + ww] = squashed
        return fx.flash(out, (255, 255, 255), (1 - k) * 0.6)
    if kind == "crt_off":
        if p >= 0.5:
            return B
        k = 1 - ease_in_expo(p * 2)
        out = np.zeros_like(A)
        import cv2
        hh = max(2, int(H * k))
        ww = W if k > 0.1 else max(2, int(W * k / 0.1))
        squashed = cv2.resize(A, (ww, hh), interpolation=cv2.INTER_AREA)
        out[(H - hh) // 2:(H - hh) // 2 + hh, (W - ww) // 2:(W - ww) // 2 + ww] = squashed
        return fx.flash(out, (255, 255, 255), (1 - k) * 0.5)
    return B if p >= 0.5 else A


def ease_in_expo(t):
    t = clamp(t)
    return 0 if t == 0 else 2 ** (10 * t - 10)


def ease_in_out_cubic(t):
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def render_frame(tl, schedule, fns, frame, t=None):
    if t is None:
        t = frame / FPS
    k = scene_at(schedule, t)
    cur = schedule[k]
    arr, ctx = run_scene(tl, cur, t, frame, fns)
    hud_alpha = 1.0 if ctx.hud else 0.0
    # transition in (from previous scene) around the boundary
    if k > 0 and cur.tdur > 0:
        half = cur.tdur / 2
        if t < cur.start + half:
            prev = schedule[k - 1]
            A, _ = run_scene(tl, prev, t, frame, fns)
            p = clamp((t - (cur.start - half)) / cur.tdur)
            arr = transition(cur.trans, A, arr, p, frame, t)
    if k + 1 < len(schedule):
        nxt = schedule[k + 1]
        half = nxt.tdur / 2
        if nxt.tdur > 0 and t >= nxt.start - half:
            B, _ = run_scene(tl, nxt, t, frame, fns)
            p = clamp((t - (nxt.start - half)) / nxt.tdur)
            arr = transition(nxt.trans, arr, B, p, frame, t)
    # global beat-reactive finish
    p = ctx.fx
    kick = tl.pulse(tl.kicks, t, 0.1)
    snare = tl.pulse(tl.snares, t, 0.08)
    split = p.get("rgb", 0) + 3 + 5 * snare * p.get("react", 1.0)
    arr = fx.rgb_split(arr, split)
    g = p.get("glitch", 0.0)
    if g > 0:
        arr = fx.block_glitch(arr, g, frame * 7 + 3)
    punch = p.get("punch", 1.0)
    if punch and kick > 0.05 and p.get("react", 1.0) > 0:
        arr = fx.warp(arr, 1 + 0.018 * kick * punch)
    # HUD last-ish so it stays legible through the glitching
    if hud_alpha > 0:
        arr = hud_composite(np.ascontiguousarray(arr), t, frame,
                            hud_alpha * p.get("hud_alpha", 1.0))
    arr = fx.crt(arr, scan=p.get("scan", 0.55), vignette=p.get("vignette", 0.8))
    arr = fx.grain(arr, p.get("grain", float(os.environ.get("MV_GRAIN", "0"))), frame)
    return arr
