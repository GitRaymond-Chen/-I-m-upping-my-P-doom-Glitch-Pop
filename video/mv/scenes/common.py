"""Shared scene building blocks: chat UI, chorus gauge motif, particles, props."""
import math

from PIL import Image, ImageDraw

from .. import bg
from ..clawd import blink_at, draw_clawd
from ..core import (CREAM, CYAN, H, PINK, RED, W, YELLOW, GREEN, clamp, ease_out, ease_out_back,
                    hash01, hsv, lerp, mix)
from ..gfx import draw_text, font, paste, text_layer
from ..typo import pop_words

PERSON = [
    "..EEE..",
    ".EEEEE.",
    ".EEEEE.",
    "..EEE..",
    ".BBBBB.",
    "BBBBBBB",
    "E.BBB.E",
    "E.BBB.E",
    "..ddd..",
    "..d.d..",
    "..d.d..",
    ".KK.KK.",
]


def register_person():
    from ..gfx import parse_sprite
    from ..sprites import PAL, SPRITES
    if "person" not in SPRITES:
        SPRITES["person"] = parse_sprite(PERSON, PAL)


register_person()


# --- chat window motif (pre-choruses) -----------------------------------------------
THEMES = {
    "gpt": dict(bg=(30, 30, 36), header=(44, 44, 52), accent=(200, 200, 210),
                user=(217, 119, 87), bot=(62, 62, 72), text=CREAM),
    "sydney": dict(bg=(40, 16, 48), header=(70, 24, 84), accent=(255, 120, 210),
                   user=(217, 119, 87), bot=(120, 40, 140), text=CREAM),
    "gato": dict(bg=(34, 26, 20), header=(70, 46, 30), accent=(255, 190, 110),
                 user=(217, 119, 87), bot=(110, 70, 40), text=CREAM),
}


def chat_window(img, box, title, theme, t, title_text=None, online=True):
    th = THEMES[theme]
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle((x0 + 12, y0 + 16, x1 + 12, y1 + 16), radius=28, fill=(0, 0, 0, 120))
    d.rounded_rectangle(box, radius=28, fill=(*th["bg"], 255), outline=(*th["accent"], 255),
                        width=4)
    d.rounded_rectangle((x0, y0, x1, y0 + 96), radius=28, fill=(*th["header"], 255))
    d.rectangle((x0 + 2, y0 + 60, x1 - 2, y0 + 96), fill=(*th["header"], 255))
    # avatar + name + status
    d.ellipse((x0 + 28, y0 + 20, x0 + 84, y0 + 76), fill=(*th["accent"], 255))
    draw_text(img, title_text if title_text is not None else title, x0 + 104, y0 + 60,
              "grotesk", 38, CREAM, anchor="ls")
    if online:
        d.ellipse((x0 + 70, y0 + 62, x0 + 86, y0 + 78), fill=(60, 220, 110, 255))
    for k in range(3):
        cx = x1 - 50 - k * 34
        d.ellipse((cx - 9, y0 + 39, cx + 9, y0 + 57), fill=(*th["accent"], 160))
    return (x0 + 30, y0 + 120, x1 - 30, y1 - 30)


def bubble(img, x, y, text, theme, side="right", t=0.0, t0=0.0, size=42, max_w=900,
           fname="grotesk", tail=True, alpha=1.0, color=None):
    """Chat bubble anchored at its top-right (side='right') or top-left corner."""
    if t < t0:
        return None
    th = THEMES[theme]
    k = ease_out_back(clamp((t - t0) / 0.25), 2.0)
    f = font(fname, size)
    w = min(max_w, f.getlength(text)) + 60
    h = size * 1.1 + 44
    col = color or (th["user"] if side == "right" else th["bot"])
    layer = Image.new("RGBA", (int(w + 40), int(h + 40)), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((10, 10, 10 + w, 10 + h), radius=int(h / 2.3), fill=(*col, 255))
    if tail:
        if side == "right":
            d.polygon([(w - 10, h), (w + 26, h + 24), (w - 36, h + 4)], fill=(*col, 255))
        else:
            d.polygon([(30, h), (-6 + 10, h + 24), (56, h + 4)], fill=(*col, 255))
    li, ox, oy = text_layer(text, fname, size, CREAM)
    layer.paste(li, (int(40 - ox), int(10 + h / 2 + size * 0.36 - oy)), li)
    ax = x - w - 10 if side == "right" else x - 10
    cx, cy = ax + (w + 40) / 2, y + (h + 40) / 2
    paste(img, layer, cx, cy, scale=0.3 + 0.7 * k, alpha=clamp(k * 2) * alpha)
    return (ax + 10, y + 10, ax + 10 + w, y + 10 + h)


# --- chorus motif: the P(doom) speedometer ---------------------------------------
def gauge(img, cx, cy, r, value, t, glow=0.0):
    d = ImageDraw.Draw(img, "RGBA")
    # coloured arc segments
    n = 40
    for i in range(n):
        a0 = math.pi + math.pi * i / n
        a1 = math.pi + math.pi * (i + 0.8) / n
        v = i / n
        if v < 0.3:
            col = mix(GREEN, YELLOW, v / 0.3)
        elif v < 0.7:
            col = mix(YELLOW, (255, 120, 40), (v - 0.3) / 0.4)
        else:
            col = mix((255, 120, 40), RED, (v - 0.7) / 0.3)
        lit = v <= value
        c = col if lit else mix((40, 30, 50), col, 0.25)
        pts = [(cx + (r - 70) * math.cos(a0), cy + (r - 70) * math.sin(a0)),
               (cx + r * math.cos(a0), cy + r * math.sin(a0)),
               (cx + r * math.cos(a1), cy + r * math.sin(a1)),
               (cx + (r - 70) * math.cos(a1), cy + (r - 70) * math.sin(a1))]
        d.polygon(pts, fill=(*c, 255))
    # ticks + labels
    for k in range(11):
        a = math.pi + math.pi * k / 10
        d.line((cx + (r - 90) * math.cos(a), cy + (r - 90) * math.sin(a),
                cx + (r - 110) * math.cos(a), cy + (r - 110) * math.sin(a)),
               fill=(230, 230, 240, 255), width=5)
    for k, lab in ((0, "0"), (5, ".5"), (10, "1")):
        a = math.pi + math.pi * k / 10
        draw_text(img, lab, cx + (r - 150) * math.cos(a), cy + (r - 150) * math.sin(a) + 10,
                  "mono", 34, CREAM)
    # needle (with a little overshoot wobble handled by caller)
    a = math.pi + math.pi * clamp(value, 0, 1.02)
    tip = (cx + (r - 40) * math.cos(a), cy + (r - 40) * math.sin(a))
    base_l = (cx + 22 * math.cos(a + math.pi / 2), cy + 22 * math.sin(a + math.pi / 2))
    base_r = (cx + 22 * math.cos(a - math.pi / 2), cy + 22 * math.sin(a - math.pi / 2))
    d.polygon([tip, base_l, base_r], fill=(255, 250, 240, 255))
    d.ellipse((cx - 36, cy - 36, cx + 36, cy + 36), fill=(30, 26, 40, 255),
              outline=(255, 250, 240, 255), width=6)


def chorus_hook(ctx, hook_end, v_before, v_after, theme_hue=0.0):
    """'I'm upping my P(doom)' motif. Draws until hook_end; returns True while active."""
    t = ctx.t
    img = ctx.img
    line = ctx.line
    i_pd = next(i for i, w in enumerate(line["words"]) if w["w"].lower().startswith("p(doom"))
    t_pd = line["words"][i_pd]["start"]
    if t >= hook_end:
        return False
    beat = ctx.beat(0.18)
    bg.sunburst(img, t, W / 2, H * 0.72, n=20, c1=mix((40, 6, 20), hsv(theme_hue, 1, 0.25), 0.4),
                c2=mix((95, 12, 40), hsv(theme_hue, 1, 0.5), 0.4), speed=0.35)
    # stacked hook words on the left
    words = ["I'M", "UPPING", "MY"]
    for k, wd in enumerate(words):
        ts = line["words"][k]["start"]
        if t + 0.04 < ts:
            continue
        kk = ease_out(clamp((t + 0.04 - ts) / 0.12))
        sc = 1.6 - 0.6 * kk
        draw_text(img, wd, 330, 220 + k * 190, "anton", 170, CREAM, scale=sc, alpha=kk,
                  shadow=(8, 8, (255, 40, 110)))
    # gauge
    kpd = clamp((t - t_pd) / 0.5)
    val = lerp(v_before, v_after, ease_out_back(kpd, 3.0)) if t >= t_pd else v_before
    wob = 0.015 * math.sin(t * 40) * (1 - kpd) if t >= t_pd else 0.004 * math.sin(t * 30)
    gauge(img, 1270, 700, 430, val + wob, t)
    lab = "P(DOOM)"
    if t >= t_pd:
        kk = ease_out(clamp((t - t_pd) / 0.1))
        draw_text(img, lab, 1270, 360, "anton", 150, (255, 60, 60), scale=2.2 - 1.2 * kk,
                  alpha=kk, shadow=(7, 7, (40, 0, 10)))
        pct = f"{val * 100:.0f}%" if v_after < 0.99 else f"{val * 100:.1f}%"
        draw_text(img, pct, 1270, 860, "anton", 110, CREAM)
        ctx.fx["shake"] = ((hash01(ctx.frame, 1) - 0.5) * 30 * (1 - kpd),
                           (hash01(ctx.frame, 2) - 0.5) * 30 * (1 - kpd))
        ctx.fx["flash"] = max(0.0, 0.7 - (t - t_pd) * 4)
        ctx.fx["rgb"] = 10 * (1 - kpd)
    else:
        draw_text(img, "P(DOOM)", 1270, 360, "anton", 150, (120, 40, 60), alpha=0.5)
    # Clawd dancing bottom-left
    bi, ph = ctx.beat_phase()
    arms = ("up", "out") if bi % 2 == 0 else ("out", "up")
    if t >= t_pd:
        arms = ("up", "up")
    hop = ctx.bounce(24)
    draw_clawd(img, 330, 1010 - hop, 13, arms=arms, eyes="wide" if t >= t_pd else "normal",
               blink=blink_at(t), squash=0.12 * beat, t=t)
    if hook_end - t < 0.07:
        ctx.fx["glitch"] = 0.3
    return True


# --- particles ------------------------------------------------------------------------
def burst(img, t, t0, x, y, n=40, speed=900, life=0.9, colors=(YELLOW, PINK, CYAN), size=12,
          gravity=1400, seed=0, spread=math.pi * 2, angle0=0.0):
    """Deterministic particle burst of pixel squares starting at t0."""
    if t < t0 or t > t0 + life:
        return
    d = ImageDraw.Draw(img, "RGBA")
    dt = t - t0
    for i in range(n):
        a = angle0 + (hash01(seed, i) - 0.5) * spread
        sp = speed * (0.35 + 0.65 * hash01(seed, i, 2))
        px = x + math.cos(a) * sp * dt
        py = y + math.sin(a) * sp * dt + 0.5 * gravity * dt * dt
        fade = 1 - dt / life
        col = colors[i % len(colors)]
        s = size * (0.5 + hash01(seed, i, 3))
        d.rectangle((px - s / 2, py - s / 2, px + s / 2, py + s / 2), fill=(*col, int(255 * fade)))


def lightning(img, x0, y0, x1, y1, t, color=(170, 250, 255), width=5, seed=0, segs=10,
              jag=40):
    d = ImageDraw.Draw(img, "RGBA")
    fr = int(t * 20)
    pts = [(x0, y0)]
    for i in range(1, segs):
        k = i / segs
        px = lerp(x0, x1, k) + (hash01(seed, fr, i) - 0.5) * jag * 2
        py = lerp(y0, y1, k) + (hash01(seed, fr, i, 1) - 0.5) * jag * 2
        pts.append((px, py))
    pts.append((x1, y1))
    d.line(pts, fill=(*color, 120), width=width * 3)
    d.line(pts, fill=(255, 255, 255, 255), width=width)


def lyric_caption(ctx, y=965, size=64, fname="grotesk", color=CREAM, hot=YELLOW, band=True,
                  max_w=1700, upper=False, line=None):
    """Consistent fallback caption: dark band + words popping in as sung."""
    line = line or ctx.line
    if band:
        d = ImageDraw.Draw(ctx.img, "RGBA")
        d.rectangle((0, y - size * 1.1, W, y + size * 0.75), fill=(8, 6, 14, 170))
    pop_words(ctx.img, line, ctx.t, W / 2, y, fname, size, max_w, color=color, active=hot,
              upper=upper)
