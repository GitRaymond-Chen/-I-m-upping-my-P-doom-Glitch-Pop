"""Kinetic typography helpers driven by word timings."""
import math

from .core import CREAM, clamp, ease_out, ease_out_back, hash01, hsv, mix
from .gfx import draw_text, font, layout_words, paste, text_layer

GLYPHS = "!<>-_\\/[]{}=+*^?#$%&@01X"


def word_times(line):
    return [w["start"] for w in line["words"]]


def display_words(line):
    return [w["w"] for w in line["words"]]


def pop_words(img, line, t, cx, cy, fname, size, max_width, color=CREAM, active=None,
              stroke=0, stroke_fill=None, shadow=None, pop=1.35, lead=0.05,
              show_future=0.0, upper=False, weight=None, line_gap=1.12, alpha=1.0,
              jitter=0.0, seed=0, colors=None, rise=0):
    """Words appear as they're sung (scale pop); optional dim preview of future words.

    colors: optional per-word colour overrides {index: rgb}.
    Returns [(word, x, y_baseline, width)] in screen coordinates.
    """
    words = display_words(line)
    if upper:
        words = [w.upper() for w in words]
    lay = layout_words(words, fname, size, max_width, line_gap=line_gap, weight=weight)
    times = word_times(line)
    f = font(fname, size, weight)
    out = []
    for i, (w, x, y) in enumerate(lay):
        dt = t + lead - times[i]
        ww = f.getlength(w)
        X, Y = cx + x, cy + y
        col = (colors or {}).get(i, color)
        if dt < 0:
            if show_future > 0:
                draw_text(img, w, X, Y, fname, size, col, anchor="ls", alpha=show_future * alpha,
                          weight=weight)
            out.append((w, X, Y, ww))
            continue
        k = ease_out(dt / 0.14)
        sc = 1 + (pop - 1) * (1 - k)
        if active is not None and i == _active_index(times, t + lead):
            col = active
        jx = jy = 0
        if jitter:
            jx = (hash01(seed, i, int(t * 24)) - 0.5) * jitter
            jy = (hash01(seed, i, int(t * 24), 9) - 0.5) * jitter
        # scale about the word centre
        img_w, ox, oy = text_layer(w, fname, size, tuple(col), stroke,
                                   tuple(stroke_fill) if stroke_fill else None, weight)
        cxw = X + ww / 2 + jx
        cyw = Y - size * 0.35 + jy + rise * (1 - k)
        if shadow:
            sdx, sdy, scol = shadow
            s_img, _, _ = text_layer(w, fname, size, tuple(scol), stroke,
                                     tuple(scol) if stroke else None, weight)
            paste(img, s_img, cxw + sdx, cyw + sdy, anchor="origin",
                  origin=(ox + ww / 2, oy - size * 0.35), scale=sc, alpha=k * alpha)
        paste(img, img_w, cxw, cyw, anchor="origin", origin=(ox + ww / 2, oy - size * 0.35),
              scale=sc, alpha=min(1.0, k * 1.5) * alpha)
        out.append((w, X, Y, ww))
    return out


def _active_index(times, t):
    idx = -1
    for i, s in enumerate(times):
        if t >= s:
            idx = i
    return idx


def karaoke(img, line, t, cx, cy, fname, size, max_width, base=(90, 90, 110), lit=CREAM,
            hot=None, upper=False, weight=None, stroke=0, stroke_fill=None, line_gap=1.12):
    """Whole line visible; words light up as sung (current word in `hot`)."""
    words = display_words(line)
    if upper:
        words = [w.upper() for w in words]
    lay = layout_words(words, fname, size, max_width, line_gap=line_gap, weight=weight)
    times = word_times(line)
    cur = _active_index(times, t + 0.03)
    for i, (w, x, y) in enumerate(lay):
        col = base if i > cur else (hot if (hot and i == cur) else lit)
        draw_text(img, w, cx + x, cy + y, fname, size, col, anchor="ls", weight=weight,
                  stroke=stroke, stroke_fill=stroke_fill)
    return lay


def typewriter(img, text, t, t0, t1, x, y, fname, size, color=CREAM, cursor=True,
               anchor="ls", weight=None, cursor_color=None, glitch=0.0, seed=0):
    """Reveal characters of `text` linearly between t0 and t1 (left-baseline at x,y)."""
    n = len(text)
    k = int(clamp((t - t0) / max(1e-3, t1 - t0)) * n + 0.0001) if t >= t0 else 0
    shown = text[:k]
    if glitch and 0 < k < n:
        shown = shown[:-1] + GLYPHS[int(hash01(seed, int(t * 30)) * len(GLYPHS))]
    if shown:
        draw_text(img, shown, x, y, fname, size, color, anchor="ls", weight=weight)
    if cursor and (int(t * 3) % 2 == 0 or k < n):
        cw = font(fname, size, weight).getlength(shown) if shown else 0
        from PIL import ImageDraw
        d = ImageDraw.Draw(img, "RGBA")
        cc = cursor_color or color
        d.rectangle((x + cw + 4, y - size * 0.78, x + cw + 4 + size * 0.5, y + size * 0.08),
                    fill=(*cc, 230))
    return k >= n


def scramble(text, t, t_reveal, dur=0.35, seed=0):
    """Characters resolve from random glyphs to `text` over `dur` after t_reveal."""
    if t < t_reveal:
        return ""
    k = clamp((t - t_reveal) / dur)
    out = []
    for i, ch in enumerate(text):
        if ch == " ":
            out.append(" ")
            continue
        settle = hash01(seed, i) * 0.8
        if k >= settle + 0.2 or k >= 1:
            out.append(ch)
        else:
            out.append(GLYPHS[int(hash01(seed, i, int(t * 30)) * len(GLYPHS))])
    return "".join(out)


def char_positions(text, fname, size, weight=None):
    """x offsets of each character (from kerning-aware prefix widths) and total width."""
    f = font(fname, size, weight)
    xs = [f.getlength(text[:i]) for i in range(len(text))]
    return xs, f.getlength(text)


def per_char(img, text, x, y, fname, size, color_fn, offset_fn, anchor="ms", weight=None,
             stroke=0, stroke_fill=None, alpha_fn=None, scale_fn=None, angle_fn=None):
    """Draw each character with its own colour/offset/alpha/scale/angle.

    anchor 'ms' centres the string horizontally on x, baseline y; 'ls' left-aligns.
    """
    xs, total = char_positions(text, fname, size, weight)
    x0 = x - total / 2 if anchor == "ms" else x
    f = font(fname, size, weight)
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        dx, dy = offset_fn(i)
        col = color_fn(i)
        a = alpha_fn(i) if alpha_fn else 1.0
        sc = scale_fn(i) if scale_fn else 1.0
        ang = angle_fn(i) if angle_fn else 0.0
        if a <= 0:
            continue
        cimg, ox, oy = text_layer(ch, fname, size, tuple(col), stroke,
                                  tuple(stroke_fill) if stroke_fill else None, weight)
        cw = f.getlength(ch)
        paste(img, cimg, x0 + xs[i] + cw / 2 + dx, y - size * 0.35 + dy, anchor="origin",
              origin=(ox + cw / 2, oy - size * 0.35), alpha=a, scale=sc, angle=ang)
    return x0, total


def shaky(img, text, t, x, y, fname, size, color, amp=6, seed=0, anchor="ms", weight=None,
          stroke=0, stroke_fill=None):
    fr = int(t * 20)
    return per_char(img, text, x, y, fname, size, lambda i: color,
                    lambda i: ((hash01(seed, i, fr) - 0.5) * 2 * amp,
                               (hash01(seed, i, fr, 3) - 0.5) * 2 * amp),
                    anchor=anchor, weight=weight, stroke=stroke, stroke_fill=stroke_fill)


def wavy(img, text, t, x, y, fname, size, amp=18, freq=0.55, speed=6, rainbow=False,
         color=CREAM, anchor="ms", weight=None, stroke=0, stroke_fill=None):
    return per_char(img, text, x, y, fname, size,
                    (lambda i: hsv(t * 0.6 + i * 0.07, 0.85, 1)) if rainbow else (lambda i: color),
                    lambda i: (0, math.sin(t * speed + i * freq) * amp), anchor=anchor,
                    weight=weight, stroke=stroke, stroke_fill=stroke_fill)


def dropping(img, text, t, t0, x, y, fname, size, color, stagger=0.035, fall=500,
             anchor="ms", weight=None, stroke=0, stroke_fill=None):
    """Letters fall into place with a bounce, starting at t0."""
    def off(i):
        k = clamp((t - t0 - i * stagger) / 0.35)
        # bounce ease
        if k < 1:
            yb = (1 - ease_out_back(k, 2.2)) * -fall
        else:
            yb = 0
        return (0, yb)

    def al(i):
        return 1.0 if t >= t0 + i * stagger else 0.0

    return per_char(img, text, x, y, fname, size, lambda i: color, off, anchor=anchor,
                    weight=weight, alpha_fn=al, stroke=stroke, stroke_fill=stroke_fill)


def shuffled(img, text, t, t0, t1, x, y, fname, size, color, seed=0, anchor="ms",
             weight=None, hot=None):
    """Letters swap places randomly and settle into `text` between t0 and t1."""
    n = len(text)
    xs, total = char_positions(text, fname, size, weight)
    x0 = x - total / 2 if anchor == "ms" else x
    k = clamp((t - t0) / max(1e-3, t1 - t0))
    step = int(t * 10)
    import random
    rnd = random.Random(int(hash01(seed, step) * 1e9))
    fins = [k >= 1 or k > 0.35 + hash01(seed, i, 5) * 0.6 for i in range(n)]
    # unsettled letters swap among the unsettled slots only (no gaps/overlaps)
    loose = [i for i in range(n) if not fins[i]]
    shuffled_slots = loose[:]
    rnd.shuffle(shuffled_slots)
    slot = dict(zip(loose, shuffled_slots))
    f = font(fname, size, weight)
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        fin = fins[i]
        tx = xs[i] if fin else xs[slot[i]]
        col = color if fin or hot is None else hot
        cimg, ox, oy = text_layer(ch, fname, size, tuple(col), 0, None, weight)
        cw = f.getlength(ch)
        paste(img, cimg, x0 + tx + cw / 2, y - size * 0.35, anchor="origin",
              origin=(ox + cw / 2, oy - size * 0.35))
    return x0, total


def glitch_word(img, text, t, x, y, fname, size, color, amount=1.0, seed=0, weight=None):
    """Draw text with RGB-offset ghost copies that jitter (cheap per-text glitch)."""
    fr = int(t * 24)
    for k, col in enumerate(((255, 40, 90), (40, 230, 255))):
        dx = (hash01(seed, fr, k) - 0.5) * 22 * amount
        dy = (hash01(seed, fr, k, 7) - 0.5) * 8 * amount
        draw_text(img, text, x + dx, y + dy, fname, size, col, alpha=0.75, weight=weight)
    draw_text(img, text, x, y, fname, size, color, weight=weight)


def stamp(img, text, t, t_hit, x, y, fname, size, color, angle=-8, stroke=0,
          stroke_fill=None, from_scale=3.0, dur=0.12, weight=None):
    """Big word slamming down at t_hit (returns 0..1 impact intensity)."""
    if t < t_hit:
        return 0.0
    k = clamp((t - t_hit) / dur)
    sc = from_scale + (1 - from_scale) * ease_out(k)
    draw_text(img, text, x, y, fname, size, color, alpha=clamp(k * 2.5), scale=sc,
              angle=angle, stroke=stroke, stroke_fill=stroke_fill, weight=weight)
    return max(0.0, 1 - (t - t_hit - dur) / 0.25) if t >= t_hit + dur else 0.0


def colorize_active(times, t, base, hot):
    cur = _active_index(times, t)
    return {i: (hot if i == cur else base) for i in range(len(times))}


def fade_color(c, k):
    return mix((0, 0, 0), c, k)
