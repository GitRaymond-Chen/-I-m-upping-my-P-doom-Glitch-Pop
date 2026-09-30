"""Clawd, the Claude Code mascot, drawn from the CLI startup banner:

     ▐▛███▜▌
    ▝▜█████▛▘
      ▘▘ ▝▝

Each quadrant block is one pixel twice as tall as it is wide, so in square units
Clawd is an 18x10 sprite: a 12x8 body, two 1x2 eye slits, 2x2 arms, four 1x2 legs.
Coordinates below are in those units, origin at the sprite's top-left; the
drawing anchor is the bottom-centre (9, 10), i.e. where the feet touch the floor.
"""
import math

from PIL import Image, ImageDraw

from .core import CLAWD, PINK, RED, CYAN, WHITE, clamp, hsv

EYE_DARK = (18, 16, 26)
SPRITE_W, SPRITE_H = 18, 10

BODY = (3, 0, 15, 8)
EYES_X = (5, 12)
LEGS_X = (4, 6, 11, 13)


def _arm_rects(side, state):
    """Arm rectangles for side -1 (left) / +1 (right) in units."""
    if state == "up":
        rects = [(1, 3, 3, 5), (0, 1, 2, 3)]
    elif state == "down":
        rects = [(1, 5, 3, 7)]
    elif state == "none":
        rects = []
    else:  # out
        rects = [(1, 4, 3, 6)]
    if side > 0:
        rects = [(SPRITE_W - x1, y0, SPRITE_W - x0, y1) for x0, y0, x1, y1 in rects]
    return rects


def clawd_rects(arms=("out", "out"), walk=None, jump_legs=False):
    """Body-colour rectangles (units) for the given pose."""
    rects = [BODY]
    rects += _arm_rects(-1, arms[0])
    rects += _arm_rects(1, arms[1])
    for i, lx in enumerate(LEGS_X):
        lift = 0.0
        if walk is not None:
            group = i % 2  # legs 0,2 vs 1,3 alternate
            ph = (walk + 0.5 * group) % 1.0
            lift = 1.0 if ph < 0.5 else 0.0
        if jump_legs:
            lift = 0.0
        rects.append((lx, 8, lx + 1, 10 - lift))
    return rects


def draw_clawd(canvas, x, y, u, color=CLAWD, eyes="normal", blink=0.0,
               arms=("out", "out"), walk=None, squash=0.0, look=0.0, alpha=1.0,
               eye_color=EYE_DARK, angle=0.0, t=0.0, outline=None):
    """Draw Clawd with feet centred at (x, y); `u` = size of one unit in px.

    squash > 0 squashes (landing), < 0 stretches (jumping).
    Returns the bounding box (x0, y0, x1, y1) in screen pixels.
    """
    if u <= 0.3 or alpha <= 0:
        return (x, y, x, y)
    ux = u * (1 + 0.5 * squash)
    uy = u * (1 - squash)
    if angle:
        # draw onto a temp layer, then rotate around the feet
        pad = 4
        wpx = int(SPRITE_W * ux + 2 * pad + 4 * u)
        hpx = int(SPRITE_H * uy + 2 * pad + 4 * u)
        layer = Image.new("RGBA", (wpx, hpx), (0, 0, 0, 0))
        draw_clawd(layer, wpx / 2, hpx - pad - 2 * u, u, color, eyes, blink, arms, walk,
                   squash, look, 1.0, eye_color, 0.0, t, outline)
        from .gfx import paste
        paste(canvas, layer, x, y, anchor="origin", origin=(wpx / 2, hpx - pad - 2 * u),
              alpha=alpha, angle=angle)
        return (x - SPRITE_W * ux / 2, y - SPRITE_H * uy, x + SPRITE_W * ux / 2, y)

    d = ImageDraw.Draw(canvas, "RGBA")
    a = int(255 * clamp(alpha))
    x0 = x - SPRITE_W / 2 * ux
    y0 = y - SPRITE_H * uy

    def R(rx0, ry0, rx1, ry1, col, aa=a):
        d.rectangle(
            (round(x0 + rx0 * ux), round(y0 + ry0 * uy),
             round(x0 + rx1 * ux) - 1, round(y0 + ry1 * uy) - 1),
            fill=(*col[:3], aa),
        )

    rects = clawd_rects(arms, walk)
    if outline:
        oc, ow = outline
        for rx0, ry0, rx1, ry1 in rects:
            R(rx0 - ow, ry0 - ow, rx1 + ow, ry1 + ow, oc)
    for r in rects:
        R(*r, color)
    draw_eyes(R, eyes, blink, look, eye_color, color, t)
    return (x0, y0, x0 + SPRITE_W * ux, y)


def draw_eyes(R, style, blink, look, eye_color, body_color, t):
    shift = clamp(look, -1, 1) * 0.5
    for i, ex in enumerate(EYES_X):
        ex = ex + shift
        side = -1 if i == 0 else 1
        if style == "none":
            continue
        if blink > 0.02 and style in ("normal", "red", "spark", "wide", "screen"):
            b = clamp(blink)
            top = 2 + 1.6 * b
            R(ex, top, ex + 1, 4, eye_color)
            continue
        if style == "normal":
            R(ex, 2, ex + 1, 4, eye_color)
        elif style == "wide":
            R(ex - 0.25, 1.75, ex + 1.25, 4.25, eye_color)
            R(ex - 0.25, 1.75, ex + 0.25, 2.25, WHITE)
        elif style == "happy":  # ^ ^
            R(ex - 0.5, 3.0, ex, 3.5, eye_color)
            R(ex, 2.5, ex + 1, 3.0, eye_color)
            R(ex + 1, 3.0, ex + 1.5, 3.5, eye_color)
        elif style == "closed":  # - -
            R(ex - 0.25, 3.0, ex + 1.25, 3.5, eye_color)
        elif style == "x":
            for (qx, qy) in ((-0.5, 2), (1, 2), (0.25, 2.75), (-0.5, 3.5), (1, 3.5)):
                R(ex + qx, qy, ex + qx + 0.5, qy + 0.5, eye_color)
        elif style == "heart":
            hc = PINK
            cells = [(1, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
                     (1, 2), (2, 2), (3, 2), (2, 3)]
            for (qx, qy) in cells:
                R(ex - 0.75 + qx * 0.5, 1.75 + qy * 0.5, ex - 0.25 + qx * 0.5,
                  2.25 + qy * 0.5, hc)
        elif style == "red":
            R(ex - 0.2, 1.8, ex + 1.2, 4.2, (90, 0, 0))
            R(ex, 2, ex + 1, 4, RED)
            R(ex + 0.25, 2.25, ex + 0.75, 2.75, (255, 200, 200))
        elif style == "spark":
            c = CYAN if int(t * 12) % 2 == 0 else WHITE
            R(ex, 2, ex + 1, 4, c)
            ph = (t * 3 + i * 0.37) % 1.0
            k = 0.5 + ph * 1.2
            R(ex + 0.25, 3 - k, ex + 0.75, 3 - k + 0.5, WHITE, 255)
            R(ex - k, 2.75, ex - k + 0.5, 3.25, WHITE, 255)
            R(ex + 0.5 + k, 2.75, ex + 1 + k, 3.25, WHITE, 255)
        elif style == "spiral":
            for k in range(4):
                c = hsv(t * 1.5 + k * 0.25, 1, 1)
                inset = k * 0.25
                R(ex - 0.5 + inset, 1.75 + inset, ex + 1.5 - inset, 4.25 - inset, c)
        elif style == "dollar":
            g = (60, 255, 120)
            for (qx, qy) in ((0, 0), (1, 0), (2, 0), (0, 1), (1, 2), (2, 3), (0, 4),
                             (1, 4), (2, 4), (1, -0.5), (1, 4.5)):
                R(ex - 0.25 + qx * 0.5, 1.75 + qy * 0.5, ex + 0.25 + qx * 0.5,
                  2.25 + qy * 0.5, g)
        elif style == "screen":
            R(ex, 2, ex + 1, 4, CYAN)
            R(ex, 2.5, ex + 1, 2.75, (0, 120, 140))
            R(ex, 3.25, ex + 1, 3.5, (0, 120, 140))
        elif style == "angry":
            R(ex, 2.5, ex + 1, 4, eye_color)
            inner = ex + (0.5 if side < 0 else 0)
            R(inner, 2.5, inner + 0.5, 3.0, body_color)
        elif style == "sad":
            R(ex, 2.5, ex + 1, 4, eye_color)
            outer = ex + (0 if side < 0 else 0.5)
            R(outer, 2.5, outer + 0.5, 3.0, body_color)
        else:
            R(ex, 2, ex + 1, 4, eye_color)


def blink_at(t, seed=0.0, period=3.1):
    """Natural-looking periodic blink amount for time t."""
    ph = ((t + seed) % period) / period
    dur = 0.12 / period
    if ph < dur:
        return math.sin(ph / dur * math.pi)
    return 0.0


def clawd_cells(arms=("out", "out")):
    """Unit cells (x, y, kind) making up Clawd: for pixel-particle effects."""
    cells = []
    for (x0, y0, x1, y1) in clawd_rects(arms):
        for yy in range(int(y0), int(math.ceil(y1))):
            for xx in range(int(x0), int(math.ceil(x1))):
                eye = (xx in EYES_X and yy in (2, 3))
                cells.append((xx, yy, "eye" if eye else "body"))
    return cells


def clawd_mask(u, arms=("out", "out")):
    """L-mode silhouette image of Clawd with unit size u (for iris transitions)."""
    wpx, hpx = int(SPRITE_W * u) + 2, int(SPRITE_H * u) + 2
    img = Image.new("L", (wpx, hpx), 0)
    d = ImageDraw.Draw(img)
    for (x0, y0, x1, y1) in clawd_rects(arms):
        d.rectangle((round(x0 * u), round(y0 * u), round(x1 * u) - 1, round(y1 * u) - 1),
                    fill=255)
    return img
