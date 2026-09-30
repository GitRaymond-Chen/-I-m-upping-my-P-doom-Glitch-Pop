"""Procedural backgrounds. Functions draw onto a PIL RGB canvas (or return arrays)."""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .core import H, W, clamp, hsv, lerp, mix
from .gfx import cached_gradient, text_layer, paste


def gradient(img, top, bottom):
    img.paste(Image.fromarray(cached_gradient(tuple(top), tuple(bottom))))


def synth_grid(img, t, horizon=560, color=(255, 60, 200), speed=1.0, sky=((8, 4, 30), (70, 10, 80)),
               floor=(10, 4, 24), sun=True, sun_color=((255, 220, 90), (255, 60, 140)),
               lines_alpha=1.0, sun_y=None, sun_r=230):
    """Synthwave: gradient sky, striped sun, perspective grid floor scrolling toward us."""
    sky_img = cached_gradient(tuple(sky[0]), tuple(sky[1]))
    img.paste(Image.fromarray(sky_img[:horizon]), (0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, horizon, W, H), fill=floor)
    if sun:
        sy = sun_y if sun_y is not None else horizon - 60
        # sun: vertical gradient disc with horizontal cut stripes near the bottom
        top_c, bot_c = sun_color
        for yy in range(sy - sun_r, min(sy + sun_r, horizon)):
            dy = yy - sy
            half = math.sqrt(max(0, sun_r * sun_r - dy * dy))
            k = (yy - (sy - sun_r)) / (2 * sun_r)
            # stripes get thicker toward the bottom
            if dy > sun_r * 0.05:
                band = (yy - sy) % 26
                gap = 2 + int(10 * (dy / sun_r))
                if band < gap:
                    continue
            c = mix(top_c, bot_c, k)
            d.line((W / 2 - half, yy, W / 2 + half, yy), fill=(*c, 255))
    # grid
    col = (*color, int(255 * lines_alpha))
    vp_x = W / 2
    depth_n = 18
    off = (t * speed) % 1.0
    for i in range(depth_n):
        z = (i + 1 - off)
        if z <= 0.05:
            continue
        y = horizon + (H - horizon) * 0.9 / z
        if y > H:
            continue
        w_line = max(1, int(3 / z + 0.5))
        d.line((0, y, W, y), fill=col, width=w_line)
    for k in range(-16, 17):
        xb = vp_x + k * 260
        d.line((vp_x + k * 10, horizon, xb, H), fill=col, width=2)
    # horizon glow line
    d.line((0, horizon, W, horizon), fill=(*mix(color, (255, 255, 255), 0.4), 255), width=3)


@lru_cache(maxsize=4)
def _stars(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-1, 1, n)
    z = rng.uniform(0.05, 1.0, n)
    hue = rng.uniform(0, 1, n)
    return x, y, z, hue


def starfield(img, t, n=500, speed=0.25, warp=0.0, cx=W / 2, cy=H / 2, color=(230, 235, 255),
              seed=1, colorful=False, streak=1.0):
    x, y, z0, hue = _stars(n, seed)
    z = (z0 - t * speed) % 1.0 + 0.02
    d = ImageDraw.Draw(img, "RGBA")
    f = 900.0
    sx = cx + x * f / z * 0.9
    sy = cy + y * f / z * 0.9
    for i in range(n):
        if not (-50 < sx[i] < W + 50 and -50 < sy[i] < H + 50):
            continue
        b = clamp(1.2 - z[i])
        col = hsv(hue[i], 0.5, 1) if colorful else color
        a = int(255 * b)
        size = max(1, int(3.2 * (1 - z[i]) + 0.5))
        if warp > 0:
            z2 = z[i] + warp * 0.12 * streak
            px = cx + x[i] * f / z2 * 0.9
            py = cy + y[i] * f / z2 * 0.9
            d.line((px, py, sx[i], sy[i]), fill=(*col, a), width=size)
        else:
            d.rectangle((sx[i], sy[i], sx[i] + size, sy[i] + size), fill=(*col, a))


@lru_cache(maxsize=4)
def pcb_layer(seed=4, board=(10, 24, 60), trace=(217, 119, 87), center=(W // 2, H // 2)):
    """Static PCB image (RGB array) + list of trace polylines for animated pulses."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (W, H), board)
    d = ImageDraw.Draw(img)
    g = 40
    cols, rows = W // g + 1, H // g + 1
    used = np.zeros((rows, cols), bool)
    cx0, cy0 = center[0] // g, center[1] // g
    # keep the centre free for the chip
    used[cy0 - 3:cy0 + 4, cx0 - 4:cx0 + 5] = True
    traces = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    starts = []
    for k in range(70):
        ang = k / 70 * 2 * math.pi
        sx = int(cx0 + math.cos(ang) * 5)
        sy = int(cy0 + math.sin(ang) * 4)
        starts.append((sx, sy, ang))
    for sx, sy, ang in starts:
        x, y = sx, sy
        pts = [(x, y)]
        dx = int(round(math.cos(ang)))
        dy = int(round(math.sin(ang)))
        if dx == 0 and dy == 0:
            dx = 1
        for step in range(int(rng.integers(8, 40))):
            if rng.random() < 0.25:
                # change direction (never straight back)
                choices = [d_ for d_ in dirs if d_ != (-dx, -dy)]
                dx, dy = choices[int(rng.integers(0, len(choices)))]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < cols and 0 <= ny < rows) or used[ny, nx]:
                break
            used[ny, nx] = True
            x, y = nx, ny
            pts.append((x, y))
        if len(pts) > 3:
            poly = [(px * g, py * g) for px, py in pts]
            traces.append(poly)
    dim = mix(board, trace, 0.55)
    for poly in traces:
        d.line(poly, fill=dim, width=5, joint="curve")
        ex, ey = poly[-1]
        d.ellipse((ex - 9, ey - 9, ex + 9, ey + 9), outline=dim, width=4)
    # a few chips / pads
    for _ in range(26):
        x = int(rng.integers(0, W - 120))
        y = int(rng.integers(0, H - 80))
        if abs(x - center[0]) < 360 and abs(y - center[1]) < 260:
            continue
        w_, h_ = int(rng.integers(40, 120)), int(rng.integers(30, 70))
        d.rectangle((x, y, x + w_, y + h_), fill=(18, 20, 30), outline=(60, 70, 90), width=2)
        for px in range(x + 6, x + w_ - 4, 10):
            d.rectangle((px, y - 6, px + 4, y), fill=(190, 170, 110))
            d.rectangle((px, y + h_, px + 4, y + h_ + 6), fill=(190, 170, 110))
    return np.asarray(img).copy(), traces


def pcb(img, t, pulse_color=(255, 200, 120), n_pulses=2, speed=5.0, seed=4,
        center=(W // 2, H // 2)):
    arr, traces = pcb_layer(seed, center=tuple(center))
    img.paste(Image.fromarray(arr))
    d = ImageDraw.Draw(img, "RGBA")
    for ti, poly in enumerate(traces):
        L = len(poly) - 1
        for p in range(n_pulses):
            s = (t * speed + p * L / n_pulses + ti * 1.7) % (L + 6)
            if s > L:
                continue
            i = int(s)
            fr = s - i
            x = lerp(poly[i][0], poly[i + 1][0], fr) if i < L else poly[L][0]
            y = lerp(poly[i][1], poly[i + 1][1], fr) if i < L else poly[L][1]
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(*pulse_color, 255))
            d.ellipse((x - 14, y - 14, x + 14, y + 14), fill=(*pulse_color, 70))
    return traces


def sunburst(img, t, cx=W / 2, cy=H / 2, n=18, c1=(40, 10, 40), c2=(80, 10, 60), speed=0.2):
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, H), fill=c1)
    R = 2400
    rot = t * speed
    for k in range(n):
        a0 = rot + k * 2 * math.pi / n
        a1 = a0 + math.pi / n
        d.polygon([(cx, cy), (cx + R * math.cos(a0), cy + R * math.sin(a0)),
                   (cx + R * math.cos(a1), cy + R * math.sin(a1))], fill=c2)


def speedlines(img, t, cx=W / 2, cy=H / 2, color=(255, 255, 255), n=90, inner=300, alpha=160,
               seed=3, width=(2, 9)):
    d = ImageDraw.Draw(img, "RGBA")
    fr = int(t * 24)
    rng = np.random.default_rng(seed * 1000 + fr)
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        r0 = inner * rng.uniform(0.8, 1.6)
        r1 = 1600
        w0 = rng.uniform(*width)
        a2 = w0 / r1 * 8
        pts = [(cx + r0 * math.cos(a), cy + r0 * math.sin(a)),
               (cx + r1 * math.cos(a - a2), cy + r1 * math.sin(a - a2)),
               (cx + r1 * math.cos(a + a2), cy + r1 * math.sin(a + a2))]
        d.polygon(pts, fill=(*color, alpha))


def graph_paper(img, spacing=60, color=(30, 40, 60), major=5, major_color=(45, 60, 90),
                offset=(0, 0)):
    d = ImageDraw.Draw(img)
    ox, oy = offset
    for i, x in enumerate(range(int(ox % spacing) - spacing, W + spacing, spacing)):
        d.line((x, 0, x, H), fill=color, width=1)
    for i, y in enumerate(range(int(oy % spacing) - spacing, H + spacing, spacing)):
        d.line((0, y, W, y), fill=color, width=1)
    ms = spacing * major
    for x in range(int(ox % ms) - ms, W + ms, ms):
        d.line((x, 0, x, H), fill=major_color, width=2)
    for y in range(int(oy % ms) - ms, H + ms, ms):
        d.line((0, y, W, y), fill=major_color, width=2)


def matrix_rain(img, t, color=(57, 255, 136), chars="01", size=28, speed=9.0, seed=5,
                density=1.0, fname="mono", alpha=1.0):
    rng = np.random.default_rng(seed)
    cols = W // size
    heads = rng.uniform(0, 60, cols)
    speeds = rng.uniform(0.6, 1.4, cols) * speed
    lengths = rng.integers(6, 22, cols)
    for c in range(cols):
        if rng.random() > density:
            continue
        head = (heads[c] + t * speeds[c]) % (H / size + 30)
        for k in range(int(lengths[c])):
            row = int(head) - k
            if row < 0 or row * size > H:
                continue
            ch_i = int((c * 131 + row * 17 + int(t * 8) * (k == 0)) % len(chars))
            fade = 1 - k / lengths[c]
            col = (230, 255, 240) if k == 0 else mix((0, 0, 0), color, fade)
            li, ox, oy = text_layer(chars[ch_i], fname, size, col)
            paste(img, li, c * size + size / 2, row * size, alpha=alpha)


def plasma(t, hue0=0.0, scale=1.0, sat=0.9, val=0.9, speed=1.0):
    """Psychedelic sine plasma as an RGB array (computed at low res)."""
    w, h = W // 8, H // 8
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= w / 6.0 * scale
    yy /= h / 3.4 * scale
    ts = t * speed
    v = (np.sin(xx + ts) + np.sin(yy * 1.3 + ts * 1.7) + np.sin((xx + yy) * 0.7 + ts * 1.3) +
         np.sin(np.sqrt(xx * xx + yy * yy) * 1.5 - ts * 2.0)) / 4.0
    hue = (hue0 + v * 0.5 + 0.5) % 1.0
    hsv_img = np.stack([hue * 179, np.full_like(hue, sat * 255), np.full_like(hue, val * 255)],
                       axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_CUBIC)


def rain(img, t, color=(120, 160, 255), n=220, angle=0.12, speed=1400, length=38, seed=9,
         alpha=150):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-200, W + 200, n)
    y0 = rng.uniform(0, H, n)
    sp = rng.uniform(0.7, 1.3, n)
    d = ImageDraw.Draw(img, "RGBA")
    for i in range(n):
        y = (y0[i] + t * speed * sp[i]) % (H + 100) - 50
        x = xs[i] - y * angle
        d.line((x, y, x - length * angle, y - length), fill=(*color, alpha), width=2)


def rings(img, t, cx=W / 2, cy=H / 2, color=(255, 255, 255), n=12, speed=120, spacing=110,
          width=6, alpha=200):
    d = ImageDraw.Draw(img, "RGBA")
    off = (t * speed) % spacing
    for k in range(n):
        r = k * spacing + off
        a = int(alpha * clamp(1 - r / (n * spacing)))
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*color, a), width=width)


def checker(img, t, size=120, c1=(20, 16, 30), c2=(30, 24, 44), scroll=(40, 20)):
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, H), fill=c1)
    ox = (t * scroll[0]) % (2 * size)
    oy = (t * scroll[1]) % (2 * size)
    for gy in range(-2, H // size + 3):
        for gx in range(-2, W // size + 3):
            if (gx + gy) % 2 == 0:
                x = gx * size + ox
                y = gy * size + oy
                d.rectangle((x, y, x + size - 1, y + size - 1), fill=c2)
