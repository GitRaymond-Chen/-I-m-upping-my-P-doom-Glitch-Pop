"""Frame post-processing (numpy/cv2 on HxWx3 uint8) and transition masks."""
import math

import cv2
import numpy as np
from PIL import Image

from .core import H, W, clamp, rng_for

_cache = {}


def _res(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def _scan_vignette():
    rows = np.ones(H, np.float32)
    rows[1::3] = 0.78  # subtle 3px scanline pattern
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2 * 0.6)
    vig = np.clip(1.15 - 0.42 * d ** 1.8, 0.35, 1.0)
    return rows[:, None], vig


def _crt_mask(scan, vignette):
    rows, vig = _res("scanvig", _scan_vignette)
    m = np.ones((H, W), np.float32)
    if scan:
        m = m * (1 - (1 - rows) * scan)
    if vignette:
        m = m * (1 - (1 - vig) * vignette)
    m8 = np.clip(m * 255, 0, 255).astype(np.uint8)
    return cv2.merge([m8, m8, m8])


def crt(arr, scan=1.0, vignette=1.0):
    key = ("crt", round(scan, 2), round(vignette, 2))
    mask = _res(key, lambda: _crt_mask(scan, vignette))
    return cv2.multiply(arr, mask, scale=1 / 255.0)


def rgb_split(arr, dx, dy=0):
    dx, dy = int(round(dx)), int(round(dy))
    if dx == 0 and dy == 0:
        return arr
    out = arr.copy()
    if dx > 0:
        out[:, :-dx, 0] = arr[:, dx:, 0]
        out[:, dx:, 2] = arr[:, :-dx, 2]
    elif dx < 0:
        out[:, -dx:, 0] = arr[:, :dx, 0]
        out[:, :dx, 2] = arr[:, -dx:, 2]
    if dy:
        out[..., 0] = np.roll(out[..., 0], -dy, axis=0)
        out[..., 2] = np.roll(out[..., 2], dy, axis=0)
    return out


N_GRAIN = 4


def _grain_pair(i, amount):
    rng = np.random.default_rng(7 + i)
    n = rng.normal(0, 5.5 * amount / 0.12, (H // 2, W // 2)).astype(np.float32)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_NEAREST)
    pos = np.clip(n, 0, 255).astype(np.uint8)
    neg = np.clip(-n, 0, 255).astype(np.uint8)
    return cv2.merge([pos, pos, pos]), cv2.merge([neg, neg, neg])


def grain(arr, amount, frame):
    if amount <= 0:
        return arr
    amount = round(amount * 25) / 25  # quantise so the cache stays small
    i = frame % N_GRAIN
    pos, neg = _res(("grain", i, amount), lambda: _grain_pair(i, amount))
    out = cv2.add(arr, pos)
    return cv2.subtract(out, neg)


def block_glitch(arr, intensity, seed, color=True):
    """Horizontal band displacement + channel swaps + stuck blocks."""
    if intensity <= 0.01:
        return arr
    rng = rng_for(seed, 911)
    out = arr.copy()
    n = int(3 + intensity * 22)
    for _ in range(n):
        h = int(rng.integers(4, int(10 + 90 * intensity)))
        y = int(rng.integers(0, H - h))
        shift = int(rng.normal(0, 60 + 260 * intensity))
        band = out[y:y + h]
        band = np.roll(band, shift, axis=1)
        if color and rng.random() < 0.35:
            band = band[..., rng.permutation(3)]
        out[y:y + h] = band
    # displaced blocks
    for _ in range(int(intensity * 8)):
        bw, bh = int(rng.integers(40, 420)), int(rng.integers(10, 120))
        sx, sy = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
        tx, ty = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
        out[ty:ty + bh, tx:tx + bw] = arr[sy:sy + bh, sx:sx + bw]
    return out


def warp(arr, zoom=1.0, angle=0.0, dx=0.0, dy=0.0, cx=W / 2, cy=H / 2):
    if abs(zoom - 1) < 1e-4 and abs(angle) < 1e-3 and abs(dx) < 0.5 and abs(dy) < 0.5:
        return arr
    M = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def skew(arr, sx=0.0, sy=0.0):
    M = np.float32([[1, sx, -sx * H / 2], [sy, 1, -sy * W / 2]])
    return cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def pixelate(arr, block):
    block = int(block)
    if block <= 1:
        return arr
    small = cv2.resize(arr, (max(1, W // block), max(1, H // block)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def _grid():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return xx, yy


def wave(arr, amp, freq, phase, vertical=False):
    if amp < 0.5:
        return arr
    xx, yy = _res("grid", _grid)
    if vertical:
        mx = xx
        my = yy + amp * np.sin(xx * freq + phase)
    else:
        mx = xx + amp * np.sin(yy * freq + phase)
        my = yy
    return cv2.remap(arr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def swirl(arr, strength, radius=700, cx=W / 2, cy=H / 2):
    if abs(strength) < 0.01:
        return arr
    xx, yy = _res("grid", _grid)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    a = strength * np.clip(1 - r / radius, 0, 1) ** 2
    ca, sa = np.cos(a), np.sin(a)
    mx = cx + dx * ca - dy * sa
    my = cy + dx * sa + dy * ca
    return cv2.remap(arr, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def barrel(arr, k):
    if abs(k) < 0.005:
        return arr

    def maps():
        xx, yy = _grid()
        nx, ny = (xx - W / 2) / (W / 2), (yy - H / 2) / (W / 2)
        r2 = nx * nx + ny * ny
        return nx, ny, r2

    nx, ny, r2 = _res("barrel", maps)
    f = 1 + k * r2
    mx = (nx * f) * (W / 2) + W / 2
    my = (ny * f) * (W / 2) + H / 2
    return cv2.remap(arr, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def hue_shift(arr, amount):
    """amount in [0,1) of a full hue turn."""
    if abs(amount) < 1e-3:
        return arr
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + int(amount * 180)) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def flash(arr, color, amount):
    if amount <= 0.003:
        return arr
    c = np.array(color, np.float32)[None, None, :]
    return np.clip(arr * (1 - amount) + c * amount, 0, 255).astype(np.uint8)


def invert(arr, amount=1.0):
    if amount <= 0:
        return arr
    if amount >= 1:
        return 255 - arr
    return np.clip(arr * (1 - amount) + (255 - arr.astype(np.float32)) * amount, 0,
                   255).astype(np.uint8)


def tint(arr, color, amount):
    """Multiply-tint towards a colour (duotone-ish)."""
    if amount <= 0:
        return arr
    lum = arr.mean(axis=2, keepdims=True) / 255.0
    col = np.array(color, np.float32)[None, None, :]
    tinted = lum * col
    return np.clip(arr * (1 - amount) + tinted * amount, 0, 255).astype(np.uint8)


def mix(a, b, m):
    """Blend arrays with mask m (HxW float in [0,1] or scalar)."""
    if np.isscalar(m):
        if m <= 0:
            return a
        if m >= 1:
            return b
        return cv2.addWeighted(a, 1 - m, b, m, 0)
    m3 = m[:, :, None]
    return (a * (1 - m3) + b * m3).astype(np.uint8)


# --- transition masks (1.0 = show incoming scene) -----------------------------------
def mask_slices(p, n=12, horizontal=True):
    m = np.zeros((H, W), np.float32)
    if horizontal:
        bh = H / n
        for i in range(n):
            q = clamp(p * 1.6 - (i % 2) * 0.3 - i / n * 0.3)
            w = int(W * q)
            y0, y1 = int(i * bh), int((i + 1) * bh)
            if i % 2 == 0:
                m[y0:y1, :w] = 1
            else:
                m[y0:y1, W - w:] = 1
    else:
        bw = W / n
        for i in range(n):
            q = clamp(p * 1.6 - (i % 2) * 0.3 - i / n * 0.3)
            h = int(H * q)
            x0, x1 = int(i * bw), int((i + 1) * bw)
            if i % 2 == 0:
                m[:h, x0:x1] = 1
            else:
                m[H - h:, x0:x1] = 1
    return m


def _dissolve_order(block):
    rng = np.random.default_rng(block)
    return rng.random((H // block + 1, W // block + 1)).astype(np.float32)


def mask_dissolve(p, block=40):
    order = _res(("dissolve", block), lambda: _dissolve_order(block))
    m = (order < p).astype(np.float32)
    return cv2.resize(m, (W // block * block + block, H // block * block + block),
                      interpolation=cv2.INTER_NEAREST)[:H, :W]


def mask_checker(p, n=8):
    cell = W // (n * 2)
    m = np.zeros((H, W), np.float32)
    for gy in range(0, H // cell + 1):
        for gx in range(0, W // cell + 1):
            delay = ((gx + gy) % 2) * 0.35 + (gx / (W / cell)) * 0.3
            q = clamp((p - delay) / 0.35)
            if q <= 0:
                continue
            s = int(cell * q / 2)
            cx, cy = gx * cell + cell // 2, gy * cell + cell // 2
            m[max(0, cy - s):cy + s, max(0, cx - s):cx + s] = 1
    return m


def mask_circle(p, cx=W / 2, cy=H / 2, soft=4):
    xx, yy = _res("grid", _grid)
    rmax = math.hypot(max(cx, W - cx), max(cy, H - cy))
    r = rmax * p
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip((r - d) / soft + 0.5, 0, 1)


def mask_from_image(img_l, cx, cy, scale):
    """Place an L-mode mask image (e.g. Clawd silhouette) scaled at (cx, cy)."""
    w, h = img_l.size
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    big = img_l.resize((nw, nh), Image.NEAREST)
    canvas = Image.new("L", (W, H), 0)
    canvas.paste(big, (int(cx - nw / 2), int(cy - nh / 2)))
    return np.asarray(canvas, np.float32) / 255.0


def mask_diagonal(p, soft=60):
    xx, yy = _res("grid", _grid)
    v = (xx + yy * 0.6) / (W + H * 0.6)
    edge = p * (1 + soft / W * 2) - soft / W
    return np.clip((edge - v) * W / soft + 0.5, 0, 1)
