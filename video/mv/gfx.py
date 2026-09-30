"""Drawing helpers: fonts, cached text layers, compositing, pixel sprites, glow."""
import math
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import FONTS, H, W, clamp

FONT_FILES = {
    "pixel": "PressStart2P-Regular.ttf",
    "anton": "Anton-Regular.ttf",
    "glitch": "RubikGlitch-Regular.ttf",
    "grotesk": "SpaceGrotesk[wght].ttf",
    "mono": "JetBrainsMono[wght].ttf",
    "silkb": "Silkscreen-Bold.ttf",
    "bungee": "Bungee-Regular.ttf",
    "shade": "BungeeShade-Regular.ttf",
    "monoton": "Monoton-Regular.ttf",
    "type": "SpecialElite-Regular.ttf",
    "fraktur": "UnifrakturMaguntia-Book.ttf",
    "cjk": "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "jp": "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "dejavu": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}
WEIGHTS = {"grotesk": 700, "mono": 700}


@lru_cache(maxsize=256)
def font(name, size, weight=None):
    path = FONT_FILES[name]
    if not os.path.isabs(path):
        path = os.path.join(FONTS, path)
    f = ImageFont.truetype(path, int(size))
    wt = weight or WEIGHTS.get(name)
    if wt:
        try:
            f.set_variation_by_axes([wt])
        except OSError:
            pass
    return f


def text_width(s, fname, size, weight=None):
    return font(fname, size, weight).getlength(s)


@lru_cache(maxsize=4096)
def text_layer(s, fname, size, fill=(255, 255, 255, 255), stroke=0, stroke_fill=None,
               weight=None, pad=None):
    """RGBA image of `s`. Returns (img, ox, oy): the text's left-baseline origin in img."""
    f = font(fname, size, weight)
    if pad is None:
        pad = int(stroke + 4)
    if len(fill) == 3:
        fill = (*fill, 255)
    if stroke_fill is not None and len(stroke_fill) == 3:
        stroke_fill = (*stroke_fill, 255)
    x0, y0, x1, y1 = f.getbbox(s, anchor="ls", stroke_width=stroke)
    w, h = max(1, x1 - x0 + 2 * pad), max(1, y1 - y0 + 2 * pad)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ox, oy = pad - x0, pad - y0
    d.text((ox, oy), s, font=f, fill=fill, anchor="ls", stroke_width=stroke,
           stroke_fill=stroke_fill)
    return img, ox, oy


def paste(canvas, img, x, y, anchor="center", alpha=1.0, scale=1.0, angle=0.0,
          origin=None):
    """Composite RGBA `img` onto RGB/RGBA canvas.

    anchor: 'center' puts img center at (x, y); 'topleft' puts its corner there;
    'origin' puts the given `origin` point of img (e.g. text baseline) at (x, y).
    """
    if alpha <= 0.003 or scale <= 0.001:
        return
    iw, ih = img.size
    if anchor == "origin" and origin is not None:
        cx_img, cy_img = origin
    elif anchor == "topleft":
        cx_img, cy_img = 0, 0
    else:
        cx_img, cy_img = iw / 2, ih / 2
    if scale != 1.0:
        nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
        img = img.resize((nw, nh), Image.BICUBIC if scale > 0.5 else Image.LANCZOS)
        cx_img, cy_img = cx_img * nw / iw, cy_img * nh / ih
        iw, ih = nw, nh
    if angle:
        # rotate about the anchor point
        big = img.rotate(angle, resample=Image.BICUBIC, expand=True)
        bw, bh = big.size
        a = math.radians(angle)
        dx, dy = cx_img - iw / 2, cy_img - ih / 2
        rx = dx * math.cos(a) + dy * math.sin(a)
        ry = -dx * math.sin(a) + dy * math.cos(a)
        cx_img, cy_img = bw / 2 + rx, bh / 2 + ry
        img = big
    if alpha < 1.0:
        a = img.getchannel("A").point(lambda v: int(v * alpha))
        img = img.copy()
        img.putalpha(a)
    px, py = int(round(x - cx_img)), int(round(y - cy_img))
    canvas.paste(img, (px, py), img)


def draw_text(canvas, s, x, y, fname, size, fill=(255, 255, 255), anchor="center",
              alpha=1.0, scale=1.0, angle=0.0, stroke=0, stroke_fill=None, weight=None,
              shadow=None):
    """Draw text centred (anchor='center') or by left-baseline (anchor='ls')/'ms'/'rs'."""
    img, ox, oy = text_layer(s, fname, size, tuple(fill), stroke,
                             tuple(stroke_fill) if stroke_fill else None, weight)
    if anchor == "center":
        # centre on the visual box, not the baseline
        org = None
        mode = "center"
    else:
        mode = "origin"
        tw = img.size[0] - 2 * (stroke + 4)
        if anchor == "ms":
            org = (ox + tw / 2, oy)
        elif anchor == "rs":
            org = (ox + tw, oy)
        else:
            org = (ox, oy)
    if shadow:
        dx, dy, col = shadow
        simg, _, _ = text_layer(s, fname, size, tuple(col), stroke,
                                tuple(col) if stroke else None, weight)
        paste(canvas, simg, x + dx, y + dy, anchor=mode, origin=org, alpha=alpha,
              scale=scale, angle=angle)
    paste(canvas, img, x, y, anchor=mode, origin=org, alpha=alpha, scale=scale, angle=angle)
    return img.size


def layout_words(words, fname, size, max_width, line_gap=1.15, weight=None):
    """Greedy wrap. Returns list of (word, x, y_baseline) centred on (0, 0)."""
    f = font(fname, size, weight)
    space = f.getlength(" ")
    rows, cur, cur_w = [], [], 0.0
    for w in words:
        ww = f.getlength(w)
        add = ww if not cur else space + ww
        if cur and cur_w + add > max_width:
            rows.append((cur, cur_w))
            cur, cur_w = [w], ww
        else:
            cur.append(w)
            cur_w += add
    if cur:
        rows.append((cur, cur_w))
    asc, desc = f.getmetrics()
    lh = size * line_gap
    total_h = lh * (len(rows) - 1)
    out = []
    for r, (ws, rw) in enumerate(rows):
        x = -rw / 2
        y = -total_h / 2 + r * lh + (asc - desc) / 2 * 0.9
        for w in ws:
            out.append((w, x, y))
            x += f.getlength(w) + space
    return out


# --- pixel sprites -------------------------------------------------------------------
def parse_sprite(rows, palette):
    h = len(rows)
    w = max(len(r) for r in rows)
    arr = np.zeros((h, w, 4), np.uint8)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch in palette:
                c = palette[ch]
                arr[y, x] = (*c, 255) if len(c) == 3 else c
    return arr


@lru_cache(maxsize=1024)
def _sprite_scaled(key, px, tint):
    from .sprites import SPRITES
    arr = SPRITES[key]
    if tint is not None:
        arr = arr.copy()
        m = arr[..., 3] > 0
        arr[m, :3] = tint
    img = Image.fromarray(arr, "RGBA")
    return img.resize((arr.shape[1] * px, arr.shape[0] * px), Image.NEAREST)


def sprite(canvas, key, x, y, px=8, alpha=1.0, angle=0.0, tint=None, anchor="center",
           scale=1.0):
    img = _sprite_scaled(key, max(1, int(px)), tuple(tint) if tint else None)
    paste(canvas, img, x, y, anchor=anchor, alpha=alpha, angle=angle, scale=scale)
    return img.size


# --- numpy / cv2 helpers ----------------------------------------------------------------
def to_np(img):
    return np.asarray(img, dtype=np.uint8).copy()


def to_pil(arr):
    return Image.fromarray(arr)


def glow_layer(arr, radius=24, strength=1.0, threshold=0):
    """Additive bloom from bright parts of an RGB uint8 array (returns uint8).

    `threshold` is on luminance, so saturated mid-tones (Clawd orange) don't bloom
    while whites, cyans and hot highlights do.
    """
    small = cv2.resize(arr, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    if threshold:
        lum = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        small = cv2.bitwise_and(small, small, mask=(lum > threshold).astype(np.uint8))
    r = max(1, radius // 4) | 1
    blur = cv2.GaussianBlur(small, (0, 0), r / 2)
    blur = cv2.resize(blur, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(arr, 1.0, blur, float(strength), 0)


def new_layer(size=(W, H)):
    return Image.new("RGBA", size, (0, 0, 0, 0))


def vertical_gradient(top, bottom, size=(W, H)):
    w, h = size
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    a = np.array(top, np.float32)[None, None, :]
    b = np.array(bottom, np.float32)[None, None, :]
    col = a + (b - a) * t
    return np.broadcast_to(col, (h, w, 3)).astype(np.uint8)


@lru_cache(maxsize=32)
def cached_gradient(top, bottom):
    return vertical_gradient(top, bottom)


def radial_mask(cx=0.5, cy=0.5, inner=0.2, outer=0.9, size=(W, H)):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - cx * w) / (w / 2)) ** 2 + ((yy - cy * h) / (w / 2)) ** 2)
    return np.clip((outer - d) / (outer - inner), 0, 1)


def fill_alpha_rect(canvas, box, color, alpha):
    d = ImageDraw.Draw(canvas, "RGBA")
    d.rectangle(box, fill=(*color[:3], int(255 * clamp(alpha))))


def rounded_box(canvas, box, radius, fill=None, outline=None, width=1, alpha=1.0):
    d = ImageDraw.Draw(canvas, "RGBA")
    f = (*fill[:3], int(255 * alpha)) if fill else None
    o = (*outline[:3], int(255 * alpha)) if outline else None
    d.rounded_rectangle(box, radius=radius, fill=f, outline=o, width=width)
