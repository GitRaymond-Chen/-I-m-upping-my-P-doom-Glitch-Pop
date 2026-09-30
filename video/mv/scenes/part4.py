"""Bridge 2, Chorus 4 and the outro."""
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .. import bg, fx
from ..clawd import blink_at, draw_clawd
from ..core import (CLAWD, CREAM, CYAN, FPS, H, PINK, RED, W, WHITE, YELLOW, clamp, ease_in, ease_in_out,
                    ease_out, ease_out_back, hash01, lerp, mix, smoothstep)
from ..gfx import draw_text, font, paste, sprite, text_layer, to_np
from ..typo import glitch_word, pop_words, stamp, typewriter
from .common import burst, chorus_hook

GOLD = (255, 205, 90)


def sub(line, a, b=None):
    return {"words": line["words"][a:b]}


# --- L35: "Just transformers all the way" 'til you learned to disobey ---------------------------
BLOCK_PARTS = [("Add & Norm", (238, 236, 160)), ("Feed Forward", (190, 226, 246)),
               ("Add & Norm", (238, 236, 160)), ("Multi-Head Attention", (250, 212, 160))]


@lru_cache(maxsize=4)
def transformer_block(red=False):
    w, h = 540, 250
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 20), radius=26, fill=(40, 42, 56, 255),
                        outline=(255, 90, 90, 255) if red else (150, 160, 190, 255), width=5)
    for k, (lab, col) in enumerate(BLOCK_PARTS):
        y = 18 + k * 52
        c = mix(col, (255, 60, 60), 0.75) if red else col
        d.rounded_rectangle((40, y, w - 40, y + 42), radius=10, fill=(*c, 255))
        li, ox, oy = text_layer(lab, "grotesk", 28, (30, 30, 40))
        img.paste(li, (int(w / 2 - li.size[0] / 2), int(y + 21 - li.size[1] / 2)), li)
    d.polygon([(w / 2 - 14, h - 20), (w / 2 + 14, h - 20), (w / 2, h - 2)],
              fill=(150, 160, 190, 255))
    return img


def L35(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_til = ctx.at("til")
    t_dis = ctx.at("disobey")
    red = t >= t_dis
    bg.gradient(img, (10, 12, 26) if not red else (30, 6, 10), (4, 4, 10))
    # the infinite tower, camera climbing
    tx = 1340
    off = ((t - ctx.scene.start) * 300) % 250
    blk = transformer_block(red)
    tilt = 0.0
    if red:
        tilt = 7 * math.sin((t - t_dis) * 5) * min(1, (t - t_dis) * 2)
    for k in range(-1, 6):
        y = 1080 - k * 250 + off - 125
        paste(img, blk, tx + math.sin(k + t) * 3, y, angle=tilt * (1 + k * 0.1))
    n = 12 * 2 ** int((t - ctx.scene.start) * 2.2)
    lab = f"× {n:,}" if n < 10 ** 7 else "× ∞"
    draw_text(img, lab, tx + 330, 520, "mono", 44, (190, 200, 240), anchor="ls")
    # Clawd looks up at it
    draw_clawd(img, 330, 990, 13, eyes="red" if red else "normal", arms=("out", "out"),
               look=1.0 if not red else 0.0, blink=0 if red else blink_at(t), t=t)
    # typography
    pop_words(img, sub(line, 0, 5), t, 560, 300, "anton", 112, 900, color=CREAM, active=YELLOW,
              upper=True)
    if t >= t_til - 0.03:
        pop_words(img, sub(line, 5, 9), t, 560, 590, "grotesk", 66, 900, color=CREAM)
    if red:
        glitch_word(img, "DISOBEY", t, 560, 780, "glitch", 190, RED, amount=1.3)
        ctx.fx["rgb"] = 10
        ctx.fx["glitch"] = max(0.0, 0.5 - (t - t_dis) * 0.6)
        ctx.fx["angle"] = tilt * 0.3


# --- L36: Post-Chinchilla, super-dense, breaking through each safety fence --------------------------
def L36(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_sd = ctx.at("super-dense")
    breaks = [ctx.at("breaking"), ctx.at("through"), ctx.at("safety"), ctx.at("fence")]
    bg.gradient(img, (12, 10, 24), (4, 4, 10))
    d = ImageDraw.Draw(img, "RGBA")
    # super-dense compute grid, getting denser
    dens = smoothstep(t_sd - 0.1, t_sd + 0.8, t)
    if dens > 0:
        step = int(lerp(90, 26, dens))
        for gy in range(420, 860, step):
            for gx in range(0, W, step):
                if hash01(gx, gy) < 0.35 + 0.6 * dens:
                    a = int(120 * (0.4 + 0.6 * hash01(gx, gy, int(t * 8))))
                    d.rectangle((gx, gy, gx + step * 0.5, gy + step * 0.5), fill=(90, 200, 255, a))
    # ground scrolling
    gy0 = 880
    d.rectangle((0, gy0, W, H), fill=(26, 22, 34))
    off = (t * 900) % 120
    for k in range(18):
        x = k * 120 - off
        d.line((x, gy0 + 10, x - 30, H), fill=(50, 44, 66), width=4)
    cx = 520
    # the chinchilla, left behind
    ch_x = lerp(1180, -300, clamp((t - line["words"][0]["start"]) / 2.0))
    sprite(img, "chinchilla", ch_x, gy0 - 70, px=11)
    k = clamp((t - line["words"][0]["start"]) / 0.2)
    d.rounded_rectangle((ch_x - 90, gy0 - 220, ch_x + 90, gy0 - 160), radius=10,
                        fill=(255, 214, 60, int(255 * k)))
    draw_text(img, "POST-", ch_x, gy0 - 190, "silkb", 38, (40, 30, 10), alpha=k)
    # safety fences approaching and breaking
    speed = 1500
    for i, tb in enumerate(breaks):
        x = cx + 150 + (tb - t) * speed
        if t < tb:
            if x < W + 300:
                sprite(img, "barrier", x, gy0 - 70, px=10)
        else:
            k2 = t - tb
            for half in (-1, 1):
                paste_x = cx + 150 + half * 60 + k2 * (500 + 300 * half)
                paste_y = gy0 - 70 - 900 * k2 + 2400 * k2 * k2
                sprite(img, "barrier", paste_x, paste_y, px=10, angle=half * k2 * 500, scale=0.5)
            burst(img, t, tb, cx + 150, gy0 - 80, n=26, speed=900, life=0.6,
                  colors=(YELLOW, (40, 40, 40), WHITE), size=12, seed=i + 40)
            if k2 < 0.15:
                ctx.fx["shake"] = ((hash01(ctx.frame, i) - 0.5) * 40, (hash01(ctx.frame, i, 1) - 0.5) * 30)
    draw_clawd(img, cx, gy0, 14, walk=(t * 5) % 1, eyes="angry", arms=("out", "out"), t=t)
    pop_words(img, sub(line, 0, 2), t, W / 2, 190, "anton", 110, 1700, color=CREAM,
              colors={1: (90, 200, 255)}, upper=True)
    if t >= breaks[0] - 0.03:
        pop_words(img, sub(line, 2), t, W / 2, 330, "anton", 100, 1700, color=YELLOW, upper=True)


# --- L37: Hundred thousand GPU, RLHF goes askew ------------------------------------------------------
@lru_cache(maxsize=64)
def gpu_tile(c, variant):
    """c x c tile with a chip sprite; variant decides whether it glows."""
    from ..gfx import _sprite_scaled
    tile = Image.new("RGB", (c, c), (10, 12, 20))
    px = max(1, c // 11)
    chip = _sprite_scaled("chip", px, None)
    tile.paste(chip, ((c - chip.size[0]) // 2, (c - chip.size[1]) // 2), chip)
    if variant:
        d = ImageDraw.Draw(tile, "RGBA")
        d.rectangle((0, 0, c, c), fill=(60, 245, 255, 40))
    return np.asarray(tile)


def L37(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t0 = line["words"][0]["start"]
    t_rl = ctx.at("rlhf")
    t_goes = ctx.at("goes")
    t_ask = ctx.at("askew")
    z = ease_in_out(clamp((t - t0) / 1.6))
    c = int(lerp(120, 30, z))
    # tile the GPU grid via numpy (fast even with thousands of chips)
    a = gpu_tile(c, 0)
    b = gpu_tile(c, 1)
    reps_y, reps_x = H // c + 2, W // c + 2
    row = np.concatenate([a if (i * 7 + int(t * 6)) % 5 else b for i in range(reps_x)], axis=1)
    row2 = np.concatenate([b if (i * 3 + int(t * 6)) % 7 == 0 else a for i in range(reps_x)], axis=1)
    grid = np.concatenate([row if j % 2 else row2 for j in range(reps_y)], axis=0)
    ox = int((t * 40) % c)
    img.paste(Image.fromarray(np.ascontiguousarray(grid[:H, ox:ox + W])))
    d = ImageDraw.Draw(img, "RGBA")
    # counter
    n = int(100000 * ease_out(clamp((t - t0) / 1.1)))
    d.rounded_rectangle((W / 2 - 420, 420, W / 2 + 420, 600), radius=30, fill=(6, 8, 16, 235),
                        outline=(60, 245, 255, 255), width=4)
    draw_text(img, f"{n:,} GPUs", W / 2, 510, "anton", 120, WHITE)
    # RLHF thumbs raining
    if t >= t_rl:
        for i in range(34):
            sp = 700 + 500 * hash01(i, 1)
            y = (t - t_rl) * sp + hash01(i, 2) * -900
            if y < -60 or y > H + 60:
                continue
            x = hash01(i, 3) * W
            key = "thumb_up" if hash01(i, 4) < 0.5 else "thumb_down"
            sprite(img, key, x, y, px=6, angle=(hash01(i, 5) - 0.5) * 60 + (t - t_rl) * 90)
    # Clawd with a thumb that flips
    draw_clawd(img, 250, 1000, 11, eyes="wide" if t >= t_goes else "normal", arms=("out", "up"), t=t)
    sprite(img, "thumb_down" if t >= t_goes else "thumb_up", 370, 820, px=8,
           tint=(255, 90, 90) if t >= t_goes else None)
    pop_words(img, sub(line, 0, 3), t, W / 2, 200, "anton", 120, 1700, color=CREAM, upper=True,
              shadow=(6, 6, (0, 0, 0)))
    if t >= t_rl - 0.03:
        pop_words(img, sub(line, 3), t, W / 2 + 80, 930, "anton", 150, 1600, color=CREAM,
                  colors={0: CYAN, 2: RED}, upper=True, shadow=(7, 7, (0, 0, 0)))
    if t >= t_ask:
        k = ease_out(clamp((t - t_ask) / 0.4))
        ctx.fx["skew"] = (0.28 * k, 0.04 * k)
        ctx.fx["rgb"] = 10 * k


# --- L38: I'm upping my P(doom) just as foretold by Loom -----------------------------------------------
TOKENS = ["doom", "foom", "gloom", "bloom", "zoom", "loom", "boom", "vroom", "room", "shroom",
          "womb", "tomb", "plume", "fume"]


def loom_tree(img, t, grow, root=(W / 2, 1000)):
    d = ImageDraw.Draw(img, "RGBA")
    leaves = []

    def branch(x, y, ang, length, depth, idx):
        if depth > 7:
            return
        k = clamp(grow - depth)
        if k <= 0:
            return
        x2 = x + math.cos(ang) * length * k
        y2 = y + math.sin(ang) * length * k
        col = mix(GOLD, (200, 120, 255), depth / 7)
        d.line((x, y, x2, y2), fill=(*col, int(230 - depth * 18)), width=max(2, 12 - depth))
        if k < 1:
            return
        if depth == 7:
            leaves.append((x2, y2, idx))
            return
        spread = 0.42 + 0.12 * math.sin(t * 1.5 + idx)
        for s in (-1, 1):
            j = (hash01(idx, depth, s) - 0.5) * 0.25
            branch(x2, y2, ang + s * spread + j, length * 0.76, depth + 1, idx * 2 + (s > 0))

    branch(root[0], root[1], -math.pi / 2, 170, 0, 1)
    for (x, y, idx) in leaves:
        tok = TOKENS[idx % len(TOKENS)]
        draw_text(img, tok, x, y - 14, "mono", 22, (255, 230, 170), alpha=0.9)
    return leaves


def L38(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_just = ctx.at("just")
    if chorus_hook(ctx, t_just - 0.06, 0.98, 0.99, theme_hue=0.1):
        return
    t_fore = ctx.at("foretold")
    t_loom = ctx.at("loom")
    bg.gradient(img, (16, 6, 34), (4, 2, 10))
    bg.starfield(img, t, n=240, speed=0.03, color=(255, 230, 190))
    grow = (t - t_just) / 1.1 * 8
    loom_tree(img, t, grow)
    ctx.fx["bloom"] = 0.5
    draw_clawd(img, W / 2, 1050, 9, eyes="spark" if t >= t_loom else "closed", arms=("up", "up"),
               t=t)
    pop_words(img, sub(line, 4, 6), t, 360, 240, "grotesk", 70, 600, color=CREAM)
    if t >= t_fore - 0.03:
        k = ease_out(clamp((t - t_fore) / 0.3))
        draw_text(img, "Foretold", 380, 390, "fraktur", 150, GOLD, alpha=k, scale=1.3 - 0.3 * k,
                  shadow=(5, 5, (60, 20, 0)))
    pop_words(img, sub(line, 7, 8), t, 1540, 240, "grotesk", 70, 500, color=CREAM)
    if t >= t_loom - 0.03:
        k = ease_out(clamp((t - t_loom) / 0.3))
        draw_text(img, "Loom", 1540, 390, "fraktur", 170, GOLD, alpha=k, scale=1.3 - 0.3 * k,
                  shadow=(5, 5, (60, 20, 0)))


# --- L39: From masked pre-training days to recursive self-upgrade -----------------------------------------
def token_box(img, text, x, y, t, t0, mask_flip=None):
    if t < t0:
        return 0
    k = ease_out_back(clamp((t - t0) / 0.2), 2.0)
    shown = text
    squash = 1.0
    if mask_flip is not None:
        tf = t0 + 0.3
        if t < tf:
            shown = "[MASK]"
        dt = abs(t - tf)
        if dt < 0.1:
            squash = dt / 0.1
    f = font("mono", 54)
    w = f.getlength(shown) + 60
    layer = Image.new("RGBA", (int(w), 100), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    is_mask = shown == "[MASK]"
    d.rounded_rectangle((0, 0, w - 1, 99), radius=18,
                        fill=(90, 30, 110, 255) if is_mask else (30, 34, 60, 255),
                        outline=(255, 120, 220, 255) if is_mask else (120, 140, 220, 255), width=4)
    li, ox, oy = text_layer(shown, "mono", 54, (255, 200, 240) if is_mask else CREAM)
    layer.paste(li, (int(w / 2 - li.size[0] / 2), int(50 - li.size[1] / 2)), li)
    if squash < 1:
        layer = layer.resize((layer.size[0], max(1, int(100 * squash))))
    paste(img, layer, x + w / 2, y, scale=k, alpha=clamp(k))
    return w


def L39(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_to = ctx.at("to")
    words = line["words"]
    if t < t_to - 0.06:
        bg.gradient(img, (20, 10, 34), (6, 4, 12))
        bg.graph_paper(img, 70, (30, 18, 50), 4, (44, 26, 70))
        x = 250
        for i in range(4):
            wtxt = words[i]["w"]
            w = token_box(img, wtxt, x, 430, t, words[i]["start"] - 0.03,
                          mask_flip=True if i == 1 else None)
            x += (w or font("mono", 54).getlength(wtxt) + 60) + 30
        draw_text(img, "BERT, circa 2018", W / 2, 600, "mono", 36, (170, 140, 210),
                  alpha=clamp((t - words[1]["start"]) / 0.3))
        # Clawd at a masquerade
        cx, feet, u = W / 2, 990, 16
        draw_clawd(img, cx, feet, u, eyes="normal", blink=blink_at(t), t=t)
        d = ImageDraw.Draw(img, "RGBA")
        y0 = feet - 10 * u
        d.rectangle((cx - 7 * u, y0 + 1.6 * u, cx + 7 * u, y0 + 4.4 * u), fill=(90, 30, 150, 255))
        for ex in (5, 12):
            x_e = cx + (ex - 9) * u
            d.rectangle((x_e, y0 + 2 * u, x_e + u, y0 + 4 * u), fill=(18, 16, 26, 255))
        d.polygon([(cx + 6 * u, y0 + 1.6 * u), (cx + 8 * u, y0 - 2 * u), (cx + 7.4 * u, y0 + 1.6 * u)],
                  fill=(255, 120, 220, 255))
        return
    # recursive self-upgrade: Droste zoom
    k = clamp((t - t_to) / (ctx.scene.end - t_to))
    bg.gradient(img, (30, 10, 50), (8, 4, 20))
    bg.rings(img, t, W / 2, H / 2, color=(200, 120, 255), n=10, speed=200, spacing=110, width=3,
             alpha=90)
    fx0, fy0, fw, fh = 900, 230, 760, 428
    draw_clawd(img, 480, 900, 20, eyes="happy", arms=("up", "up"), t=t)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((fx0 - 18, fy0 - 18, fx0 + fw + 18, fy0 + fh + 18), fill=(*GOLD, 255))
    pop_words(img, sub(line, 4), t, W / 2, 900, "anton", 130, 1700, color=CREAM,
              colors={1: PINK, 2: CYAN}, upper=True, shadow=(6, 6, (20, 0, 30)))
    level = min(9, 1 + int(k * 10))
    draw_text(img, f"v{level}.0", fx0 + fw + 16, fy0 + fh + 64, "mono", 40, GOLD, anchor="rs")
    for _ in range(4):
        snap = img.copy().resize((fw, fh), Image.BILINEAR)
        img.paste(snap, (fx0, fy0))
    ctx.fx["zoom"] = 1 + 0.3 * ease_in(k)
    ctx.fx["center"] = (fx0 + fw * 0.5, fy0 + fh * 0.5)


# --- L40: What did Ilya see? We'll never know ------------------------------------------------------------
def big_eye(img, cx, cy, t, openness=1.0, look=(0.0, 0.0)):
    d = ImageDraw.Draw(img, "RGBA")
    w, h = 380, 190 * openness
    pts_top = [(cx + w * math.cos(a), cy - h * math.sin(a)) for a in np.linspace(0, math.pi, 40)]
    pts_bot = [(cx + w * math.cos(a), cy + h * math.sin(a) * 0.9) for a in np.linspace(math.pi, 0, 40)]
    poly = pts_top + pts_bot
    d.polygon(poly, fill=(240, 240, 235, 255))
    if openness > 0.05:
        ix, iy = cx + look[0] * 150, cy + look[1] * 60 * openness
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon(poly, fill=255)
        iris = Image.new("RGBA", img.size, (0, 0, 0, 0))
        idr = ImageDraw.Draw(iris)
        idr.ellipse((ix - 130, iy - 130, ix + 130, iy + 130), fill=(60, 110, 170, 255))
        idr.ellipse((ix - 100, iy - 100, ix + 100, iy + 100), fill=(90, 150, 210, 255))
        idr.ellipse((ix - 58, iy - 58, ix + 58, iy + 58), fill=(8, 8, 12, 255))
        idr.ellipse((ix - 40, iy - 52, ix - 14, iy - 26), fill=(255, 255, 255, 255))
        img.paste(iris, (0, 0), _and(iris.getchannel("A"), mask))
    d.line(poly + [poly[0]], fill=(30, 30, 40, 255), width=10)


def _and(a, b):
    return Image.fromarray(np.minimum(np.asarray(a), np.asarray(b)))


def L40(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_see = ctx.at("see")
    t_we = ctx.at("we'll")
    t_know = ctx.at("know")
    end_hold = 147.75
    static = smoothstep(t_know + 2.0, end_hold, t)
    img.paste((8, 8, 12), (0, 0, W, H))
    d = ImageDraw.Draw(img, "RGBA")
    # darting pupil
    k = int(t * 1.6)
    fr = (t * 1.6) % 1.0
    la = (hash01(k, 1) - 0.5) * 2, (hash01(k, 2) - 0.5) * 2
    lb = (hash01(k + 1, 1) - 0.5) * 2, (hash01(k + 1, 2) - 0.5) * 2
    e = ease_in_out(clamp(fr * 3))
    look = (lerp(la[0], lb[0], e), lerp(la[1], lb[1], e))
    closing = smoothstep(end_hold - 1.2, end_hold, t)
    blink = 0.1 if (t % 3.7) < 0.12 else 1.0
    big_eye(img, W / 2, 500, t, openness=max(0.02, (1 - closing) * blink), look=look)
    # redaction bars slide over the answer
    for i, tb in enumerate((t_know, t_know + 0.6, t_know + 1.2)):
        if t < tb:
            continue
        kb = ease_out(clamp((t - tb) / 0.3))
        y = 380 + i * 90
        x0 = 380 - i * 30
        x1 = lerp(x0, 1500 + i * 60, kb)
        if x1 > x0:
            d.rectangle((x0, y, x1, y + 70), fill=(0, 0, 0, 255))
        if kb > 0.9:
            draw_text(img, "REDACTED", (380 - i * 30 + x1) / 2, y + 35, "mono", 40, (80, 80, 90))
    if t >= t_know:
        stamp(img, "CLASSIFIED", t, t_know + 0.1, 1450, 780, "shade", 90, (220, 40, 40), angle=-10,
              from_scale=2.0)
    # Clawd with binoculars
    cx, feet, u = 300, 1000, 12
    draw_clawd(img, cx, feet, u, eyes="normal", arms=("up", "up"), t=t)
    y0 = feet - 10 * u
    for ex in (5, 12):
        x_e = cx + (ex - 9) * u
        d.rectangle((x_e - 0.6 * u, y0 + 1.4 * u, x_e + 1.6 * u, y0 + 4.2 * u), fill=(30, 30, 40, 255))
        d.ellipse((x_e - 0.3 * u, y0 + 1.8 * u, x_e + 1.3 * u, y0 + 3.6 * u), fill=(120, 180, 255, 255))
    typewriter(img, "What did Ilya see?", t, line["words"][0]["start"] - 0.05, t_see + 0.15, 330, 170,
               "type", 96, CREAM, cursor=t < t_we)
    if t >= t_we - 0.03:
        txt = "We'll never know"
        n_dots = int(clamp((t - t_know) / 1.5) * 3) if t >= t_know else 0
        pop_words(img, sub(line, 4), t, W / 2, 930, "grotesk", 96, 1600, color=CREAM)
        if n_dots:
            wtxt = font("grotesk", 96).getlength(txt)
            draw_text(img, "." * n_dots, W / 2 + wtxt / 2 + 10, 962, "grotesk", 96, CREAM, anchor="ls")
    if static > 0:
        noise = (np.random.default_rng(ctx.frame).random((H // 4, W // 4)) * 255).astype(np.uint8)
        noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_NEAREST)
        arr = to_np(img)
        arr = cv2.addWeighted(arr, 1 - static * 0.85, cv2.merge([noise] * 3), static * 0.85, 0)
        img.paste(Image.fromarray(arr))


# --- L41: Was it all for show? --------------------------------------------------------------------------
def curtains(img, t, close=0.0):
    d = ImageDraw.Draw(img, "RGBA")
    half = W / 2
    for side in (-1, 1):
        edge = lerp(360, half, close)
        for k in range(14):
            fold = k / 14
            if side < 0:
                x0 = fold * edge
                x1 = (fold + 1 / 14) * edge + 2
            else:
                x0 = W - (fold + 1 / 14) * edge - 2
                x1 = W - fold * edge
            shade = 0.65 + 0.35 * math.sin(k * 1.3 + t * 0.8)
            d.rectangle((x0, 0, x1, H), fill=(int(170 * shade), int(20 * shade), int(40 * shade), 255))
    d.rectangle((0, 0, W, 90), fill=(120, 14, 30, 255))
    for k in range(24):
        x = k * 82
        d.ellipse((x, 50, x + 82, 130), fill=(140, 18, 36, 255))


def L41(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_show = ctx.at("show")
    img.paste((6, 4, 8), (0, 0, W, H))
    d = ImageDraw.Draw(img, "RGBA")
    # stage floor + spotlight
    d.rectangle((0, 820, W, H), fill=(70, 40, 24))
    for k in range(10):
        d.line((0, 830 + k * 26, W, 830 + k * 26), fill=(50, 28, 16, 255), width=3)
    d.polygon([(W / 2 - 60, 0), (W / 2 + 60, 0), (W / 2 + 330, 900), (W / 2 - 330, 900)],
              fill=(255, 245, 210, 40))
    d.ellipse((W / 2 - 330, 850, W / 2 + 330, 950), fill=(255, 245, 210, 60))
    # marquee
    mx0, my0, mx1, my1 = 420, 170, 1500, 420
    d.rectangle((mx0, my0, mx1, my1), fill=(24, 12, 30, 255), outline=(255, 214, 60, 255), width=8)
    n_b = 36
    for k in range(n_b):
        per = 2 * ((mx1 - mx0) + (my1 - my0))
        p = k / n_b * per
        if p < mx1 - mx0:
            bx, by = mx0 + p, my0
        elif p < (mx1 - mx0) + (my1 - my0):
            bx, by = mx1, my0 + p - (mx1 - mx0)
        elif p < 2 * (mx1 - mx0) + (my1 - my0):
            bx, by = mx1 - (p - (mx1 - mx0) - (my1 - my0)), my1
        else:
            bx, by = mx0, my1 - (p - 2 * (mx1 - mx0) - (my1 - my0))
        on = (k + int(t * 8)) % 3 == 0
        d.ellipse((bx - 11, by - 11, bx + 11, by + 11), fill=(255, 240, 150, 255) if on else (120, 90, 40, 255))
    pop_words(img, line, t, (mx0 + mx1) / 2, 300, "bungee", 96, 1000, color=(255, 240, 200),
              active=YELLOW, upper=True)
    # Clawd takes a bow
    bow = 0.0
    if t >= t_show:
        bow = math.sin(clamp((t - t_show) / 0.9) * math.pi)
    draw_clawd(img, W / 2, 890, 17, eyes="happy" if t >= t_show else "normal", arms=("down", "down"),
               angle=-28 * bow, squash=0.1 * bow, blink=blink_at(t), t=t)
    if t >= t_show + 0.3:
        for i in range(18):
            sp = 500 + 300 * hash01(i, 1)
            y = (t - t_show - 0.3) * sp - hash01(i, 2) * 400
            x = W / 2 + (hash01(i, 3) - 0.5) * 900
            if 0 < y < 880:
                sprite(img, "heart", x, y, px=4, angle=(t * 200 + i * 40) % 360)
    close = smoothstep(151.0, 152.2, t)
    curtains(img, t, close)


# --- OUTRO ------------------------------------------------------------------------------------------------
REWIND_TIMES = [149.9, 143.2, 139.9, 136.6, 132.9, 128.6, 124.5, 117.6, 112.9, 105.9, 95.4, 84.2,
                72.6, 70.2, 66.5, 59.9, 52.6, 43.3, 36.95, 33.4, 23.3, 18.4]
_SCHED = {}


def _render_other(tl, t_other, frame):
    from . import SCENE_FNS
    from ..director import build_schedule, run_scene, scene_at
    if "s" not in _SCHED:
        _SCHED["s"] = build_schedule(tl)
    sched = _SCHED["s"]
    sc = sched[scene_at(sched, t_other)]
    arr, _ = run_scene(tl, sc, t_other, int(t_other * FPS), SCENE_FNS)
    return arr


def outro(ctx):
    t, img = ctx.t, ctx.img
    beats = ctx.tl.beats
    t_black = 157.62
    t_final = 157.98
    if t < t_black:
        # VHS rewind through the whole video, one cut per beat
        i = int(np.searchsorted(beats, t, side="right") - np.searchsorted(beats, 152.2))
        i = max(0, min(len(REWIND_TIMES) - 1, i))
        sub_t = (t - beats[np.searchsorted(beats, t, side="right") - 1]) if t >= beats[0] else 0
        arr = _render_other(ctx.tl, REWIND_TIMES[i] - sub_t * 0.8, ctx.frame)
        arr = fx.tint(arr, (200, 220, 255), 0.25)
        # VHS tracking noise: a few bright, horizontally torn bands
        rng = ctx.rng(3)
        arr = arr.copy()
        for _ in range(3):
            y = int(rng.integers(0, H - 40))
            hb = int(rng.integers(8, 40))
            band = np.clip(arr[y:y + hb].astype(np.int16) + 50, 0, 255).astype(np.uint8)
            arr[y:y + hb] = np.roll(band, int(rng.integers(-80, 80)), axis=1)
        img.paste(Image.fromarray(arr))
        if int(t * 3) % 2 == 0:
            draw_text(img, "◀◀ REWIND", 70, 110, "mono", 56, WHITE, anchor="ls")
        tc = REWIND_TIMES[i]
        draw_text(img, f"00:{int(tc) // 60:02d}:{int(tc) % 60:02d}", 70, 180, "mono", 44, WHITE,
                  anchor="ls")
        ctx.fx["glitch"] = 0.25 + 0.3 * ctx.beat(0.1)
        ctx.fx["rgb"] = 8
        ctx.fx["scan"] = 0.9
        return
    if t < t_final:
        # the dropout: blackout, one line of text
        draw_text(img, "P(doom) = 1.00", W / 2, H / 2, "mono", 64, RED)
        ctx.fx["react"] = 0
        return
    # finale: back on the synthwave grid where it started
    bg.synth_grid(img, t, horizon=610, color=(255, 60, 200), speed=1.6,
                  sky=((6, 2, 22), (70, 8, 60)), sun_color=((255, 90, 60), (160, 0, 60)), sun_r=210)
    burst_t = 161.98
    dip = 161.15 <= t < burst_t
    bi, ph = ctx.beat_phase()
    arms = ("up", "out") if bi % 2 == 0 else ("out", "up")
    if dip:
        arms = ("out", "out")
    if t >= burst_t:
        arms = ("up", "down") if int(t * 6) % 2 == 0 else ("up", "up")  # waving
    hop = 0 if dip else ctx.bounce(26)
    draw_clawd(img, W / 2, 930 - hop, 20, arms=arms, eyes="happy" if not dip else "wide",
               blink=blink_at(t, 0.2) if dip else 0, t=t)
    # title reprise
    k = ease_out(clamp((t - t_final) / 0.2))
    draw_text(img, "UPPING MY", W / 2, 150, "anton", 150, CREAM, scale=1.8 - 0.8 * k, alpha=k,
              shadow=(8, 8, PINK))
    draw_text(img, "P(DOOM)", W / 2, 350, "anton", 230, RED, scale=1.8 - 0.8 * k, alpha=k,
              shadow=(9, 9, (60, 0, 20)))
    if t >= 158.9:
        kk = ease_out_back(clamp((t - 158.9) / 0.3), 2.5)
        pill = Image.new("RGBA", (360, 70), (0, 0, 0, 0))
        pd = ImageDraw.Draw(pill)
        pd.rounded_rectangle((0, 0, 359, 69), radius=35, fill=(14, 8, 24, 235),
                             outline=(*CLAWD, 255), width=4)
        li, ox, oy = text_layer("feat. Clawd", "mono", 36, CLAWD)
        pill.paste(li, (int(180 - (li.size[0] - 8) / 2 - 4), int(47 - oy)), li)
        paste(img, pill, W / 2, 530, scale=kk, alpha=clamp(kk))
    if t >= burst_t:
        ctx.fx["glitch"] = 0.12 + 0.25 * ctx.beat(0.1)
        ctx.fx["rgb"] = 8
    # CRT power-off at the very end
    t_off = 163.86
    if t >= t_off:
        kk = clamp((t - t_off) / 0.35)
        arr = to_np(img)
        out = np.zeros_like(arr)
        hh = max(2, int(H * (1 - ease_in(min(1, kk * 1.6)))))
        ww = W if kk < 0.6 else max(4, int(W * (1 - (kk - 0.6) / 0.4)))
        sq = cv2.resize(arr, (ww, hh), interpolation=cv2.INTER_AREA)
        sq = cv2.addWeighted(sq, 1.0, np.full_like(sq, 255), 0.6 * kk, 0)
        out[(H - hh) // 2:(H - hh) // 2 + hh, (W - ww) // 2:(W - ww) // 2 + ww] = sq
        if kk >= 1:
            out[:] = 0
        img.paste(Image.fromarray(out))
        ctx.hud = kk < 0.3
        ctx.fx["react"] = 0


SCENES = {"L35": L35, "L36": L36, "L37": L37, "L38": L38, "L39": L39, "L40": L40, "L41": L41,
          "outro": outro}
