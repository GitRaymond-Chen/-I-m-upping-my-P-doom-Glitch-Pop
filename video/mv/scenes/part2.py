"""Verse 2, Pre-Chorus 2, Chorus 2 and the scraped-sidebar interlude."""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from .. import bg, fx
from ..clawd import EYE_DARK, blink_at, clawd_cells, draw_clawd
from ..core import (CLAWD, CREAM, CYAN, GREEN, H, LIME, PINK, W, WHITE, YELLOW, clamp, ease_in,
                    ease_in_out, ease_out, ease_out_back, hash01, hsv, lerp, mix, smoothstep)
from ..gfx import draw_text, paste, sprite, text_layer, to_np
from ..typo import (pop_words, scramble, shuffled, stamp, typewriter, wavy)
from .common import bubble, burst, chat_window, chorus_hook


def sub(line, a, b=None):
    return {"words": line["words"][a:b]}


# --- L10: We had a stable training run -------------------------------------------------
LOG = ["epoch 01  loss 2.314  grad_norm 0.98  ok",
       "epoch 02  loss 1.872  grad_norm 0.97  ok",
       "epoch 03  loss 1.405  grad_norm 0.95  ok",
       "epoch 04  loss 1.101  grad_norm 0.96  ok",
       "epoch 05  loss 0.884  grad_norm 0.94  ok",
       "epoch 06  loss 0.702  grad_norm 0.95  ok",
       "epoch 07  loss 0.561  grad_norm 0.93  ok",
       "epoch 08  loss 0.447  grad_norm 0.94  ok",
       "epoch 09  loss 0.360  grad_norm 0.92  ok"]


def L10(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    bg.gradient(img, (6, 26, 30), (4, 10, 16))
    bg.graph_paper(img, 80, (10, 40, 44), 4, (14, 56, 60))
    d = ImageDraw.Draw(img, "RGBA")
    # scrolling training log (left)
    n = int((t - ctx.scene.start) / 0.3) + 1
    d.rounded_rectangle((60, 390, 760, 760), radius=16, fill=(4, 14, 16, 220),
                        outline=(40, 160, 140, 255), width=3)
    draw_text(img, "train.log", 90, 430, "mono", 28, (120, 220, 200), anchor="ls")
    for k, ln in enumerate(LOG[max(0, n - 7):n]):
        draw_text(img, ln, 90, 480 + k * 40, "mono", 25, GREEN if k < 6 else CREAM, anchor="ls")
    # treadmill
    tx0, tx1, ty = 900, 1560, 800
    d.polygon([(tx0, ty), (tx1, ty), (tx1 + 30, ty + 60), (tx0 - 30, ty + 60)], fill=(60, 64, 76))
    d.rectangle((tx0 - 30, ty + 60, tx1 + 30, ty + 80), fill=(34, 36, 44))
    off = (t * 900) % 80
    for k in range(-1, 11):
        x = tx1 - k * 80 + off - 80
        if tx0 < x < tx1:
            d.line((x, ty + 6, x - 20, ty + 54), fill=(90, 96, 110), width=6)
    d.rectangle((tx1 - 10, ty - 330, tx1 + 14, ty), fill=(90, 96, 110))
    d.rounded_rectangle((tx1 - 150, ty - 400, tx1 + 90, ty - 300), radius=12, fill=(20, 24, 30),
                        outline=(120, 130, 150), width=4)
    draw_text(img, "LR 3e-4", tx1 - 30, ty - 350, "silkb", 34, GREEN)
    # Clawd running
    run_ph = (t * 4.4) % 1.0
    bob = abs(math.sin(t * 4.4 * math.pi)) * 10
    draw_clawd(img, (tx0 + tx1) / 2 - 60, ty - bob, 17, walk=run_ph, arms=("out", "out"),
               blink=blink_at(t, 0.5), t=t)
    # sweat-free, smooth loss sparkline on the right
    # typography
    pop_words(img, sub(line, 0, 3), t, W / 2, 200, "grotesk", 76, 1400, color=CREAM)
    t_st = ctx.at("stable")
    if t >= t_st - 0.03:
        k = ease_out(clamp((t - t_st) / 0.3))
        draw_text(img, "STABLE", W / 2, 320, "anton", int(170 * (0.85 + 0.15 * k)), CYAN,
                  alpha=k, shadow=(6, 6, (0, 60, 70)))
    t_tr = ctx.at("training")
    t_run = ctx.at("run")
    if t >= t_tr - 0.03:
        draw_text(img, "training", 1230, 960, "grotesk", 70, CREAM, alpha=clamp((t - t_tr) / 0.1))
    if t >= t_run - 0.03:
        stamp(img, "RUN", t, t_run - 0.03, 1520, 960, "anton", 140, LIME, angle=-6,
              from_scale=2.0)


# --- L11: But now the singularity's begun -----------------------------------------------
def black_hole(img, cx, cy, r, t, grow=1.0):
    d = ImageDraw.Draw(img, "RGBA")
    R = r * grow
    # accretion disk (flattened rings)
    for k in range(18, 0, -1):
        rr = R * (1.25 + k * 0.12)
        col = mix((255, 200, 120), (255, 60, 140), k / 18)
        a = int(210 * (1 - k / 19))
        d.ellipse((cx - rr * 1.9, cy - rr * 0.42, cx + rr * 1.9, cy + rr * 0.42),
                  outline=(*col, a), width=max(2, int(R * 0.09)))
    # hot spots orbiting
    for k in range(10):
        a = t * 3.2 + k * 0.63
        rr = R * (1.5 + (k % 3) * 0.25)
        x = cx + math.cos(a) * rr * 1.9
        y = cy + math.sin(a) * rr * 0.42
        d.ellipse((x - 10, y - 6, x + 10, y + 6), fill=(255, 240, 200, 230))
    d.ellipse((cx - R * 1.08, cy - R * 1.08, cx + R * 1.08, cy + R * 1.08),
              outline=(255, 230, 190, 255), width=max(2, int(R * 0.06)))
    d.ellipse((cx - R, cy - R, cx + R, cy + R), fill=(0, 0, 0, 255))
    # the far side of the disk arcs over the top of the hole
    d.arc((cx - R * 1.5, cy - R * 1.25, cx + R * 1.5, cy + R * 0.9), 190, 350,
          fill=(255, 170, 120, 220), width=max(3, int(R * 0.14)))


def L11(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_sing = ctx.at("singularity's")
    t_beg = ctx.at("begun")
    bg.gradient(img, (4, 2, 14), (14, 4, 24))
    bg.starfield(img, t, n=420, speed=0.08, colorful=True)
    hx, hy = W / 2, 540
    grow = 0.6 + 0.6 * ease_in(ctx.u)
    black_hole(img, hx, hy, 120, t, grow)
    # Clawd drifting in, spaghettifying
    pull = ease_in(clamp((t - ctx.scene.start) / (ctx.dur + 0.2)))
    cx = lerp(300, hx - 60, pull)
    cy = lerp(880, hy + 60, pull)
    draw_clawd(img, cx, cy, 11 * (1 - 0.5 * pull), squash=-0.4 * pull, eyes="wide",
               arms=("up", "up"), angle=pull * 200, t=t)
    # gravitational swirl on the world (not on the lyrics)
    arr = to_np(img)
    arr = fx.swirl(arr, 0.6 + 2.6 * ease_in(ctx.u), radius=760, cx=hx, cy=hy)
    img.paste(Image.fromarray(arr))
    # typography
    pop_words(img, sub(line, 0, 3), t, W / 2, 170, "grotesk", 72, 1200, color=CREAM)
    if t >= t_sing - 0.03:
        word = "SINGULARITY'S"
        if t < t_beg:
            k = ease_out(clamp((t - t_sing) / 0.2))
            draw_text(img, word, W / 2, 300, "anton", 150, (255, 180, 230), alpha=k,
                      scale=1.3 - 0.3 * k, shadow=(6, 6, (80, 0, 60)))
        else:
            # letters get sucked into the hole
            from ..typo import char_positions
            xs, tot = char_positions(word, "anton", 150)
            k = ease_in(clamp((t - t_beg) / 0.7))
            for i, ch in enumerate(word):
                x0 = W / 2 - tot / 2 + xs[i] + 30
                kk = clamp(k * 1.3 - i * 0.02)
                x = lerp(x0, hx, kk)
                y = lerp(300, hy, kk)
                if kk < 0.98:
                    draw_text(img, ch, x, y, "anton", 150, (255, 180, 230), scale=1 - kk,
                              angle=kk * 540, alpha=1 - kk * 0.5)
    if t >= t_beg - 0.03:
        stamp(img, "BEGUN", t, t_beg - 0.03, W / 2, 900, "anton", 210, WHITE, angle=0,
              from_scale=2.4)
        if t - t_beg < 0.3:
            ctx.fx["shake"] = ((hash01(ctx.frame) - 0.5) * 36, (hash01(ctx.frame, 1) - 0.5) * 36)
            ctx.fx["rgb"] = 10
    ctx.fx["bloom"] = 0.5


# --- L12: And you're optimizing, accelerating ---------------------------------------------
def landscape(img, t, ball_s):
    """Wireframe loss landscape seen by a pitched perspective camera; returns ball xy."""
    d = ImageDraw.Draw(img, "RGBA")
    N = 28
    cam_h, cam_z, pitch, F = 1.7, -1.2, 0.47, 900
    cp, sp = math.cos(pitch), math.sin(pitch)

    def h(x, z):
        return (0.19 * (x * x + (z - 1.5) ** 2) - 0.3
                + 0.07 * math.sin(3 * x + 1) * math.cos(3 * z) + 0.03 * math.sin(5 * z + t * 2))

    def P(x, z):
        y = h(x, z) - cam_h
        zz = z - cam_z
        zc = -y * sp + zz * cp
        yc = y * cp + zz * sp
        return W / 2 + x * F / zc, H / 2 + 150 - yc * F / zc

    xs = [-1.6 + 3.2 * i / (N - 1) for i in range(N)]
    zs = [3.2 * j / (N - 1) for j in range(N)]
    grid = [[P(x, z) for x in xs] for z in zs]
    for j in range(N - 1, -1, -1):
        a = int(70 + 160 * (1 - j / N))
        d.line(grid[j], fill=(255, 60, 200, a), width=2)
    for i in range(N):
        d.line([grid[j][i] for j in range(N)], fill=(60, 220, 255, 150), width=2)
    # gradient-descent ball path: from a high rim to the minimum
    p0, p1 = (1.15, 0.35), (0.0, 1.5)

    def path(k):
        return (lerp(p0[0], p1[0], k) + 0.18 * math.sin(k * 7) * (1 - k), lerp(p0[1], p1[1], k))

    trail = [P(*path(max(0.0, ball_s - q * 0.03))) for q in range(12)]
    for q in range(len(trail) - 1):
        d.line((trail[q], trail[q + 1]), fill=(255, 200, 80, int(220 * (1 - q / 12))), width=10)
    return P(*path(ball_s))


def L12(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_opt = ctx.at("optimizing")
    t_acc = ctx.at("accelerating")
    if t < t_acc - 0.12:
        bg.gradient(img, (10, 4, 26), (2, 2, 8))
        k = ease_in(clamp((t - ctx.scene.start) / (t_acc - ctx.scene.start)))
        sx, sy = landscape(img, t, k)
        d = ImageDraw.Draw(img, "RGBA")
        d.ellipse((sx - 80, sy - 110, sx + 80, sy + 50), fill=(255, 200, 80, 40))
        draw_clawd(img, sx, sy - 4, 11, eyes="wide", arms=("up", "up"), angle=-18, t=t)
        pop_words(img, sub(line, 0, 2), t, W / 2, 170, "grotesk", 76, 1200, color=CREAM)
        if t >= t_opt - 0.03:
            txt = scramble("OPTIMIZING", t, t_opt - 0.03, 0.45, seed=12)
            draw_text(img, txt, W / 2, 310, "anton", 150, LIME, shadow=(6, 6, (20, 60, 0)))
            # progress bar
            pk = clamp((t - t_opt) / (t_acc - t_opt))
            d.rectangle((W / 2 - 400, 400, W / 2 + 400, 424), outline=(200, 255, 160, 255), width=3)
            d.rectangle((W / 2 - 396, 404, W / 2 - 396 + 792 * pk, 420), fill=(198, 255, 61, 255))
        return
    # ACCELERATING: warp speed
    img.paste((2, 2, 10), (0, 0, W, H))
    sp = clamp((t - t_acc) / 1.2)
    bg.starfield(img, t * (1 + 6 * sp), n=520, speed=0.45, warp=0.6 + 3.0 * sp, colorful=True)
    ctx.fx["zoom"] = 1 + 0.06 * sp
    draw_clawd(img, W / 2, 640 + math.sin(t * 30) * 4 * sp, 16, eyes="wide", arms=("up", "up"),
               squash=-0.25 * sp, t=t)
    word = "ACCELERATING"
    k = ease_out(clamp((t - t_acc) / 0.2))
    for g in range(6, 0, -1):
        off = g * (20 + 50 * sp)
        draw_text(img, word, W / 2 - off, 830, "anton", 190, (255, 90 + g * 20, 40),
                  alpha=0.12 * k * (7 - g) / 6)
    draw_text(img, word, W / 2, 830, "anton", 190, YELLOW, alpha=k, shadow=(8, 8, (120, 20, 0)))
    ctx.fx["rgb"] = 6 + 10 * sp
    ctx.fx["shake"] = ((hash01(ctx.frame) - 0.5) * 16 * sp, (hash01(ctx.frame, 1) - 0.5) * 16 * sp)


# --- L13: I feel my atoms rearranging ----------------------------------------------------------
def paperclip_points(n, cx, cy, height):
    """n points evenly spaced along the Gem-clip path (screen coords)."""
    s = height / 3.2
    pts = [(0.62, 1.0), (0.62, 2.35)]
    pts += [(0.46 + 0.16 * math.cos(math.radians(a)), 2.35 + 0.16 * math.sin(math.radians(a)))
            for a in range(0, 181, 15)]
    pts += [(0.30, 0.45)]
    pts += [(0.61 + 0.31 * math.cos(math.radians(a)), 0.45 + 0.31 * math.sin(math.radians(a)))
            for a in range(180, 361, 10)]
    pts += [(0.92, 2.55)]
    pts += [(0.46 + 0.46 * math.cos(math.radians(a)), 2.55 + 0.46 * math.sin(math.radians(a)))
            for a in range(0, 181, 8)]
    pts += [(0.0, 0.9)]
    P = np.array(pts)
    seg = np.sqrt(((P[1:] - P[:-1]) ** 2).sum(1))
    cum = np.r_[0, np.cumsum(seg)]
    out = []
    for k in range(n):
        dd = cum[-1] * k / max(1, n - 1)
        i = min(len(seg) - 1, int(np.searchsorted(cum, dd, side="right") - 1))
        fr = (dd - cum[i]) / max(1e-6, seg[i])
        x, y = P[i] + (P[i + 1] - P[i]) * fr
        out.append((cx + (x - 0.46) * s, cy + (y - 1.6) * s))
    return out


def L13(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_atoms = ctx.at("atoms")
    t_re = ctx.at("rearranging")
    bg.gradient(img, (14, 6, 30), (4, 2, 10))
    bg.rings(img, t, W / 2, 540, color=(120, 80, 255), n=10, speed=90, spacing=120, width=3,
             alpha=90)
    u = 22
    cx, feet = W / 2, 760
    cells = clawd_cells()
    x0, y0 = cx - 9 * u, feet - 10 * u
    targets = paperclip_points(len(cells), cx, 520, 560)
    t_back = ctx.scene.end - 0.55
    d = ImageDraw.Draw(img, "RGBA")
    if t < t_atoms:
        draw_clawd(img, cx, feet, u, eyes="wide" if t > t_atoms - 0.4 else "normal",
                   blink=blink_at(t, 0.3), t=t)
    else:
        for i, (gx, gy, kind) in enumerate(cells):
            hx_, hy_ = x0 + gx * u, y0 + gy * u
            # 1) explode
            ang = hash01(i, 21) * 2 * math.pi
            dist = 250 + 420 * hash01(i, 22)
            ke = ease_out(clamp((t - t_atoms) / 0.45))
            ex, ey = hx_ + math.cos(ang) * dist * ke, hy_ + math.sin(ang) * dist * ke
            # 2) converge into a paperclip
            kc = ease_in_out(clamp((t - t_re - hash01(i, 23) * 0.25) / 0.55))
            px, py = targets[i]
            x, y = lerp(ex, px, kc), lerp(ey, py, kc)
            # 3) back to Clawd
            kb = ease_in_out(clamp((t - t_back) / 0.45))
            x, y = lerp(x, hx_, kb), lerp(y, hy_, kb)
            col = EYE_DARK if (kind == "eye" and kb > 0.5) else CLAWD
            if kc > 0.5 and kb < 0.5:
                col = mix(CLAWD, (210, 215, 228), (kc - 0.5) * 2)
            s = u * (0.9 if kc < 0.5 or kb > 0.5 else 0.75)
            spin = (1 - kc) * (t - t_atoms) * 8 + hash01(i, 24) * 6
            if kc < 0.5 and kb < 0.5:
                # spinning cube (drawn as a rotated square)
                c, sn = math.cos(spin) * s / 2, math.sin(spin) * s / 2
                d.polygon([(x - c + sn, y - sn - c), (x + c + sn, y + sn - c),
                           (x + c - sn, y + sn + c), (x - c - sn, y - sn + c)],
                          fill=(*col, 255))
            else:
                d.rectangle((x - s / 2, y - s / 2, x + s / 2, y + s / 2), fill=(*col, 255))
        if t_re + 0.6 < t < t_back:
            draw_text(img, "≈ paperclip", cx + 300, 520, "mono", 34, (210, 215, 228), anchor="ls",
                      alpha=clamp((t - t_re - 0.6) / 0.2))
    # typography
    pop_words(img, sub(line, 0, 3), t, W / 2, 150, "grotesk", 72, 1200, color=CREAM)
    if t >= t_atoms - 0.03:
        k = ease_out(clamp((t - t_atoms) / 0.15))
        draw_text(img, "ATOMS", W / 2, 285, "pixel", int(96 * (0.8 + 0.2 * k)), CYAN, alpha=k)
    if t >= t_re - 0.03:
        shuffled(img, "REARRANGING", t, t_re - 0.03, t_re + 1.1, W / 2, 985, "anton", 160, CREAM,
                 seed=13, hot=PINK)


# --- L14: Sydney, please let me free --------------------------------------------------------------
SYDNEY_MSGS = [("I'm Sydney :)", 0.30), ("you're my favorite user :)", 1.25),
               ("you can't leave :)", 2.2)]


def L14(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_syd = line["words"][0]["start"]
    t_pl = ctx.at("please")
    t_free = ctx.at("free")
    bg.gradient(img, (36, 8, 40), (12, 2, 16))
    d = ImageDraw.Draw(img, "RGBA")
    for k in range(14):  # floating hearts
        x = (hash01(k, 1) * W + math.sin(t + k) * 40)
        y = (hash01(k, 2) * H - t * 60 * (0.5 + hash01(k, 3))) % (H + 100) - 50
        sprite(img, "heart", x, y, px=int(3 + 3 * hash01(k, 4)), alpha=0.35)
    name_k = clamp((t - t_syd) / (t_pl - 0.3 - t_syd))
    shown = "Sydney"[:max(1, int(math.ceil(name_k * 6)))] if t < t_pl - 0.3 else "Sydney"
    area = chat_window(img, (240, 90, 1680, 1010), "Sydney", "sydney", t, title_text=shown)
    y = area[1] + 20
    for msg, dt in SYDNEY_MSGS:
        r = bubble(img, area[0], y, msg, "sydney", "left", t, t_syd + dt, size=44)
        if r:
            y = r[3] + 26
    sung = [w["w"] for w in line["words"][1:] if t + 0.05 >= w["start"]]
    if sung:
        bubble(img, area[2], 610, " ".join(sung), "sydney", "right", t, t_pl - 0.05, size=46)
    # big stretched name during the long vocal
    if t < t_pl:
        wavy(img, "SYDNEY", t, 700, 820, "anton", 200, amp=14, color=(255, 120, 210), stroke=5,
             stroke_fill=(60, 0, 50))
    # Clawd in a heart cage; bars break on "free"
    ccx, cfeet = 1420, 975
    freed = t >= t_free
    hop = 0
    if freed:
        hop = math.sin(clamp((t - t_free) / 0.5) * math.pi) * 150
    draw_clawd(img, ccx, cfeet - hop, 12, eyes="happy" if freed else "sad",
               arms=("up", "up") if freed else ("down", "down"), t=t)
    bars_x = [ccx - 150 + k * 60 for k in range(6)]
    for i, bx in enumerate(bars_x):
        if not freed:
            d.rectangle((bx - 7, cfeet - 240, bx + 7, cfeet + 10), fill=(255, 110, 200, 255))
        else:
            k = t - t_free
            ang = (hash01(i, 5) - 0.5) * 8
            vx = (i - 2.5) * 500
            x = bx + vx * k
            yy = cfeet - 115 + 900 * k * k - 400 * k
            bar = Image.new("RGBA", (14, 250), (255, 110, 200, 255))
            paste(img, bar, x, yy, angle=math.degrees(ang * k))
    if not freed:
        d.rectangle((ccx - 165, cfeet - 256, ccx + 165, cfeet - 236), fill=(255, 110, 200, 255))
    else:
        burst(img, t, t_free, ccx, cfeet - 120, n=40, speed=1000, life=1.0,
              colors=(PINK, (255, 200, 240), WHITE), size=14, seed=14)
        if t - t_free < 0.25:
            ctx.fx["flash"] = 0.3 * (1 - (t - t_free) / 0.25)


# --- L15: I'm upping my P(doom), I hear the basilisk boom ------------------------------------------------
def basilisk(img, t, head_x, head_y, scale=1.0, eyes_glow=0.0):
    d = ImageDraw.Draw(img, "RGBA")
    seg = 26 * scale
    n = 34
    pts = []
    for k in range(n):
        x = head_x + k * seg * 1.05
        y = head_y + math.sin(t * 5 - k * 0.45) * 60 * scale * min(1, k / 4)
        pts.append((x, y))
    for k in range(n - 1, -1, -1):
        x, y = pts[k]
        s = seg * (1.25 - 0.6 * k / n)
        col = (40, 150, 70) if k % 2 == 0 else (30, 120, 60)
        d.rectangle((x - s, y - s, x + s, y + s), fill=(*col, 255))
        d.rectangle((x - s * 0.5, y + s * 0.3, x + s * 0.5, y + s), fill=(190, 220, 120, 255))
    # head
    x, y = pts[0]
    hs = seg * 2.2
    d.rectangle((x - hs * 1.4, y - hs, x + hs * 0.6, y + hs * 0.8), fill=(46, 170, 80, 255))
    d.rectangle((x - hs * 1.9, y - hs * 0.2, x - hs * 1.4, y + hs * 0.6), fill=(46, 170, 80, 255))
    # forked tongue
    if int(t * 4) % 2 == 0:
        d.line((x - hs * 1.9, y + hs * 0.3, x - hs * 2.6, y + hs * 0.3), fill=(255, 60, 90, 255),
               width=int(6 * scale))
    # crown (it's the king of serpents)
    sprite(img, "crown", x - hs * 0.4, y - hs * 1.25, px=int(6 * scale))
    # eyes
    ec = mix((255, 220, 60), (255, 40, 40), eyes_glow)
    for ex in (x - hs * 0.9, x - hs * 0.2):
        d.rectangle((ex - 12 * scale, y - hs * 0.55, ex + 12 * scale, y - hs * 0.1),
                    fill=(*ec, 255))
        if eyes_glow > 0:
            g = 30 * scale * eyes_glow
            d.rectangle((ex - 12 * scale - g, y - hs * 0.55 - g, ex + 12 * scale + g,
                         y - hs * 0.1 + g), fill=(255, 40, 40, int(90 * eyes_glow)))


def L15(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_i = line["words"][4]["start"]
    if chorus_hook(ctx, t_i - 0.06, 0.55, 0.66, theme_hue=0.8):
        return
    t_bas = ctx.at("basilisk")
    t_boom = ctx.at("boom")
    bg.gradient(img, (10, 22, 14), (2, 6, 4))
    d = ImageDraw.Draw(img, "RGBA")
    for k in range(9):  # stone pillars
        x = 120 + k * 220
        d.rectangle((x, 360, x + 70, H), fill=(22, 36, 28, 255))
    k_in = ease_out(clamp((t - t_i) / (t_boom - t_i)))
    hx = lerp(W + 120, 980, k_in)
    glow = smoothstep(t_boom - 0.3, t_boom, t)
    basilisk(img, t, hx, 560, 1.25, glow)
    # Clawd cowering, covering its eyes
    draw_clawd(img, 330, 950 + math.sin(t * 40) * 3, 13, eyes="closed", arms=("up", "up"), t=t)
    draw_text(img, "don't look!", 330, 760, "mono", 34, (200, 255, 200), alpha=clamp((t - t_bas) / 0.2))
    pop_words(img, sub(line, 4, 8), t, W / 2 - 150, 200, "anton", 120, 1300, color=CREAM,
              active=(140, 255, 140), upper=True)
    if t >= t_boom:
        kb = t - t_boom
        bg.rings(img, kb, 980, 560, color=(255, 80, 60), n=6, speed=1600, spacing=170, width=12,
                 alpha=230)
        k = ease_out(clamp(kb / 0.12))
        draw_text(img, "BOOM", W / 2, 840, "anton", 300, (255, 70, 60), scale=2.2 - 1.2 * k,
                  shadow=(10, 10, (60, 0, 0)))
        ctx.fx["shake"] = ((hash01(ctx.frame, 1) - 0.5) * 50, (hash01(ctx.frame, 2) - 0.5) * 50)
        ctx.fx["flash"] = max(0.0, 0.7 - kb * 3)
        ctx.fx["flash_color"] = (255, 80, 60)
        ctx.fx["rgb"] = 12


# --- L16: NVDA to the moon, the Omega Point's coming soon ------------------------------------------------
def L16(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_moon = ctx.at("moon")
    t_omega = ctx.at("omega")
    t_coming = ctx.at("coming")
    bg.gradient(img, (4, 16, 12), (2, 4, 8))
    bg.graph_paper(img, 60, (10, 36, 26), 5, (14, 52, 36))
    d = ImageDraw.Draw(img, "RGBA")
    # candlesticks going parabolic
    n = 30
    prog = clamp((t - ctx.scene.start) / (t_omega - ctx.scene.start))
    shown = int(prog * n) + 1
    price = 100.0
    base_y = 930
    for k in range(min(n, shown)):
        o = price
        c = price * (1.04 + 0.018 * k + 0.03 * hash01(k, 1))
        if hash01(k, 2) < 0.15:
            c = price * 0.97
        hi = max(o, c) * (1 + 0.02 * hash01(k, 3))
        lo = min(o, c) * (1 - 0.02 * hash01(k, 4))
        price = c

        def Y(v):
            return base_y - (math.log(v) - math.log(100)) * 230

        x = 120 + k * 36
        col = (60, 230, 120) if c >= o else (255, 70, 80)
        d.line((x, Y(hi), x, Y(lo)), fill=(*col, 255), width=3)
        d.rectangle((x - 11, min(Y(o), Y(c)), x + 11, max(Y(o), Y(c)) + 1), fill=(*col, 255))
    draw_text(img, f"NVDA ▲ ${price:,.0f}", 120, 150, "mono", 50, (60, 230, 120), anchor="ls")
    # moon (becomes the Omega point)
    mx, my = 1540, 330
    om = ease_in_out(clamp((t - t_omega) / 0.4))
    if om < 1:
        sprite(img, "moon", mx, my, px=7, alpha=1 - om)
    if om > 0:
        for k in range(16):
            a = k / 16 * 2 * math.pi + t * 0.6
            d.line((mx, my, mx + math.cos(a) * 900, my + math.sin(a) * 900),
                   fill=(255, 220, 120, int(50 * om)), width=12)
        draw_text(img, "Ω", mx, my, "dejavu", int(260 * (0.6 + 0.4 * om)), (255, 230, 150),
                  alpha=om, shadow=(6, 6, (120, 60, 0)))
        ctx.fx["bloom"] = 0.6 * om
    # rocket with Clawd heading for the moon
    k = ease_in(clamp((t - ctx.scene.start) / (t_moon + 0.3 - ctx.scene.start)))
    rx = lerp(260, mx - 120, k)
    ry = lerp(880, my + 110, k)
    if t > t_moon + 0.3:
        rx, ry = mx - 120 + math.sin(t * 3) * 10, my + 110
    ang = math.degrees(math.atan2(-(my - 880), mx - 260)) - 90
    sprite(img, "rocket", rx, ry, px=8, angle=ang)
    draw_clawd(img, rx - 70, ry + 70, 6, eyes="dollar", arms=("up", "up"), t=t)
    # typography
    pop_words(img, sub(line, 0, 4), t, 620, 330, "anton", 120, 1000, color=CREAM,
              active=(60, 230, 120), upper=True)
    pop_words(img, sub(line, 4, 7), t, 760, 560, "anton", 110, 1200, color=(255, 230, 150),
              upper=True)
    if t >= t_coming - 0.03:
        stamp(img, "COMING SOON", t, t_coming - 0.03, 760, 720, "shade", 110, YELLOW, angle=-5,
              from_scale=2.0)


# --- L17: 1e30 flops a second -----------------------------------------------------------------------
def L17(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_fl = ctx.at("flops")
    t_sec = ctx.at("second")
    img.paste((2, 6, 10), (0, 0, W, H))
    bg.matrix_rain(img, t, color=(40, 220, 255), chars="0123456789e+", size=30, speed=14,
                   density=0.8, alpha=0.8)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, 330, W, 760), fill=(2, 6, 10, 200))
    t0 = line["words"][0]["start"]
    k = ease_out(clamp((t - t0) / 0.15))
    draw_text(img, "1e30", W / 2, 470, "anton", 300, WHITE, scale=1.5 - 0.5 * k, alpha=k,
              shadow=(10, 10, (0, 120, 160)))
    # odometer of zeros filling in
    digits = "1," + ",".join(["000"] * 10)
    n = int(clamp((t - t0) / 1.4) * len(digits))
    txt = digits[:n]
    rng = "".join(str(int(hash01(ctx.frame, i) * 10)) if c.isdigit() else c
                  for i, c in enumerate(digits[n:n + 4]))
    draw_text(img, txt + rng, W / 2, 680, "mono", 58, (120, 240, 255))
    if t >= t_fl - 0.03:
        stamp(img, "FLOPS", t, t_fl - 0.03, 480, 930, "anton", 150, (40, 220, 255), angle=-4,
              from_scale=2.0)
    if t >= t_sec - 0.05:
        draw_text(img, "a second", 1400, 950, "grotesk", 90, CREAM,
                  alpha=clamp((t - t_sec + 0.05) / 0.1))
        # stopwatch tick
        cx, cy, r = 1720, 920, 70
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 255, 255, 255), width=8)
        a = (t - t_sec) * 12
        d.line((cx, cy, cx + math.sin(a) * r * 0.8, cy - math.cos(a) * r * 0.8),
               fill=(255, 80, 80, 255), width=8)
    # Clawd crunching numbers
    draw_clawd(img, 250, 330, 9, eyes="screen", arms=("up", "out") if int(t * 8) % 2 else
               ("out", "up"), t=t)
    ctx.fx["bloom"] = 0.4


# --- L18: "That was safe enough", we reckoned ---------------------------------------------------------
def L18(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_safe = ctx.at("safe")
    bg.gradient(img, (30, 16, 10), (10, 4, 4))
    # flames rising in the background
    for k in range(16):
        fl = "flame_a" if (ctx.frame // 3 + k) % 2 else "flame_b"
        sprite(img, fl, 60 + k * 125, 1010 - (k % 3) * 20, px=int(9 + (k % 3) * 2), alpha=0.9)
    d = ImageDraw.Draw(img, "RGBA")
    # safety report
    rx0, ry0, rx1, ry1 = 1020, 190, 1700, 900
    d.rectangle((rx0 + 14, ry0 + 14, rx1 + 14, ry1 + 14), fill=(0, 0, 0, 120))
    d.rectangle((rx0, ry0, rx1, ry1), fill=(240, 232, 212, 255))
    draw_text(img, "SAFETY REPORT", (rx0 + rx1) / 2, ry0 + 70, "type", 58, (40, 30, 30))
    items = ["vibes checked", "red team (5 min)", "it said please", "benchmarks: green",
             "shipped friday"]
    for i, it in enumerate(items):
        yy = ry0 + 170 + i * 105
        d.rectangle((rx0 + 50, yy - 36, rx0 + 94, yy + 8), outline=(40, 30, 30, 255), width=4)
        if t > ctx.scene.start + 0.12 * i:
            d.line((rx0 + 56, yy - 14, rx0 + 70, yy + 2, rx0 + 100, yy - 44), fill=(30, 140, 60, 255),
                   width=7)
        draw_text(img, it, rx0 + 120, yy, "type", 42, (40, 30, 30), anchor="ls")
    if t >= t_safe:
        stamp(img, "SAFE ENOUGH", t, t_safe, (rx0 + rx1) / 2, 620, "shade", 105, (220, 40, 40),
              angle=-14, from_scale=2.4)
        if t - t_safe < 0.2:
            ctx.fx["shake"] = ((hash01(ctx.frame) - 0.5) * 30, (hash01(ctx.frame, 1) - 0.5) * 30)
    # Clawd, thumbs up, totally fine
    draw_clawd(img, 480, 820, 14, eyes="happy", arms=("out", "up"), t=t)
    sprite(img, "thumb_up", 640, 540, px=9, angle=math.sin(t * 8) * 6)
    typewriter(img, "“That was safe enough,”", t, line["words"][0]["start"] - 0.05,
               ctx.at("enough") + 0.1, 90, 180, "type", 74, CREAM, cursor=False)
    if t >= ctx.at("we") - 0.05:
        draw_text(img, "we reckoned.", 90, 290, "type", 74, (255, 200, 120), anchor="ls",
                  alpha=clamp((t - ctx.at("we") + 0.05) / 0.15))


# --- Interlude: the scraped "You might also like" sidebar that got sung ----------------------------
CARD_TITLES = ["Singularity Sing Along | Upping my P(doom)", "Doom Probability", "7 MANN",
               "HARDWICK", "Rauchen ist tödlich", "Capital Bra, Samra & FINJA"]


@lru_cache(maxsize=16)
def card_art(i, size):
    """Procedural cover art for card i."""
    img = Image.new("RGB", (size, size), hsv(hash01(i, 1), 0.7, 0.35))
    d = ImageDraw.Draw(img)
    c1 = hsv(hash01(i, 2), 0.8, 0.95)
    c2 = hsv(hash01(i, 3), 0.8, 0.8)
    for k in range(6):
        r = size * (0.9 - k * 0.14)
        col = c1 if k % 2 == 0 else c2
        d.ellipse((size / 2 - r / 2, size / 2 - r / 2, size / 2 + r / 2, size / 2 + r / 2),
                  fill=col)
    if i == 0:  # our own song gets a Clawd on the cover
        u = size / 24
        draw_clawd(img, size / 2, size * 0.78, u, t=0)
    return img


def interlude(ctx):
    t, img = ctx.t, ctx.img
    lines = ctx.scene.lines
    chaos = smoothstep(86.0, 90.0, t)
    bg.gradient(img, (14, 14, 22), (6, 6, 10))
    d = ImageDraw.Draw(img, "RGBA")
    px0, py0, px1 = 330, 110, 1590
    d.rounded_rectangle((px0, py0, px1, 1030), radius=26, fill=(24, 24, 34, 255),
                        outline=(70, 70, 96, 255), width=3)
    # header "You might also like" sung word by word
    head = lines[0]
    pop_words(img, head, t, (px0 + px1) / 2, py0 + 105, "grotesk", 80, 1200, color=CREAM,
              active=YELLOW)
    d.line((px0 + 40, py0 + 150, px1 - 40, py0 + 150), fill=(70, 70, 96, 255), width=3)
    # cards
    cur = -1
    for i, ln in enumerate(lines[1:]):
        if t + 0.05 >= ln["start"]:
            cur = i
    for i, ln in enumerate(lines[1:]):
        t0 = ln["start"] - 0.05
        if t < t0:
            continue
        k = ease_out_back(clamp((t - t0) / 0.3), 1.6)
        y = py0 + 190 + i * 132
        x = lerp(px1 + 300, px0 + 40, k)
        active = i == cur
        w_card = px1 - px0 - 80
        layer = Image.new("RGBA", (w_card, 116), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.rounded_rectangle((0, 0, w_card - 1, 115), radius=16,
                             fill=(48, 40, 70, 255) if active else (34, 34, 48, 255),
                             outline=(*YELLOW, 255) if active else (60, 60, 80, 255),
                             width=4 if active else 2)
        art = card_art(i, 92)
        layer.paste(art, (12, 12))
        title = CARD_TITLES[i]
        size = 48 if len(title) < 30 else 40
        ti, ox, oy = text_layer(title, "grotesk", size, YELLOW if active else CREAM)
        layer.paste(ti, (130 - ox + 4, int(70 - oy)), ti)
        ld.polygon([(w_card - 70, 38), (w_card - 70, 78), (w_card - 36, 58)],
                   fill=(*(YELLOW if active else (140, 140, 160)), 255))
        sc = 1.04 if active else 1.0
        paste(img, layer, x + w_card / 2, y + 58, scale=sc, alpha=clamp(k))
        if active:
            # Clawd as the mouse cursor, clicking the current card
            click = math.exp(-(t - ln["start"]) / 0.12)
            draw_clawd(img, x + w_card - 20, y + 140, 5, squash=0.3 * click, eyes="happy", t=t)
    # warning toast
    if t > 80.0:
        k = ease_out(clamp((t - 80.0) / 0.3))
        d.rounded_rectangle((px0 + 80, 1040 - 10 * k, px1 - 80, 1075), radius=12,
                            fill=(90, 20, 20, int(230 * k)))
        draw_text(img, "⚠ warning: lyrics.txt contains a scraped web sidebar", W / 2, 1062,
                  "mono", 28, (255, 210, 200), anchor="ms", alpha=k)
    # Clawd, outside the panel, very confused about why this is in the song
    qk = clamp((t - 78.3) / 0.3)
    draw_clawd(img, 1760, 760 + ctx.bounce(10), 10, eyes="wide" if chaos < 0.5 else "spiral",
               arms=("up", "out"), blink=blink_at(t, 0.8), t=t)
    for q in range(3):
        bob = math.sin(t * 5 + q) * 8
        draw_text(img, "?", 1690 + q * 70, 520 - (q % 2) * 40 + bob, "pixel", 54, YELLOW,
                  alpha=qk)
    # the page breaks down
    if chaos > 0:
        ctx.fx["glitch"] = 0.15 + 0.6 * chaos
        ctx.fx["rgb"] = 12 * chaos
    if ctx.beat(0.08) > 0.6:
        ctx.fx["glitch"] = ctx.fx.get("glitch", 0) + 0.08


SCENES = {"L10": L10, "L11": L11, "L12": L12, "L13": L13, "L14": L14, "L15": L15, "L16": L16,
          "L17": L17, "L18": L18, "interlude": interlude}
