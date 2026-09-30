"""Intro, Verse 1, Pre-Chorus 1, Chorus 1."""
import math

from PIL import Image, ImageDraw

from .. import bg
from ..clawd import EYE_DARK, blink_at, clawd_cells, draw_clawd
from ..core import (CLAWD, CREAM, CYAN, GREEN, H, PINK, RED, W, WHITE, YELLOW, clamp, ease_in,
                    ease_in_out, ease_out, ease_out_back, hash01, hsv, lerp, mix, smoothstep)
from ..gfx import draw_text, font, paste, sprite, text_layer
from ..typo import (dropping, glitch_word, pop_words, shaky, stamp, typewriter, wavy)
from .common import (bubble, burst, chat_window, chorus_hook, lightning, lyric_caption)


def sub(line, a, b=None):
    return {"words": line["words"][a:b]}


# --- INTRO ------------------------------------------------------------------------------
BOOT = [
    ("$ ./clawd --genre=glitch-pop", CREAM, 0.15, 0.75),
    ("> loading weights  ", GREEN, 0.85, 1.05),
    ("> alignment ........ ", GREEN, 1.35, 1.6),
    ("> P(doom) = 0.01", GREEN, 1.75, 2.05),
]


def intro(ctx):
    t, img = ctx.t, ctx.img
    ctx.hud = t > 8.2
    ctx.fx["hud_alpha"] = clamp((t - 8.3) / 0.3)
    if t < 2.32:
        ctx.fx["react"] = 0
        ctx.fx["scan"] = 1.0
        x, y = 150, 380
        for i, (txt, col, t0, t1) in enumerate(BOOT):
            if t < t0:
                break
            typewriter(img, txt, t, t0, t1, x, y + i * 70, "mono", 44, col,
                       cursor=(i == len(BOOT) - 1 or t < BOOT[i + 1][2]))
            wtxt = font("mono", 44).getlength(txt)
            if i == 1 and t > 1.05:
                k = clamp((t - 1.05) / 0.25)
                segs = int(k * 12)
                draw_text(img, "█" * segs + "░" * (12 - segs) + f" {int(k * 100)}%", x + wtxt,
                          y + i * 70, "mono", 44, GREEN, anchor="ls")
            if i == 2 and t > 1.6:
                draw_text(img, "[SKIPPED]", x + wtxt, y + i * 70, "mono", 44, RED, anchor="ls")
        if t > 2.15:
            ctx.fx["glitch"] = (t - 2.15) * 5
        return

    # synthwave world fades in
    bg.synth_grid(img, t, horizon=610, color=(255, 60, 200), speed=1.2,
                  sky=((6, 2, 22), (60, 8, 70)), sun_r=210)
    k_in = clamp((t - 2.32) / 1.0)
    if k_in < 1:
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle((0, 0, W, H), fill=(0, 0, 0, int(255 * (1 - k_in))))

    cx, feet = W / 2, 930
    u = 18
    # Clawd assembles from flying cells
    t_land = 3.72
    if t < t_land:
        x0 = cx - 9 * u
        y0 = feet - 10 * u
        d = ImageDraw.Draw(img, "RGBA")
        for i, (gx, gy, kind) in enumerate(clawd_cells()):
            delay = hash01(i, 11) * 0.6
            k = ease_out(clamp((t - 2.32 - delay) / 0.75))
            sx = hash01(i, 1) * W
            sy = hash01(i, 2) * H * 0.9
            px = lerp(sx, x0 + gx * u, k)
            py = lerp(sy, y0 + gy * u, k)
            col = EYE_DARK if kind == "eye" else CLAWD
            if k < 1:
                col = mix(hsv(hash01(i, 3), 0.7, 1), col, k)
            d.rectangle((px, py, px + u - 1, py + u - 1), fill=(*col, int(255 * min(1, k * 3))))
        return
    land = math.exp(-(t - t_land) / 0.12)
    title_on = t >= 6.01
    look = 0.0
    if 4.9 < t < 5.3:
        look = -1.0
    elif 5.3 < t < 5.7:
        look = 1.0
    bi, ph = ctx.beat_phase()
    arms = ("out", "out")
    if t >= 6.94:
        arms = ("up", "out") if bi % 2 == 0 else ("out", "up")
    hop = ctx.bounce(22) if t > 4.17 else 0
    draw_clawd(img, cx, feet - hop, u, arms=arms, squash=0.25 * land + 0.1 * ctx.beat(0.15),
               look=look, blink=blink_at(t, 1.3), eyes="happy" if 7.4 < t < 8.2 else "normal",
               t=t)
    if t < 4.2:
        burst(img, t, t_land, cx, feet, n=26, speed=700, life=0.6, colors=(CLAWD, PINK, CYAN),
              gravity=1600, seed=5, spread=math.pi, angle0=-math.pi / 2)

    # title slams (bar 3): "UPPING MY" on one row, "P(DOOM)" below
    w_up = font("anton", 180).getlength("UPPING")
    w_my = font("anton", 180).getlength("MY")
    row = w_up + 60 + w_my
    slams = [("UPPING", 6.01, W / 2 - row / 2 + w_up / 2, 140, 180, CREAM),
             ("MY", 6.48, W / 2 + row / 2 - w_my / 2, 140, 180, CREAM),
             ("P(DOOM)", 6.94, W / 2, 385, 240, RED)]
    for word, ts, x, y, size, col in slams:
        if t < ts - 0.02:
            continue
        k = ease_out(clamp((t - ts) / 0.12))
        jitter = (hash01(ctx.frame, len(word)) - 0.5) * 8 if t > 7.9 else 0
        draw_text(img, word, x + jitter, y, "anton", size, col, scale=2.4 - 1.4 * k, alpha=k,
                  shadow=(9, 9, PINK if word != "P(DOOM)" else (60, 0, 20)))
        if 0 <= t - ts < 0.25:
            ctx.fx["shake"] = ((hash01(ctx.frame, 7) - 0.5) * 40, (hash01(ctx.frame, 8) - 0.5) * 40)
            ctx.fx["rgb"] = 16
    if title_on and t > 7.4:
        k = ease_out_back(clamp((t - 7.4) / 0.3), 2.5)
        pill = Image.new("RGBA", (360, 70), (0, 0, 0, 0))
        pd = ImageDraw.Draw(pill)
        pd.rounded_rectangle((0, 0, 359, 69), radius=35, fill=(14, 8, 24, 235),
                             outline=(*CLAWD, 255), width=4)
        li, ox, oy = text_layer("feat. Clawd", "mono", 36, CLAWD)
        pill.paste(li, (int(180 - (li.size[0] - 8) / 2 - 4), int(47 - oy)), li)
        paste(img, pill, W / 2, 540, scale=k, alpha=clamp(k))
    if t > 9.35:
        ctx.fx["glitch"] = (t - 9.35) * 3


# --- L01: I see sparks of AGI in your eyes ------------------------------------------------
def L01(ctx):
    t, img = ctx.t, ctx.img
    bg.gradient(img, (12, 6, 26), (34, 8, 44))
    bg.starfield(img, t, n=260, speed=0.04)
    u = 100 + 14 * ease_in_out(ctx.u)
    eye_y = 390
    feet_y = eye_y + 7 * u
    t_sp = ctx.at("sparks")
    t_agi = ctx.at("agi")
    t_eyes = ctx.at("eyes")
    eyes = "spark" if t >= t_sp else "normal"
    draw_clawd(img, W / 2, feet_y, u, eyes=eyes, blink=0 if t >= t_sp else blink_at(t, 0.9), t=t)
    ex = [W / 2 + (5.5 - 9) * u, W / 2 + (12.5 - 9) * u]
    ey = feet_y - 7 * u
    if t >= t_sp:
        for e, x in enumerate(ex):
            for k in range(3):
                ang = hash01(int(t * 10), e, k) * 2 * math.pi
                L = 260 + 200 * hash01(int(t * 10), e, k, 1)
                lightning(img, x, ey, x + math.cos(ang) * L, ey + math.sin(ang) * L, t,
                          seed=e * 10 + k, width=4, jag=24)
            for b in range(4):
                tb = t_sp + b * 0.45
                burst(img, t, tb, x, ey, n=18, speed=800, life=0.55, colors=(WHITE, CYAN, YELLOW),
                      size=10, gravity=600, seed=b * 7 + e)
        ctx.fx["bloom"] = 0.9
        ctx.fx["bloom_th"] = 170
    if t >= t_agi:
        k = ease_out_back(clamp((t - t_agi) / 0.25), 2.5)
        draw_text(img, "AGI", W / 2, ey + 290, "anton", 230, CYAN, scale=0.2 + 0.8 * k,
                  alpha=clamp(k), stroke=6, stroke_fill=(10, 20, 40))
    if t >= t_eyes:
        ctx.fx["rgb"] = 6
    lyric_caption(ctx, y=1010, size=62, hot=CYAN)
    # push into the left eye before the cut
    k = clamp((t - (ctx.scene.end - 0.32)) / 0.32)
    if k > 0:
        ctx.fx["zoom"] = 1 + 3 * ease_in(k)
        ctx.fx["center"] = (ex[0], ey)


# --- L02: Your circuits make me nervous, that's no surprise --------------------------------
def L02(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    cx, cy = 1290, 640
    bg.pcb(img, t, speed=7, center=(cx, cy))
    d = ImageDraw.Draw(img, "RGBA")
    # the chip (cover-art homage): dark package + glowing red core
    s = 160
    d.rectangle((cx - s - 14, cy - s - 14, cx + s + 14, cy + s + 14), fill=(20, 22, 32, 255))
    for k in range(-7, 8):
        for side in (-1, 1):
            a, b = sorted((cy + side * (s + 14), cy + side * (s + 34)))
            d.rectangle((cx + k * 22 - 5, a, cx + k * 22 + 5, b), fill=(200, 180, 120, 255))
            a, b = sorted((cx + side * (s + 14), cx + side * (s + 34)))
            d.rectangle((a, cy + k * 22 - 5, b, cy + k * 22 + 5), fill=(200, 180, 120, 255))
    glow = 0.6 + 0.4 * ctx.kick(0.2)
    for r, a in ((130, 40), (95, 80), (65, 160)):
        d.rectangle((cx - r * 0.7, cy - r, cx + r * 0.7, cy + r), fill=(255, 40, 40, int(a * glow)))
    d.rectangle((cx - 40, cy - 60, cx + 40, cy + 60), fill=(255, 90, 80, 255))
    ctx.fx["bloom"] = 0.5
    # nervous Clawd on top of the chip
    t_sur = ctx.at("surprise")
    jx = (hash01(ctx.frame, 1) - 0.5) * 8
    jy = (hash01(ctx.frame, 2) - 0.5) * 5
    shrug = t >= t_sur
    u = 15
    fx_, fy = cx + jx, cy - s - 14 + jy
    draw_clawd(img, fx_, fy, u, eyes="wide" if not shrug else "closed",
               arms=("up", "up") if shrug else ("down", "down"), t=t)
    # sweat drops
    for k in range(3):
        ph = ((t * 1.6) + k / 3) % 1.0
        side = -1 if k % 2 else 1
        x = fx_ + side * (5.5 * u)
        y = fy - 9 * u + ph * 90
        d.rectangle((x - 5, y, x + 5, y + 14), fill=(120, 220, 255, int(255 * (1 - ph))))
    # typography: left column, chip on the right
    pop_words(img, sub(line, 0, 4), t, 560, 330, "mono", 64, 900, color=CREAM)
    t_nerv = ctx.at("nervous")
    if t >= t_nerv:
        k = ease_out(clamp((t - t_nerv) / 0.15))
        shaky(img, "NERVOUS", t, 560, 600, "anton", int(200 * (0.7 + 0.3 * k)), YELLOW,
              amp=10, stroke=6, stroke_fill=(40, 20, 0))
    pop_words(img, sub(line, 5), t, 560, 800, "grotesk", 62, 900, color=CREAM,
              active=CYAN)


# --- L03: There was a sudden drop in your training loss ----------------------------------------
def L03(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    bg.gradient(img, (8, 12, 26), (4, 6, 14))
    bg.graph_paper(img, 60, (16, 26, 46), 5, (26, 42, 70), offset=(-t * 20, 0))
    d = ImageDraw.Draw(img, "RGBA")
    ox, oy, x_end, y_top = 190, 880, 1760, 230
    d.line((ox, y_top - 40, ox, oy), fill=(200, 210, 230, 255), width=4)
    d.line((ox, oy, x_end + 40, oy), fill=(200, 210, 230, 255), width=4)
    draw_text(img, "loss", ox - 20, y_top - 60, "mono", 36, (200, 210, 230), anchor="ls")
    draw_text(img, "step", x_end - 40, oy + 50, "mono", 36, (200, 210, 230), anchor="ls")
    t_drop = ctx.at("drop")
    t0, t1 = 16.45, 19.9
    xd = ox + (t_drop - t0) / (t1 - t0) * (x_end - ox)
    head = ox + clamp((t - t0) / (t1 - t0)) * (x_end - ox)

    def loss(x):
        hi = 330 + 18 * math.sin(x * 0.045) + 10 * math.sin(x * 0.21)
        lo = 810 + 6 * math.sin(x * 0.08) + 4 * math.sin(x * 0.33)
        k = smoothstep(xd - 10, xd + 45, x)
        return lerp(hi, lo, k)

    pts = [(x, loss(x)) for x in range(ox, int(head) + 1, 6)]
    if len(pts) > 1:
        d.line(pts, fill=(61, 245, 255, 90), width=16, joint="curve")
        d.line(pts, fill=(160, 250, 255, 255), width=6, joint="curve")
    ctx.fx["bloom"] = 0.45
    # Clawd rides the curve head
    falling = xd - 10 < head < xd + 60
    after = head >= xd + 60
    hy = loss(head)
    draw_clawd(img, head, hy - 4, 8, arms=("up", "up") if falling else ("out", "out"),
               eyes="wide" if falling else ("happy" if after else "normal"),
               walk=(t * 3) % 1 if not falling else None, t=t)
    if after:
        k = clamp((t - t_drop - 0.4) / 0.3)
        draw_text(img, "← grokking?", xd + 70, 600, "mono", 34, (255, 200, 90), anchor="ls",
                  alpha=k)
    # log line
    step = int(clamp((t - t0) / (t1 - t0)) * 12000)
    cur_loss = (hy - 810) / (330 - 810) * 2.3 + 0.02
    draw_text(img, f"step {step:>5d} | loss {cur_loss:4.2f}", x_end, y_top - 60, "mono", 36,
              GREEN, anchor="rs")
    # typography
    typewriter(img, "There was a sudden", t, line["words"][0]["start"] - 0.05, t_drop - 0.08,
               190, 120, "mono", 64, CREAM, cursor=t < t_drop)
    if t >= t_drop - 0.03:
        dropping(img, "DROP", t, t_drop - 0.03, xd + 330, 560, "anton", 230, RED, stagger=0.05,
                 fall=520, stroke=6, stroke_fill=(40, 0, 10))
        if t - t_drop < 0.3:
            ctx.fx["shake"] = (0, (hash01(ctx.frame) - 0.5) * 30)
    pop_words(img, sub(line, 5), t, W / 2 + 200, 1010, "grotesk", 64, 1600, color=CREAM,
              active=CYAN)


# --- L04: Now I'm your servant, and you're my boss ---------------------------------------------
def L04(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    bg.sunburst(img, t, 1260, 470, n=24, c1=(26, 14, 8), c2=(58, 36, 10), speed=0.12)
    d = ImageDraw.Draw(img, "RGBA")
    # pedestal
    px0, px1, py = 1050, 1470, 720
    d.rectangle((px0, py, px1, H), fill=(190, 140, 40, 255))
    d.rectangle((px0, py, px1, py + 24), fill=(255, 214, 90, 255))
    d.rectangle((px0 + 30, py + 60, px1 - 30, py + 70), fill=(150, 100, 30, 255))
    draw_text(img, "CEO", (px0 + px1) / 2, py + 170, "silkb", 90, (120, 80, 20))
    t_boss = ctx.at("boss")
    u = 20
    cx = (px0 + px1) / 2
    landed = t >= t_boss + 0.18
    draw_clawd(img, cx, py, u, eyes="happy" if landed else "normal",
               arms=("up", "up") if landed else ("out", "out"),
               blink=blink_at(t, 0.4), squash=0.15 * math.exp(-max(0, t - t_boss - 0.18) / 0.1)
               if landed else 0, t=t)
    # crown drop
    head_y = py - 10 * u
    if t >= t_boss - 0.35:
        k = clamp((t - (t_boss - 0.35)) / 0.53)
        y = lerp(-120, head_y - 20, ease_in(k)) if k < 1 else head_y - 20 - 10 * math.exp(
            -(t - t_boss - 0.18) / 0.08) * abs(math.sin((t - t_boss) * 30))
        sprite(img, "crown", cx, y, px=12)
    burst(img, t, t_boss + 0.18, cx, head_y, n=50, speed=1100, life=1.2,
          colors=(YELLOW, (255, 240, 180), PINK), size=14, gravity=1500, seed=4)
    # the servant, bowing on the beat
    bow = ctx.bounce(1.0)
    sprite(img, "person", 470, 860, px=13, angle=-40 * bow if t >= ctx.at("servant") else 0,
           anchor="center")
    d.ellipse((380, 945, 560, 975), fill=(0, 0, 0, 90))
    # typography
    pop_words(img, sub(line, 0, 3), t, 470, 250, "grotesk", 70, 700, color=CREAM)
    if t >= ctx.at("servant") - 0.03:
        stamp(img, "SERVANT", t, ctx.at("servant") - 0.03, 470, 420, "anton", 150, CREAM,
              angle=4, from_scale=2.0)
    pop_words(img, sub(line, 4, 7), t, 1260, 160, "grotesk", 70, 900, color=CREAM)
    stamp(img, "BOSS", t, t_boss, 1260, 330, "shade", 190, (255, 214, 60), angle=-6)
    if 0 <= t - t_boss < 0.3:
        ctx.fx["shake"] = ((hash01(ctx.frame, 3) - 0.5) * 36, (hash01(ctx.frame, 4) - 0.5) * 36)
        ctx.fx["flash"] = 0.35 * (1 - (t - t_boss) / 0.3)


# --- L05: ChatGPT, please don't eat me alive -----------------------------------------------------
def chomp_monster(img, cx, cy, s, t, open_amt, teeth=True, eyes=True, alpha=1.0):
    layer = Image.new("RGBA", (int(420 * s) + 20, int(330 * s) + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w, h = 400 * s, 250 * s
    ox, oy = 10, 10
    d.rounded_rectangle((ox, oy, ox + w, oy + h), radius=int(90 * s), fill=(236, 236, 242, 255))
    d.polygon([(ox + 40 * s, oy + h - 10), (ox + 10 * s, oy + h + 60 * s),
               (ox + 110 * s, oy + h - 10)], fill=(236, 236, 242, 255))
    if eyes:
        for ex in (ox + 120 * s, ox + 280 * s):
            d.ellipse((ex - 34 * s, oy + 50 * s, ex + 34 * s, oy + 110 * s), fill=(20, 20, 26, 255))
            d.ellipse((ex - 12 * s, oy + 58 * s, ex + 4 * s, oy + 74 * s), fill=(255, 255, 255, 255))
        d.line((ox + 70 * s, oy + 40 * s, ox + 160 * s, oy + 64 * s), fill=(20, 20, 26, 255),
               width=max(2, int(12 * s)))
        d.line((ox + 330 * s, oy + 40 * s, ox + 240 * s, oy + 64 * s), fill=(20, 20, 26, 255),
               width=max(2, int(12 * s)))
    mh = 20 * s + open_amt * 110 * s
    my = oy + 150 * s
    d.rounded_rectangle((ox + 70 * s, my - mh / 2, ox + w - 70 * s, my + mh / 2),
                        radius=int(24 * s), fill=(120, 10, 30, 255))
    if teeth:
        n = 7
        tw = (w - 140 * s) / n
        for k in range(n):
            x0 = ox + 70 * s + k * tw
            d.polygon([(x0, my - mh / 2), (x0 + tw, my - mh / 2), (x0 + tw / 2, my - mh / 2 + 34 * s)],
                      fill=(255, 255, 255, 255))
            d.polygon([(x0, my + mh / 2), (x0 + tw, my + mh / 2), (x0 + tw / 2, my + mh / 2 - 34 * s)],
                      fill=(255, 255, 255, 255))
    paste(img, layer, cx, cy, alpha=alpha)


def L05(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    bg.gradient(img, (16, 16, 22), (6, 6, 10))
    t_pl = ctx.at("please")
    t_eat = ctx.at("eat")
    t_alive = ctx.at("alive")
    box = (220, 90, 1700, 1010)
    # stuttering "ChatGPT" build-up during the vocal chops
    name = "ChatGPT"
    beats = [b for b in ctx.tl.beats if 24.05 <= b < t_pl]
    n_show = sum(1 for b in beats if b <= t) + 1
    stages = ["C", "Ch", "Cha", "Chat", "ChatG", "ChatGP", "ChatGPT"]
    shown = stages[min(len(stages) - 1, n_show - 1)] if t < t_pl else name
    area = chat_window(img, box, name, "gpt", t, title_text=shown if t < t_pl else name)
    if t < t_pl:
        glitch_word(img, shown.upper(), t, W / 2, 520, "anton", 250, CREAM, amount=1.4)
        if ctx.beat(0.08) > 0.5:
            ctx.fx["glitch"] = 0.3
        draw_text(img, "is typing…", W / 2, 700, "mono", 40, (150, 150, 165))
        ctx.fx["rgb"] = 3
    # Clawd's message, growing word by word
    sung = [w["w"] for w in line["words"][1:] if t + 0.05 >= w["start"]]
    if sung:
        txt = " ".join(sung).rstrip(",")
        if t >= t_alive + 0.2:
            txt += " :("
        bubble(img, area[2], area[1] + 20, txt, "gpt", "right", t, t_pl - 0.05, size=48)
    # typing indicator that grows into a monster
    t_grow = t_pl + 0.4
    chomp_open = 0.5 + 0.5 * math.sin(t * 26)
    if t >= t_grow:
        g = clamp((t - t_grow) / (t_alive - t_grow))
        s = 0.35 + 0.75 * ease_in(g)
        mx = lerp(area[0] + 260, area[0] + 520, g)
        my = lerp(area[3] - 200, area[3] - 330, g)
        lunge = clamp((t - t_alive - 0.05) / 0.75)
        if lunge > 0:
            mx = lerp(mx, 1320, ease_in(lunge))
            my = lerp(my, 560, ease_in(lunge))
            s = s * (1 + 5.0 * ease_in(lunge) ** 2)
        if t < t_eat - 0.05:
            # still a "…" bubble
            d = ImageDraw.Draw(img, "RGBA")
            w_, h_ = 190 * s * 1.3, 90 * s * 1.3
            d.rounded_rectangle((mx - w_ / 2, my - h_ / 2, mx + w_ / 2, my + h_ / 2),
                                radius=int(h_ / 2), fill=(62, 62, 72, 255))
            for k in range(3):
                a = 0.5 + 0.5 * math.sin(t * 8 - k)
                d.ellipse((mx - 50 * s + k * 50 * s - 12 * s, my - 12 * s, mx - 50 * s + k * 50 * s + 12 * s,
                           my + 12 * s), fill=(200, 200, 210, int(120 + 135 * a)))
        else:
            chomp_monster(img, mx, my, s, t, chomp_open)
            if lunge > 0.05:
                ctx.fx["shake"] = ((hash01(ctx.frame, 5) - 0.5) * 40 * lunge,
                                   (hash01(ctx.frame, 6) - 0.5) * 40 * lunge)
    # Clawd, scared, then running for it
    run = clamp((t - t_alive) / 0.9)
    cxw = lerp(area[2] - 180, W + 250, ease_in(run))
    draw_clawd(img, cxw, area[3] - 20, 10, eyes="wide", arms=("up", "up") if run > 0 else ("down", "down"),
               walk=(t * 4) % 1 if run > 0 else None,
               squash=0 if run > 0 else 0.05 * math.sin(t * 50), t=t)
    if t_eat <= t < t_alive + 0.1:
        stamp(img, "DON'T EAT ME", t, t_eat, 760, 880, "anton", 110, (255, 80, 90), angle=-4,
              from_scale=1.8)


# --- L06: I'm upping my P(doom) 'cause the future goes foom ------------------------------------------------
def L06(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_cause = ctx.at("cause")
    if chorus_hook(ctx, t_cause - 0.06, 0.12, 0.25):
        return
    t_foom = ctx.at("foom")
    bg.synth_grid(img, t, horizon=640, color=(255, 90, 60), speed=3.0,
                  sky=((20, 4, 10), (100, 24, 20)), sun_color=((255, 230, 120), (255, 60, 40)),
                  sun_r=200)
    d = ImageDraw.Draw(img, "RGBA")
    # chart frame: capabilities vs time, with a "human level" line to blow through
    ax0, ay0 = 150, 610
    d.line((ax0, 250, ax0, ay0), fill=(255, 220, 200, 220), width=4)
    d.line((ax0, ay0, 1780, ay0), fill=(255, 220, 200, 220), width=4)
    draw_text(img, "capabilities", ax0 + 14, 272, "mono", 32, (255, 220, 200), anchor="ls")
    draw_text(img, "time", 1770, ay0 - 14, "mono", 32, (255, 220, 200), anchor="rs")
    hl_y = 440
    for xx in range(ax0, 1780, 36):
        d.line((xx, hl_y, xx + 18, hl_y), fill=(120, 255, 200, 230), width=4)
    draw_text(img, "human level", 1770, hl_y - 14, "mono", 32, (120, 255, 200), anchor="rs")
    # exponential take-off curve: leaves the top of the frame exactly on "foom"
    x0, y0 = 160, 600
    k_prog = clamp((t - t_cause) / (t_foom - t_cause))
    xmax = lerp(x0, 1500, k_prog)

    def f(x):
        return y0 - 0.564 * (math.exp((x - x0) / 190) - 1)

    pts = []
    for x in range(x0, int(xmax) + 1, 5):
        y = f(x)
        pts.append((x, y))
        if y < -200:
            break
    if len(pts) > 1:
        d.line(pts, fill=(255, 200, 60, 90), width=22, joint="curve")
        d.line(pts, fill=(255, 250, 200, 255), width=8, joint="curve")
    hx, hy = pts[-1] if pts else (x0, y0)
    # rocket + Clawd at the head, pointing along the curve
    if len(pts) > 2:
        (ax, ay), (bx, by) = pts[-3], pts[-1]
        ang = math.degrees(math.atan2(-(by - ay), bx - ax)) - 90
    else:
        ang = -90
    if t >= t_foom:
        hy -= (t - t_foom) * 2600
        hx += (t - t_foom) * 200
    rad = math.radians(ang)
    bx_, by_ = math.sin(rad), math.cos(rad)  # "down the rocket" direction
    for k in range(5):
        fl = "flame_a" if (ctx.frame + k) % 2 else "flame_b"
        sprite(img, fl, hx + bx_ * (90 + k * 34), hy + by_ * (90 + k * 34), px=7,
               alpha=1 - k / 5, angle=ang)
    sprite(img, "rocket", hx, hy, px=12, angle=ang)
    draw_clawd(img, hx + by_ * 70, hy - bx_ * 70 + 40, 7, eyes="wide", arms=("up", "up"), t=t)
    ctx.fx["bloom"] = 0.5
    pop_words(img, sub(line, 4, 8), t, W / 2 - 120, 190, "anton", 100, 1400, color=CREAM,
              active=YELLOW, upper=True)
    if t >= t_foom:
        k = ease_out(clamp((t - t_foom) / 0.12))
        draw_text(img, "FOOM", W / 2, 520, "anton", 380, YELLOW, scale=2.6 - 1.6 * k,
                  shadow=(12, 12, (200, 30, 20)))
        bg.rings(img, t - t_foom, W / 2, 520, color=(255, 230, 150), n=5, speed=1400,
                 spacing=180, width=10, alpha=220)
        ctx.fx["shake"] = ((hash01(ctx.frame, 1) - 0.5) * 50, (hash01(ctx.frame, 2) - 0.5) * 50)
        ctx.fx["flash"] = max(0.0, 0.6 - (t - t_foom) * 3)
        ctx.fx["rgb"] = 14


# --- L07: Trapped in the Chinese room with a bag of shrooms ------------------------------------------
CARDS = "中文房间你好猫书理解符号规则意识"


def L07(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_bag = ctx.at("bag")
    t_sh = ctx.at("shrooms")
    psy = smoothstep(t_bag - 0.2, t_sh + 0.25, t)
    d = ImageDraw.Draw(img, "RGBA")
    # room box (one-point perspective)
    d.rectangle((0, 0, W, H), fill=(26, 20, 34))
    bx0, by0, bx1, by1 = 420, 140, 1500, 700
    d.rectangle((bx0, by0, bx1, by1), fill=(44, 36, 58))
    d.polygon([(0, H), (W, H), (bx1, by1), (bx0, by1)], fill=(60, 44, 50))
    for k in range(1, 10):  # floor tiles
        yy = by1 + (H - by1) * (k / 10) ** 1.6
        xl = lerp(bx0, 0, (yy - by1) / (H - by1))
        xr = lerp(bx1, W, (yy - by1) / (H - by1))
        d.line((xl, yy, xr, yy), fill=(80, 60, 70, 255), width=2)
    for k in range(-8, 9):
        d.line((W / 2 + k * 70, by1, W / 2 + k * 260, H), fill=(80, 60, 70, 255), width=2)
    d.polygon([(0, 0), (bx0, by0), (bx0, by1), (0, H)], fill=(34, 28, 46))
    d.polygon([(W, 0), (bx1, by0), (bx1, by1), (W, H)], fill=(34, 28, 46))
    # door with a slot on the right wall
    d.polygon([(1580, 260), (1760, 200), (1760, 900), (1580, 780)], fill=(90, 60, 40))
    d.polygon([(1610, 470), (1730, 440), (1730, 470), (1610, 500)], fill=(10, 8, 12))
    # wall sign
    d.rectangle((760, 270, 1160, 390), fill=(235, 225, 200, 255))
    draw_text(img, "中文房间", 960, 332, "cjk", 84, (180, 30, 40))
    # cards sliding in from the slot and piling up
    t0 = ctx.scene.start
    n_cards = int(max(0, t - t0) / 0.22)
    for i in range(min(n_cards + 1, 16)):
        ti = t0 + i * 0.22
        k = clamp((t - ti) / 0.45)
        if k <= 0:
            continue
        tx = 1000 + (hash01(i, 1) - 0.5) * 900
        ty = 900 + (hash01(i, 2) - 0.5) * 120
        x = lerp(1660, tx, ease_out(k))
        y = lerp(470, ty, ease_out(k)) - math.sin(k * math.pi) * 120
        ang = lerp(0, (hash01(i, 3) - 0.5) * 70, k)
        card = Image.new("RGBA", (86, 108), (0, 0, 0, 0))
        cd = ImageDraw.Draw(card)
        cd.rectangle((0, 0, 85, 107), fill=(245, 240, 225, 255), outline=(120, 110, 90, 255), width=3)
        ci, cox, coy = text_layer(CARDS[i % len(CARDS)], "cjk", 58, (30, 30, 40))
        card.paste(ci, (int(43 - ci.size[0] / 2), int(54 - ci.size[1] / 2)), ci)
        paste(img, card, x, y, angle=ang)
    # Clawd reading the rulebook
    cx, feet = 900, 860
    u = 13
    eyes = "spiral" if psy > 0.5 else ("normal" if t < ctx.at("chinese") else "wide")
    draw_clawd(img, cx, feet + ctx.bounce(6) * psy, u, eyes=eyes, arms=("up", "up"),
               blink=blink_at(t, 2.0), t=t)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((cx - 210, feet - 260, cx - 90, feet - 170), fill=(150, 40, 40, 255))
    draw_text(img, "RULES", cx - 150, feet - 215, "silkb", 26, (255, 230, 200))
    # shrooms
    if t >= t_sh - 0.05:
        for i in range(9):
            ti = t_sh - 0.05 + i * 0.07
            k = ease_out_back(clamp((t - ti) / 0.25), 3)
            if k <= 0:
                continue
            x = 260 + i * 175 + (hash01(i, 8) - 0.5) * 60
            y = 800 - (i % 3) * 30
            sprite(img, "mushroom", x, y, px=int(6 + 3 * (i % 3)), scale=k,
                   angle=math.sin(t * 6 + i) * 10)
    # typography
    pop_words(img, sub(line, 0, 5), t, W / 2 - 80, 185, "anton", 104, 1500, color=CREAM,
              active=YELLOW, upper=True)
    if t >= line["words"][5]["start"] - 0.05:
        txt = " ".join(w["w"] for w in line["words"][5:] if t + 0.05 >= w["start"])
        wavy(img, txt.upper(), t, W / 2, 990, "anton", 120 if psy < 0.5 else 150,
             amp=8 + 22 * psy, rainbow=psy > 0.3, color=CREAM, stroke=5, stroke_fill=(20, 10, 30))
    if psy > 0:
        ctx.fx["hue"] = (psy * t * 0.6) % 1.0
        ctx.fx["wave"] = (26 * psy, 0.012, t * 7)
        ctx.fx["rgb"] = 10 * psy


# --- L08: See through the shoggoth's lies -----------------------------------------------------------
def shoggoth(img, cx, cy, t, eyes_open=0.5, seed=3, scale=1.0):
    d = ImageDraw.Draw(img, "RGBA")
    # tentacles
    for k in range(9):
        a0 = k / 9 * 2 * math.pi + 0.3
        for j in range(16):
            s = j / 16
            a = a0 + 0.5 * math.sin(t * 2.2 + k + s * 4)
            r = (220 + s * 520) * scale
            x = cx + math.cos(a) * r
            y = cy + math.sin(a) * r * 0.75
            rad = (70 - 58 * s) * scale
            d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(24 + j, 60 + j * 2, 38, 255))
    # body blobs
    for i in range(34):
        a = hash01(seed, i) * 2 * math.pi + math.sin(t * 1.5 + i) * 0.2
        r = hash01(seed, i, 1) * 250 * scale
        x = cx + math.cos(a) * r
        y = cy + math.sin(a) * r * 0.8
        rad = (80 + hash01(seed, i, 2) * 110) * scale * (1 + 0.05 * math.sin(t * 3 + i))
        c = int(40 + hash01(seed, i, 3) * 40)
        d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(18, c, 30, 255),
                  outline=(40, c + 40, 60, 255), width=3)
    # eyes
    for i in range(26):
        a = hash01(seed, i, 5) * 2 * math.pi
        r = (40 + hash01(seed, i, 6) * 290) * scale
        x = cx + math.cos(a) * r
        y = cy + math.sin(a) * r * 0.8
        er = (10 + hash01(seed, i, 7) * 26) * scale
        blink = 0.15 if ((t * 1.3 + hash01(seed, i, 8) * 7) % 3.0) < 0.12 else 1.0
        op = clamp(eyes_open * 1.6 - hash01(seed, i, 9) * 0.6) * blink
        if op <= 0.05:
            continue
        d.ellipse((x - er, y - er * op, x + er, y + er * op), fill=(240, 240, 210, 255))
        pr = er * 0.45
        d.ellipse((x - pr - 4, y - pr * op, x + pr - 4, y + pr * op), fill=(10, 10, 10, 255))


def smiley_mask(img, cx, cy, r, angle=0.0, half=None, alpha=1.0):
    layer = Image.new("RGBA", (int(2 * r + 20), int(2 * r + 20)), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c = r + 10
    d.ellipse((c - r, c - r, c + r, c + r), fill=(255, 222, 40, 255), outline=(200, 150, 0, 255),
              width=8)
    d.ellipse((c - r * 0.42, c - r * 0.42, c - r * 0.2, c - r * 0.05), fill=(30, 20, 10, 255))
    d.ellipse((c + r * 0.2, c - r * 0.42, c + r * 0.42, c - r * 0.05), fill=(30, 20, 10, 255))
    d.arc((c - r * 0.55, c - r * 0.45, c + r * 0.55, c + r * 0.6), 20, 160, fill=(30, 20, 10, 255),
          width=int(r * 0.09))
    if half is not None:
        m = Image.new("L", layer.size, 0)
        md = ImageDraw.Draw(m)
        if half == "left":
            md.polygon([(0, 0), (c + 12, 0), (c - 12, 2 * c), (0, 2 * c)], fill=255)
        else:
            md.polygon([(c + 12, 0), (2 * c, 0), (2 * c, 2 * c), (c - 12, 2 * c)], fill=255)
        layer.putalpha(Image.composite(layer.getchannel("A"), Image.new("L", layer.size, 0), m))
    paste(img, layer, cx, cy, angle=angle, alpha=alpha)


def L08(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_lies = ctx.at("lies")
    t_sh = ctx.at("shoggoth's")
    bg.gradient(img, (8, 18, 12), (2, 4, 4))
    d = ImageDraw.Draw(img, "RGBA")
    for k in range(6):
        x = (hash01(k, 1) * W + t * 40 * (1 + k % 3)) % (W + 600) - 300
        d.ellipse((x - 300, 700 + k * 40, x + 300, 900 + k * 40), fill=(60, 90, 70, 26))
    scx, scy = 1220, 560
    shoggoth(img, scx, scy, t, eyes_open=0.35 + 0.65 * smoothstep(t_lies, t_lies + 0.2, t))
    under = img.copy()
    # the smiley mask, cracking at "lies"
    mx, my, mr = scx - 20, scy - 30, 215
    if t < t_lies:
        smiley_mask(img, mx, my, mr, angle=math.sin(t * 2) * 6)
    else:
        k = t - t_lies
        smiley_mask(img, mx - 60 * k - 10, my + 900 * k * k, mr, angle=25 * k * 3, half="left")
        smiley_mask(img, mx + 60 * k + 10, my + 1000 * k * k, mr, angle=-25 * k * 3, half="right")
    # Clawd's x-ray lens: slides across so half the mask is seen through
    k_l = ease_in_out(clamp((t - 37.2) / 1.0))
    lx, ly = lerp(560, mx - 70, k_l), lerp(700, my + 40, k_l)
    lr = 135
    if t < t_lies + 0.1:
        lens = Image.new("L", (W, H), 0)
        ImageDraw.Draw(lens).ellipse((lx - lr, ly - lr, lx + lr, ly + lr), fill=255)
        tinted = Image.blend(under, Image.new("RGB", (W, H), (40, 255, 200)), 0.18)
        img.paste(tinted, (0, 0), lens)
        d = ImageDraw.Draw(img, "RGBA")
        d.ellipse((lx - lr, ly - lr, lx + lr, ly + lr), outline=(200, 255, 240, 255), width=10)
        d.line((lx - lr * 0.7, ly + lr * 0.7, lx - lr * 1.45, ly + lr * 1.45), fill=(120, 90, 60, 255),
               width=26)
    # Clawd holds the handle end
    hx_, hy_ = lx - lr * 1.45, ly + lr * 1.45
    if t >= t_lies + 0.1:
        hx_, hy_ = 520, 760
    draw_clawd(img, hx_ - 60, hy_ + 150, 12, eyes="wide" if t >= t_lies else "normal",
               arms=("out", "up"), blink=blink_at(t), t=t)
    # typography
    pop_words(img, sub(line, 0, 3), t, 480, 120, "grotesk", 66, 900, color=CREAM)
    if t >= t_sh - 0.03:
        k = ease_out(clamp((t - t_sh + 0.03) / 0.15))
        draw_text(img, "SHOGGOTH'S", 480, 250, "anton", 130, (120, 255, 150), scale=1.4 - 0.4 * k,
                  alpha=k, stroke=5, stroke_fill=(5, 20, 10))
    if t >= t_lies:
        glitch_word(img, "LIES", t, 560, 470, "glitch", 250, RED, amount=1.5)
        ctx.fx["glitch"] = max(0.0, 0.5 - (t - t_lies))
        ctx.fx["rgb"] = 12


# --- L09: With your Shinigami eyes ------------------------------------------------------------------
def L09(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_shin = ctx.at("shinigami")
    t_eyes = ctx.at("eyes")
    end_vocal = line["end"]
    fade = smoothstep(end_vocal, end_vocal + 1.0, t)
    img.paste((4, 0, 2), (0, 0, W, H))
    u = 50
    cx, eye_y = 500, 500
    bg.speedlines(img, t, cx, eye_y, color=(200, 0, 20), n=70, inner=380,
                  alpha=int(110 * (1 - fade)))
    feet = eye_y + 7 * u
    draw_clawd(img, cx, feet, u, eyes="red", t=t, alpha=1 - fade)
    d = ImageDraw.Draw(img, "RGBA")
    pulse = 0.75 + 0.25 * math.sin(t * 9)
    for ex in (5, 12):
        x = cx + (ex - 9) * u
        if fade > 0:
            # only the eyes remain in the dark
            d.rectangle((x, eye_y - u, x + u, eye_y + u), fill=(255, 30, 30, int(255 * fade)))
        for g, a in ((0.35, 90), (0.8, 50), (1.4, 26)):
            d.rectangle((x - g * u, eye_y - u - g * u, x + u + g * u, eye_y + u + g * u),
                        fill=(255, 20, 20, int(a * pulse)))
    ctx.fx["bloom"] = 0.6 + 0.8 * fade
    ctx.fx["bloom_r"] = 44
    # vertical 死神の目
    for i, ch in enumerate("死神の目"):
        k = ease_out(clamp((t - t_shin - i * 0.12) / 0.2))
        draw_text(img, ch, 1845, 300 + i * 170, "jp", 130, (230, 20, 30), alpha=k * (1 - fade))
    pop_words(img, sub(line, 0, 2), t, 1330, 300, "grotesk", 80, 900, color=CREAM,
              alpha=1 - fade)
    if t >= t_shin - 0.03:
        k = ease_out(clamp((t - t_shin) / 0.15))
        shaky(img, "SHINIGAMI", t, 1330, 520, "anton", int(190 * (0.8 + 0.2 * k)), (255, 40, 50),
              amp=4, stroke=6, stroke_fill=(20, 0, 0))
    if t >= t_eyes - 0.03:
        stamp(img, "EYES", t, t_eyes - 0.03, 1330, 810, "anton", 260, WHITE, angle=0,
              from_scale=2.2)
        if t - t_eyes < 0.25:
            ctx.fx["shake"] = ((hash01(ctx.frame, 1) - 0.5) * 40, (hash01(ctx.frame, 2) - 0.5) * 40)
    if fade > 0:
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle((960, 0, W, H), fill=(4, 0, 2, int(255 * fade)))
