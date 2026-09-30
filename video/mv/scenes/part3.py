"""Bridge 1, Pre-Chorus 3, Chorus 3."""
import math
from functools import lru_cache

from PIL import Image, ImageDraw

from .. import bg
from ..clawd import blink_at, draw_clawd
from ..core import (CLAWD, CREAM, CYAN, GREEN, H, LIME, PINK, RED, W, YELLOW, clamp, ease_in,
                    ease_in_out, ease_out, ease_out_back, hash01, hsv, lerp, mix, smoothstep)
from ..gfx import draw_text, paste, sprite, text_layer
from ..sprites import paperclip_img
from ..typo import pop_words, shaky, stamp, typewriter, wavy
from .common import bubble, chat_window, chorus_hook

ORANGE = (255, 150, 60)


def sub(line, a, b=None):
    return {"words": line["words"][a:b]}


# --- L26: Forward MLP, backward, repeat -------------------------------------------------------
LAYERS = [4, 6, 6, 3]


def net_nodes():
    xs = [430, 790, 1150, 1510]
    nodes = []
    for li, n in enumerate(LAYERS):
        ys = [620 + (k - (n - 1) / 2) * 105 for k in range(n)]
        nodes.append([(xs[li], y) for y in ys])
    return nodes


def L26(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_fw = line["words"][0]["start"]
    t_bw = ctx.at("backward")
    t_rep = ctx.at("repeat")
    bg.gradient(img, (6, 8, 24), (2, 2, 10))
    bg.graph_paper(img, 60, (12, 16, 40), 5, (18, 24, 56))
    d = ImageDraw.Draw(img, "RGBA")
    nodes = net_nodes()
    # which pass is running: position of the wave in layer units
    if t < t_bw:
        wave, col = (t - t_fw) / (t_bw - t_fw) * 3.6, ORANGE
    elif t < t_rep:
        wave, col = 3 - (t - t_bw) / (t_rep - t_bw) * 3.6, CYAN
    else:
        ph = ((t - t_rep) * 4.0) % 2.0
        if ph < 1:
            wave, col = ph * 3.6, ORANGE
        else:
            wave, col = 3 - (ph - 1) * 3.6, CYAN
    for li in range(3):
        for a in nodes[li]:
            for b in nodes[li + 1]:
                seg_pos = wave - li  # 0..1 while the wave crosses this gap
                lit = 0 <= seg_pos <= 1
                d.line((a, b), fill=(*col, 150) if lit else (60, 70, 110, 110), width=3 if lit else 2)
                if lit:
                    pk = seg_pos
                    px, py = lerp(a[0], b[0], pk), lerp(a[1], b[1], pk)
                    d.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(*col, 255))
    for li, layer in enumerate(nodes):
        glow = clamp(1 - abs(wave - li) * 1.5)
        for (x, y) in layer:
            c = mix((40, 46, 80), col, glow)
            d.ellipse((x - 30, y - 30, x + 30, y + 30), fill=(*c, 255), outline=(170, 180, 230, 255),
                      width=4)
    labels = ["input", "hidden", "hidden", "output"]
    for li, lab in enumerate(labels):
        draw_text(img, lab, nodes[li][0][0], 960, "mono", 30, (150, 160, 210))
    # Clawd rides the output layer
    draw_clawd(img, 1690, 700 + ctx.bounce(14), 8, eyes="happy" if t > t_rep else "normal",
               arms=("up", "up") if t > t_rep else ("out", "out"), t=t)
    if t > t_rep:
        k = (t - t_rep) * 6
        d.arc((1620 - 70, 360 - 70, 1620 + 70, 360 + 70), int(k * 57) % 360,
              (int(k * 57) + 290) % 360, fill=(*LIME, 255), width=12)
    colors = {0: ORANGE, 1: ORANGE, 2: CYAN, 3: LIME}
    pop_words(img, line, t, W / 2 - 60, 230, "anton", 118, 1600, color=CREAM, colors=colors,
              upper=True)


# --- L27: Now von Neumann's obsolete --------------------------------------------------------------
BOXES = [("CONTROL UNIT", 470, 420, 330, 110), ("ALU", 470, 560, 330, 110),
         ("MEMORY", 1060, 490, 330, 250), ("INPUT", 470, 800, 250, 90),
         ("OUTPUT", 1140, 800, 250, 90)]


def L27(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_obs = ctx.at("obsolete")
    img.paste((14, 20, 12), (0, 0, W, H))
    d = ImageDraw.Draw(img, "RGBA")
    ph = (57, 255, 136)
    fall = max(0.0, t - t_obs - 0.12)
    # CPU outline
    if fall <= 0:
        d.rectangle((420, 360, 850, 700), outline=(*ph, 180), width=3)
        draw_text(img, "CPU", 635, 350, "mono", 30, ph)
        for (x0, y0, x1, y1) in ((800, 560, 1060, 560), (700, 850, 1140, 850), (1225, 740, 1225, 800),
                                 (595, 700, 595, 800)):
            d.line((x0, y0, x1, y1), fill=(*ph, 220), width=5)
    for i, (lab, x, y, w, h) in enumerate(BOXES):
        dy = 0.5 * 2600 * max(0.0, fall - hash01(i, 3) * 0.15) ** 2
        ang = (hash01(i, 4) - 0.5) * 140 * max(0.0, fall - 0.05)
        box = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(box)
        bd.rectangle((0, 0, w - 1, h - 1), fill=(10, 30, 16, 255), outline=(*ph, 255), width=5)
        li, ox, oy = text_layer(lab, "mono", 32, ph)
        box.paste(li, (int(w / 2 - li.size[0] / 2), int(h / 2 - li.size[1] / 2)), li)
        paste(img, box, x + w / 2, y + h / 2 + dy, angle=ang)
    # Clawd, the new architecture, kicking the old one over
    kick = t >= t_obs
    draw_clawd(img, 1620, 900, 14, eyes="happy" if kick else "normal",
               arms=("up", "out") if kick else ("out", "out"), blink=blink_at(t), t=t)
    typewriter(img, "Now von Neumann's", t, line["words"][0]["start"] - 0.05, t_obs - 0.05, 420, 250,
               "type", 86, CREAM, cursor=t < t_obs)
    if t >= t_obs:
        stamp(img, "OBSOLETE", t, t_obs, W / 2 - 100, 610, "shade", 150, (230, 40, 40), angle=-12,
              from_scale=2.4)
        if t - t_obs < 0.25:
            ctx.fx["shake"] = ((hash01(ctx.frame) - 0.5) * 34, (hash01(ctx.frame, 1) - 0.5) * 34)
    ctx.fx["scan"] = 0.9


# --- L28: Sharp left turn and there you are -------------------------------------------------------
def L28(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_left = ctx.at("left")
    t_turned = t_left + 0.42
    if t < t_turned:
        img.paste((30, 30, 36), (0, 0, W, H))
        d = ImageDraw.Draw(img, "RGBA")
        # road
        d.polygon([(760, 380), (1160, 380), (1700, H), (220, H)], fill=(50, 50, 58))
        for k in range(8):
            y0 = 400 + ((k * 90 + t * 700) % 700)
            y1 = y0 + 40
            w0 = 4 + (y0 - 380) / 30
            d.polygon([(W / 2 - w0, y0), (W / 2 + w0, y0), (W / 2 + w0 * 1.1, y1), (W / 2 - w0 * 1.1, y1)],
                      fill=(255, 214, 60, 255))
        # the sign
        sx, sy, s = 1450, 420, 170
        d.rectangle((sx - 9, sy, sx + 9, sy + 420), fill=(160, 160, 170))
        d.polygon([(sx, sy - s), (sx + s, sy), (sx, sy + s), (sx - s, sy)], fill=(255, 205, 40),
                  outline=(20, 20, 20), width=8)
        d.line((sx + 50, sy + 70, sx + 50, sy - 20, sx - 50, sy - 20), fill=(20, 20, 20), width=22)
        d.polygon([(sx - 90, sy - 20), (sx - 45, sy - 60), (sx - 45, sy + 20)], fill=(20, 20, 20))
        draw_clawd(img, W / 2, 900, 12, walk=(t * 3) % 1, eyes="normal", t=t)
        pop_words(img, sub(line, 0, 3), t, 700, 200, "anton", 130, 1300, color=CREAM,
                  active=YELLOW, upper=True)
        if t >= t_left:
            k = ease_in_out(clamp((t - t_left) / 0.42))
            ctx.fx["angle"] = 90 * k
            ctx.fx["zoom"] = 1 + 0.35 * math.sin(k * math.pi)
            ctx.fx["rgb"] = 14 * math.sin(k * math.pi)
        return
    # ...and there you are
    bg.sunburst(img, t, W / 2, 1050, n=22, c1=(40, 10, 50), c2=(80, 20, 90), speed=0.5)
    k = ease_out_back(clamp((t - t_turned) / 0.35), 2.2)
    draw_clawd(img, W / 2, H + 260 - 330 * k, 46, eyes="wide", blink=0, t=t)
    pop_words(img, sub(line, 3), t, W / 2, 230, "anton", 170, 1700, color=CREAM, active=PINK,
              upper=True)
    if t_turned <= t < t_turned + 0.2:
        ctx.fx["flash"] = 0.4 * (1 - (t - t_turned) / 0.2)


# --- L29: Without a single `cdr` -------------------------------------------------------------------
CELLS = ["'doom", "'foom", "'boom", "'loom"]


def wizard_hat(img, x, y, s):
    d = ImageDraw.Draw(img, "RGBA")
    d.polygon([(x - 90 * s, y), (x + 90 * s, y), (x + 20 * s, y - 190 * s)], fill=(110, 60, 200, 255))
    d.rectangle((x - 110 * s, y - 14 * s, x + 110 * s, y + 6 * s), fill=(80, 40, 160, 255))
    for (qx, qy) in ((-20, -60), (15, -110), (-35, -25)):
        sprite(img, "sparkle", x + qx * s, y + qy * s, px=max(1, int(3 * s)), tint=(255, 230, 120))


def L29(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_cdr = ctx.at("`cdr`")
    img.paste((16, 8, 28), (0, 0, W, H))
    bg.matrix_rain(img, t, color=(200, 120, 255), chars="()", size=34, speed=10,
                   density=0.9 if t < line["end"] else 1.0, alpha=0.55)
    d = ImageDraw.Draw(img, "RGBA")
    # cons cells: [car | cdr] -> ...
    cut = t >= t_cdr
    for i, sym in enumerate(CELLS):
        drift = (t - t_cdr) * (i - 1.5) * 60 if cut else 0
        x0 = 260 + i * 380 + drift
        y0 = 470 + (math.sin(t * 3 + i) * 14 if cut else 0)
        d.rectangle((x0, y0, x0 + 150, y0 + 100), fill=(40, 24, 70, 255), outline=(230, 200, 255, 255),
                    width=5)
        d.rectangle((x0 + 150, y0, x0 + 250, y0 + 100), fill=(40, 24, 70, 255),
                    outline=(230, 200, 255, 255), width=5)
        draw_text(img, sym, x0 + 75, y0 + 50, "mono", 34, (255, 210, 120))
        draw_text(img, "car", x0 + 75, y0 - 26, "mono", 24, (170, 150, 210))
        draw_text(img, "cdr", x0 + 200, y0 - 26, "mono", 24, (170, 150, 210))
        if i < len(CELLS) - 1:
            if not cut:
                d.line((x0 + 200, y0 + 50, x0 + 372, y0 + 50), fill=(230, 200, 255, 255), width=6)
                d.polygon([(x0 + 380, y0 + 50), (x0 + 360, y0 + 38), (x0 + 360, y0 + 62)],
                          fill=(230, 200, 255, 255))
            else:
                k = clamp((t - t_cdr) / 0.3)
                d.line((x0 + 175, y0 + 25, x0 + 225, y0 + 75), fill=(255, 70, 90, int(255 * k)), width=8)
                d.line((x0 + 225, y0 + 25, x0 + 175, y0 + 75), fill=(255, 70, 90, int(255 * k)), width=8)
        else:
            d.line((x0 + 160, y0 + 90, x0 + 240, y0 + 10), fill=(230, 200, 255, 255), width=5)
    # wizard Clawd (SICP vibes)
    bi, _ = ctx.beat_phase()
    arms = ("up", "up") if not cut else (("up", "out") if bi % 2 else ("out", "up"))
    cx, feet = 300, 980
    draw_clawd(img, cx, feet - (ctx.bounce(16) if t > line["end"] else 0), 11, eyes="happy" if cut else "normal",
               arms=arms, blink=blink_at(t), t=t)
    wizard_hat(img, cx, feet - 10 * 11 - 4, 0.8)
    # typography
    pop_words(img, sub(line, 0, 3), t, W / 2 - 170, 250, "grotesk", 100, 1300, color=CREAM)
    if t >= t_cdr - 0.03:
        k = ease_out_back(clamp((t - t_cdr) / 0.25), 2.0)
        pill = Image.new("RGBA", (330, 150), (0, 0, 0, 0))
        pd = ImageDraw.Draw(pill)
        pd.rounded_rectangle((0, 0, 329, 149), radius=24, fill=(60, 30, 90, 255),
                             outline=(200, 140, 255, 255), width=5)
        li, ox, oy = text_layer("cdr", "mono", 100, (255, 210, 120))
        pill.paste(li, (int(165 - li.size[0] / 2), int(75 - li.size[1] / 2)), li)
        paste(img, pill, 1500, 250, scale=k, alpha=clamp(k))
    if t >= t_cdr + 0.6:
        typewriter(img, "> (cdr '())  ;=> NIL", t, t_cdr + 0.6, t_cdr + 1.4, 640, 800, "mono", 52,
                   GREEN, cursor=True)


# --- L30: Gato, please don't let me go ---------------------------------------------------------------
def L30(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_gato = line["words"][0]["start"]
    t_pl = ctx.at("please")
    t_go = ctx.at("go")
    t_drop = t_go + 0.38
    bg.gradient(img, (40, 26, 14), (12, 8, 4))
    area = chat_window(img, (220, 80, 1700, 690), "Gato", "gato", t)
    bubble(img, area[0], area[1] + 10, "meow. I can do 604 tasks :3", "gato", "left", t,
           t_gato + 0.25, size=42)
    sung = [w["w"] for w in line["words"][1:] if t + 0.05 >= w["start"]]
    if sung:
        bubble(img, area[2], area[1] + 130, " ".join(sung), "gato", "right", t, t_pl - 0.05, size=46)
    # Clawd dangling below the window's bottom edge, one hand held by Gato's paw
    edge_y = 690
    u = 16
    cat_x = 1360
    paw_x, paw_y = cat_x - 70, edge_y + 120
    fall = max(0.0, t - t_drop)
    sway = math.sin(t * 3.2) * 12
    cx = paw_x - 8 * u + sway
    feet = paw_y + 9 * u + 0.5 * 3200 * fall * fall
    d = ImageDraw.Draw(img, "RGBA")
    # the cat, peeking over the edge, arm reaching down
    sprite(img, "cat", cat_x, edge_y - 50, px=14)
    if fall == 0:
        d.line((cat_x - 30, edge_y + 10, paw_x + sway * 0.5, paw_y - 20), fill=(230, 140, 60, 255),
               width=34)
    draw_clawd(img, cx, feet, u, eyes="wide" if fall == 0 else "x", arms=("up", "up"),
               angle=0 if fall == 0 else fall * 400, t=t)
    if fall == 0:
        sprite(img, "paw", paw_x + sway * 0.5, paw_y, px=10)
    else:
        sprite(img, "paw", paw_x, paw_y - 60 * fall, px=10, alpha=clamp(1 - fall * 4))
    wavy(img, "GATO", t, 520, 900, "anton", 210, amp=12, color=(255, 190, 110), stroke=6,
         stroke_fill=(60, 30, 0))


# --- L31: I'm upping my P(doom) as paperclips fill the room -----------------------------------------
@lru_cache(maxsize=64)
def clip_rot(height, ang_bucket):
    return paperclip_img(height).rotate(ang_bucket * 15, resample=Image.BICUBIC, expand=True)


def L31(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_as = ctx.at("as")
    if chorus_hook(ctx, t_as - 0.06, 0.86, 0.92, theme_hue=0.55):
        return
    t_room = ctx.at("room")
    bg.gradient(img, (40, 42, 52), (16, 16, 22))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, 860, W, H), fill=(28, 28, 36))
    k = clamp((t - t_as) / (ctx.scene.end + 0.1 - t_as))
    level = lerp(H + 40, 430, ease_in(k) * 0.4 + k * 0.6)
    # Clawd, slowly being buried
    draw_clawd(img, W / 2, 1000, 20, eyes="wide", arms=("up", "up"), blink=0, t=t)
    # the pile (static seeded clips, only those below the current level)
    for i in range(420):
        y = 430 + hash01(i, 1) * 700
        if y < level:
            continue
        x = hash01(i, 2) * (W + 100) - 50
        im = clip_rot(110, int(hash01(i, 3) * 24))
        paste(img, im, x, y)
    # falling clips
    for i in range(46):
        sp = 900 + 500 * hash01(i, 5)
        y = ((t - t_as) * sp + hash01(i, 6) * 1200) % 1400 - 200
        if y > level:
            continue
        x = hash01(i, 7) * W
        im = clip_rot(110, int((hash01(i, 8) * 24 + t * 6)) % 24)
        paste(img, im, x, y)
    pop_words(img, sub(line, 4), t, W / 2, 200, "anton", 130, 1600, color=CREAM,
              active=(210, 215, 228), upper=True, shadow=(6, 6, (10, 10, 20)))
    if t >= t_room:
        ctx.fx["shake"] = ((hash01(ctx.frame) - 0.5) * 12, (hash01(ctx.frame, 1) - 0.5) * 12)


# --- L32: Killswitch guys on PTO, now there's nowhere left to go ------------------------------------
def L32(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_now = ctx.at("now")
    t_nowhere = ctx.at("nowhere")
    bg.gradient(img, (26, 12, 16), (8, 4, 6))
    d = ImageDraw.Draw(img, "RGBA")
    # the big red button under glass, with an out-of-office note
    bx, by = 1260, 700
    d.rectangle((bx - 220, by + 60, bx + 220, by + 320), fill=(60, 60, 70, 255))
    d.rectangle((bx - 240, by + 40, bx + 240, by + 70), fill=(90, 90, 100, 255))
    pressed = int(t * 4.4) % 2 == 0 and t < t_now
    d.ellipse((bx - 150, by - 90 + (18 if pressed else 0), bx + 150, by + 60), fill=(200, 20, 30, 255),
              outline=(120, 0, 10, 255), width=8)
    draw_text(img, "KILL SWITCH", bx, by + 180, "silkb", 50, (255, 230, 230))
    d.rectangle((bx - 200, by - 190, bx + 200, by + 50), outline=(200, 230, 255, 120), width=6)
    # sticky note
    note = Image.new("RGBA", (300, 230), (255, 236, 110, 255))
    for k, txt in enumerate(["OUT OF", "OFFICE", "back monday"]):
        li, ox, oy = text_layer(txt, "type", 40 if k < 2 else 30, (40, 30, 10))
        note.paste(li, (int(150 - li.size[0] / 2), 20 + k * 58), li)
    paste(img, note, bx + 250, by - 200, angle=12)
    sprite(img, "palm", bx + 330, by - 60, px=6, angle=12)
    # auto-reply
    t_pto = ctx.at("pto")
    if t >= t_pto:
        k = ease_out(clamp((t - t_pto) / 0.25))
        x = lerp(-700, 90, k)
        d.rounded_rectangle((x, 380, x + 700, 560), radius=14, fill=(245, 245, 250, 255))
        draw_text(img, "Auto-reply: Re: URGENT", x + 30, 430, "grotesk", 34, (40, 40, 60), anchor="ls")
        draw_text(img, "I'm on PTO until 2031.", x + 30, 490, "grotesk", 32, (80, 80, 100),
                  anchor="ls")
    # Clawd hammering the button
    look = 0
    if t >= t_nowhere:
        look = -1 if int(t * 3) % 2 == 0 else 1
    draw_clawd(img, bx - 330, 1010, 14, eyes="wide", arms=("up", "down") if pressed else ("out", "out"),
               look=look, t=t)
    if t < t_now:
        if pressed:
            draw_text(img, "ERROR: operator unavailable", 440, 640, "mono", 34, RED)
    # walls closing in
    if t >= t_now:
        k = ease_in(clamp((t - t_now) / (ctx.scene.end - t_now)))
        m = int(k * 300)
        for (x0, y0, x1, y1) in ((0, 0, m * 1.6, H), (W - m * 1.6, 0, W, H), (0, 0, W, m), (0, H - m, W, H)):
            d.rectangle((x0, y0, x1, y1), fill=(4, 2, 4, 255))
        # hazard stripes on the inner edges
        for yy in range(0, H, 60):
            d.polygon([(m * 1.6, yy), (m * 1.6 + 26, yy + 30), (m * 1.6, yy + 60)], fill=(255, 200, 0, 255))
            d.polygon([(W - m * 1.6, yy), (W - m * 1.6 - 26, yy + 30), (W - m * 1.6, yy + 60)],
                      fill=(255, 200, 0, 255))
    pop_words(img, sub(line, 0, 4), t, W / 2, 200, "anton", 120, 1500, color=CREAM, active=RED,
              upper=True)
    if t >= t_now - 0.03:
        pop_words(img, sub(line, 4), t, W / 2, 330, "anton", 96, 1500, color=(255, 200, 0),
                  upper=True)


# --- L33: Too late now, we lit the fuse -----------------------------------------------------------------
def fuse_path():
    """From Clawd's match (bottom-left) to the bomb's fuse (top of the bomb sprite)."""
    pts = []
    for k in range(120):
        s = k / 119
        x = lerp(360, 1612, s)
        y = lerp(900, 574, s) - math.sin(s * math.pi * 1.5) * 150
        pts.append((x, y))
    return pts


def L33(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_lit = ctx.at("lit")
    t_boom = ctx.scene.end - 0.12
    bg.gradient(img, (20, 10, 26), (6, 2, 8))
    d = ImageDraw.Draw(img, "RGBA")
    pts = fuse_path()
    burn = clamp((t - t_lit) / (t_boom - t_lit))
    ib = int(burn * (len(pts) - 1))
    if t < t_lit:
        ib = 0
    d.line(pts[ib:], fill=(210, 180, 120, 255), width=10, joint="curve")
    if ib > 0:
        d.line(pts[:ib + 1], fill=(40, 30, 30, 255), width=6, joint="curve")
    # bomb at the end of the fuse
    bx, by = 1560, 660
    shake = 0
    if burn > 0.7:
        shake = (hash01(ctx.frame) - 0.5) * 16 * (burn - 0.7) / 0.3
    sprite(img, "bomb", bx + shake, by, px=22)
    # the spark
    if t >= t_lit:
        sx, sy = pts[min(ib, len(pts) - 1)]
        for q in range(14):
            a = hash01(ctx.frame, q) * 2 * math.pi
            r = 10 + 40 * hash01(ctx.frame, q, 1)
            d.line((sx, sy, sx + math.cos(a) * r, sy + math.sin(a) * r),
                   fill=(255, 230, 120, 255), width=4)
        d.ellipse((sx - 16, sy - 16, sx + 16, sy + 16), fill=(255, 255, 220, 255))
        ctx.fx["bloom"] = 0.6
    # Clawd with the match, then running
    run = clamp((t - t_lit - 0.25) / 1.0)
    cx = lerp(250, -200, ease_in(run))
    draw_clawd(img, cx, 1000, 12, eyes="happy" if t < t_lit + 0.2 else "wide",
               arms=("up", "out"), walk=(t * 4) % 1 if run > 0 else None, t=t)
    if run <= 0:
        d.line((cx + 110, 860, cx + 150, 800), fill=(200, 160, 110, 255), width=8)
        if t >= t_lit - 0.4:
            sprite(img, "flame_a", cx + 152, 780, px=3)
    pop_words(img, sub(line, 0, 3), t, W / 2, 190, "anton", 130, 1600, color=CREAM, upper=True)
    if t >= ctx.at("we") - 0.03:
        txt = " ".join(w["w"].upper() for w in line["words"][3:] if t + 0.05 >= w["start"])
        col = mix((255, 200, 60), (255, 60, 30), 0.5 + 0.5 * math.sin(t * 20))
        shaky(img, txt, t, W / 2, 375, "anton", 120, col, amp=4)
    if burn > 0.95:
        ctx.fx["flash"] = (burn - 0.95) / 0.05


# --- L34: Orthogonality thesis blues ------------------------------------------------------------------
def L34(ctx):
    t, img, line = ctx.t, ctx.img, ctx.line
    t_th = ctx.at("thesis")
    t_bl = ctx.at("blues")
    bg.gradient(img, (6, 12, 40), (2, 4, 14))
    bg.rain(img, t, color=(110, 150, 255), n=180, alpha=110)
    d = ImageDraw.Draw(img, "RGBA")
    # orthogonal axes: intelligence vs goals
    ox, oy = 980, 880
    d.line((ox, oy, 1820, oy), fill=(160, 190, 255, 255), width=6)
    d.line((ox, oy, ox, 420), fill=(160, 190, 255, 255), width=6)
    d.polygon([(1840, oy), (1810, oy - 16), (1810, oy + 16)], fill=(160, 190, 255, 255))
    d.polygon([(ox, 400), (ox - 16, 430), (ox + 16, 430)], fill=(160, 190, 255, 255))
    d.rectangle((ox, oy - 40, ox + 40, oy), outline=(160, 190, 255, 255), width=4)
    draw_text(img, "intelligence →", 1820, oy + 50, "mono", 32, (160, 190, 255), anchor="rs")
    draw_text(img, "goals ↑", ox - 20, 440, "mono", 32, (160, 190, 255), anchor="rs")
    n_pts = int(clamp((t - ctx.scene.start) / 3.0) * 26)
    for i in range(n_pts):
        x = ox + 60 + hash01(i, 1) * 740
        y = oy - 60 - hash01(i, 2) * 400
        c = hsv(0.55 + hash01(i, 3) * 0.2, 0.6, 1)
        d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(*c, 230))
    # spotlight + blue Clawd
    d.polygon([(470, 0), (270, 1000), (720, 1000)], fill=(140, 170, 255, 40))
    blue = mix(CLAWD, (80, 120, 240), smoothstep(113.8, 116.5, t))
    draw_clawd(img, 490, 990, 15, color=blue, eyes="sad", blink=blink_at(t, 1.1), t=t)
    for q in range(4):
        ph = ((t * 0.6) + q / 4) % 1.0
        draw_text(img, "♪" if q % 2 else "♫", 600 + q * 30 + ph * 120, 780 - ph * 360, "dejavu", 54,
                  (170, 200, 255), alpha=math.sin(ph * math.pi))
    # typography: ORTHOGONALITY typed across the long note
    t0 = line["words"][0]["start"]
    word = "ORTHOGONALITY"
    n = int(clamp((t - t0) / (t_th - 0.1 - t0)) * len(word) + 0.999) if t >= t0 else 0
    if n:
        draw_text(img, word[:n], 90, 250, "monoton", 128, (120, 190, 255), anchor="ls")
    if t >= t_th - 0.03:
        draw_text(img, "thesis", 100, 360, "grotesk", 70, CREAM, anchor="ls",
                  alpha=clamp((t - t_th) / 0.15))
    if t >= t_bl - 0.03:
        k = ease_out(clamp((t - t_bl) / 0.3))
        wavy(img, "BLUES", t, 470, 560, "anton", int(200 * (0.8 + 0.2 * k)), amp=16, freq=0.8, speed=3,
             color=(90, 150, 255), stroke=6, stroke_fill=(10, 20, 60))
    ctx.fx["bloom"] = 0.35


SCENES = {"L26": L26, "L27": L27, "L28": L28, "L29": L29, "L30": L30, "L31": L31, "L32": L32,
          "L33": L33, "L34": L34}
