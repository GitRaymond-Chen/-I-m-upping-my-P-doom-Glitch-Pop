"""Pixel-art prop sprites (ASCII grids) plus a few procedural vector props."""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from .gfx import parse_sprite

PAL = {
    "K": (22, 20, 30), "W": (250, 248, 240), "E": (245, 232, 205),
    "O": (217, 119, 87), "o": (168, 82, 58), "F": (255, 150, 40),
    "R": (235, 45, 60), "r": (140, 20, 35), "P": (255, 110, 180),
    "C": (61, 245, 255), "c": (20, 120, 150), "G": (70, 210, 90), "g": (30, 110, 50),
    "Y": (255, 214, 10), "y": (170, 120, 20), "B": (60, 110, 255), "b": (25, 40, 120),
    "S": (200, 205, 215), "s": (95, 100, 115), "d": (55, 58, 70), "M": (160, 90, 255),
    "N": (230, 140, 60), "n": (150, 80, 30), "T": (120, 125, 140), "L": (230, 230, 235),
}

_ART = {
    "mushroom": """
....RRRRRR....
..RRWWRRRRRR..
.RRWWWWRRWWRR.
.RRWWWWRWWWWR.
RRRRWWRRWWWWRR
RWWRRRRRRWWRRR
RWWWRRRRRRRRRR
RRWWRRRRRRWWRR
.RRRRRRRRRWWR.
...EEEEEEEE...
...EEKEEKEE...
...EEKEEKEE...
...EEEEEEEE...
....EEEEEE....""",
    "heart": """
.RR...RR.
RRRR.RRRR
RRWRRRRRR
RWRRRRRRR
RRRRRRRRR
.RRRRRRR.
..RRRRR..
...RRR...
....R....""",
    "skull": """
...WWWWW...
..WWWWWWW..
.WWWWWWWWW.
.WWWWWWWWW.
.WKKWWWKKW.
.WKKWWWKKW.
.WWWWKWWWW.
..WWWWWWW..
...WKWKW...
...WWWWW...""",
    "crown": """
Y....Y....Y
YY..YYY..YY
YYY.YYY.YYY
YYYYYYYYYYY
YRYYYCYYYRY
YYYYYYYYYYY
yyyyyyyyyyy""",
    "rocket": """
....R....
...RRR...
...WWW...
..WWWWW..
..WCCCW..
..WCcCW..
..WCCCW..
..WWWWW..
..WWWWW..
.RWWWWWR.
RRWWWWWRR
RR.WWW.RR
R..RRR..R""",
    "flame_a": """
...Y....
..YY....
..YRY.Y.
.YRRY.Y.
.YRRRYY.
YRRRRRYY
YRRFFRRY
YRFFFFRY
.YRFFRY.
..YYYY..""",
    "flame_b": """
....Y...
....YY..
.Y.YRY..
.Y.YRRY.
.YYRRRY.
YYRRRRRY
YRRFFRRY
YRFFFFRY
.YRFFRY.
..YYYY..""",
    "gpu": """
SSSSSSSSSSSSSSSSSS
SddddddddddddddddS
SdsTTTsddddsTTTsdS
STdTsTdTddTdTsTdTS
STsTTTsTddTsTTTsTS
STdTsTdTddTdTsTdTS
SdsTTTsddddsTTTsdS
SddddddddddddddddS
SSSSSSSSSSSSSSSSSS
.YY.YY.YY.YY.YY...""",
    "chip": """
.S.S.S.S.
SdddddddS
.dCCCCCd.
SdCddcCdS
.dCdCdCd.
SdCcddCdS
.dCCCCCd.
SdddddddS
.S.S.S.S.""",
    "cat": """
.K..........K.
.KK........KK.
.KNK......KNK.
.KNNKKKKKKNNK.
.KNNNNNNNNNNK.
KNNWKNNNNNWKNK
KNNKKNNNNNKKNK
KNNNNNNPPNNNNK
KNNnNNNNNNnNNK
.KNNNNNNNNNNK.
..KKKKKKKKKK..""",
    "paw": """
.KK.KK.
KNNKNNK
KNNKNNK
.KKKKK.
KNNNNNK
KNPPPNK
KNPPPNK
KNNNNNK
.KNNNK.""",
    "bomb": """
..........Y.
.........Y.Y
........n...
.......n....
....ssss....
..ssssssss..
.ssWsssssss.
.sWssssssss.
ssssssssssss
ssssssssssss
.ssssssssss.
..ssssssss..
...ssssss...""",
    "thumb_up": """
.....KK.....
....KYYK....
....KYYK....
...KYYYK....
KKKKYYYYKKK.
KYYKYYYYYYYK
KYYKYYYYYYK.
KYYKYYYYYYYK
KYYKYYYYYYK.
KYYKYYYYYYYK
KKKKKKKKKKK.""",
    "palm": """
..GG..GG....
.GGGGGGGG...
GG..GnGG.GG.
G..GGnGGG..G
...G.n..G...
.....n......
.....n......
....n.......
....n.......
....n.......
...nn.......
.YYYYYYYY...""",
    "chinchilla": """
..SS......SS..
.SLLS....SLLS.
.SLPS....SPLS.
..SSSSSSSSSS..
.SSSSSSSSSSSS.
SSSKSSSSSSKSSS
SSSKSSSSSSKSSS
SSSSSSKKSSSSSS
.SSSSSSSSSSSS.
.SSLLLLLLLLSS.
..SLLLLLLLLS..
...SS....SS...""",
    "bolt": """
....YY.
...YY..
..YY...
.YYYYY.
...YY..
..YY...
.YY....
YY.....""",
    "sparkle": """
..W..
..W..
WWWWW
..W..
..W..""",
    "star4": """
...W...
...W...
..WWW..
WWWWWWW
..WWW..
...W...
...W...""",
    "barrier": """
YYKKYYKKYYKKYYKKYYKK
YKKYYKKYYKKYYKKYYKKY
KKYYKKYYKKYYKKYYKKYY
.S................S.
.S................S.
.S................S.
SSS..............SSS""",
    "eye": """
....KKKKKK....
..KKWWWWWWKK..
.KWWWWCCWWWWK.
KWWWWCCKCWWWWK
KWWWWCKKKCWWWK
.KWWWCCKCWWWK.
..KKWWCCWWKK..
....KKKKKK....""",
    "coin": """
..YYYY..
.YyyyyY.
YyYYYYyY
YyYyyYyY
YyYYYYyY
YyyYYyyY
.YyyyyY.
..YYYY..""",
    "floppy": """
BBBBBBBBBB.
BBSSSSSSBBB
BBSSSSSSBBB
BBSSSSSSBBB
BBBBBBBBBBB
BBWWWWWWWBB
BBWKKKKKWBB
BBWWWWWWWBB
BBBBBBBBBBB""",
}

SPRITES = {k: parse_sprite([r for r in v.strip("\n").split("\n")], PAL)
           for k, v in _ART.items()}
SPRITES["thumb_down"] = SPRITES["thumb_up"][::-1].copy()


def _pixel_disk(r, col, crater=None, seed=1):
    size = 2 * r + 1
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((xx - r) ** 2 + (yy - r) ** 2)
    arr = np.zeros((size, size, 4), np.uint8)
    arr[d <= r] = (*col, 255)
    if crater:
        rng = np.random.default_rng(seed)
        for _ in range(7):
            cx, cy = rng.integers(r // 3, size - r // 3, 2)
            cr = rng.integers(1, max(2, r // 4))
            m = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) <= cr) & (d <= r - 1)
            arr[m] = (*crater, 255)
    # shade the lower-right rim
    rim = (d <= r) & (d > r - max(1, r // 6)) & ((xx - r) + (yy - r) > 0)
    arr[rim, :3] = (arr[rim, :3] * 0.7).astype(np.uint8)
    return arr


SPRITES["moon"] = _pixel_disk(14, (235, 232, 220), crater=(185, 182, 175), seed=3)
SPRITES["sun"] = _pixel_disk(16, (255, 90, 120))


@lru_cache(maxsize=64)
def paperclip_img(height, color=(205, 210, 222), width_px=None, shade=(120, 125, 140)):
    """Vector paperclip (Gem clip) as an RGBA image `height` px tall."""
    s = height / 3.2
    lw = max(2, int(width_px or height * 0.055))
    pad = lw * 2
    W_, H_ = int(1.0 * s + 2 * pad), int(3.2 * s + 2 * pad)
    img = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def P(x, y):
        return (pad + x * s, pad + y * s)

    def arc(cx, cy, r, a0, a1, n=12):
        return [P(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
                  cy + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]

    pts = [P(0.62, 1.0), P(0.62, 2.35)]
    pts += arc(0.46, 2.35, 0.16, 0, 180)
    pts += [P(0.30, 0.45)]
    pts += arc(0.61, 0.45, 0.31, 180, 360)
    pts += [P(0.92, 2.55)]
    pts += arc(0.46, 2.55, 0.46, 0, 180)
    pts += [P(0.0, 0.9)]
    d.line(pts, fill=(*shade, 255), width=lw + 2, joint="curve")
    d.line(pts, fill=(*color, 255), width=lw, joint="curve")
    return img
