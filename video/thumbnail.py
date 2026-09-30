"""Render the YouTube thumbnail (1280x720 PNG).

Usage: python thumbnail.py <out.png>
"""
import math
import sys

import numpy as np
from PIL import Image

from mv import bg, fx
from mv.clawd import draw_clawd
from mv.core import CREAM, H, RED, W, YELLOW
from mv.gfx import draw_text, glow_layer, to_np
from mv.scenes.common import gauge, lightning


def main(out):
    img = Image.new("RGB", (W, H))
    bg.sunburst(img, 0.4, 560, 620, n=22, c1=(40, 4, 14), c2=(92, 10, 32), speed=0)
    # Clawd, huge, with shinigami-red eyes and sparks
    u = 40
    cx, eye_y = 560, 470
    feet = eye_y + 7 * u
    draw_clawd(img, cx, feet, u, eyes="red", arms=("up", "up"), t=0.0)
    # real red glow around the eyes: blurred emitter layer added on top
    import cv2
    glow = np.zeros((H, W, 3), np.uint8)
    for ex in (5, 12):
        x = int(cx + (ex - 9) * u)
        cv2.rectangle(glow, (x - 6, eye_y - u - 6), (x + u + 6, eye_y + u + 6), (255, 40, 30), -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 38)
    arr0 = cv2.addWeighted(to_np(img), 1.0, glow, 2.2, 0)
    img = Image.fromarray(arr0)
    for k in range(5):
        a = -math.pi / 2 + (k - 2) * 0.55
        lightning(img, cx, feet - 11 * u, cx + math.cos(a) * 420, feet - 11 * u + math.sin(a) * 300,
                  0.1 * k, color=(255, 120, 120), width=6, seed=k, jag=34)
    # gauge pinned in the red
    gauge(img, 1420, 760, 380, 0.99, 0.0)
    # title
    draw_text(img, "I'M UPPING MY", 1420, 150, "anton", 120, CREAM, stroke=10,
              stroke_fill=(0, 0, 0))
    draw_text(img, "P(DOOM)", 1420, 330, "anton", 250, RED, stroke=14, stroke_fill=(0, 0, 0),
              shadow=(12, 12, (0, 0, 0)))
    draw_text(img, "99%", 1420, 880, "anton", 150, YELLOW, stroke=10, stroke_fill=(0, 0, 0))
    arr = to_np(img)
    arr = glow_layer(arr, 40, 0.5, 170)
    arr = fx.rgb_split(arr, 5)
    arr = fx.crt(arr, scan=0.35, vignette=0.9)
    Image.fromarray(arr).resize((1280, 720), Image.LANCZOS).save(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "thumbnail.png")
