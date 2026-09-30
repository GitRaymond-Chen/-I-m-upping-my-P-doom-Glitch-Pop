"""Cut a 9:16 YouTube Shorts version from the rendered 16:9 video.

The 16:9 frame sits full-width in the middle over a blurred, zoomed copy of itself,
with the title above and a credit below (clear of the Shorts UI overlay). The text is
drawn with PIL into a transparent PNG, since the bundled ffmpeg has no drawtext.

Usage: python make_short.py <master.mp4> <out.mp4> [start] [end]
"""
import os
import subprocess
import sys
import tempfile

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(ROOT, "fonts", "Anton-Regular.ttf")
MONO = os.path.join(ROOT, "fonts", "JetBrainsMono[wght].ttf")


def title_overlay(path):
    img = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    mono = ImageFont.truetype(MONO, 64)
    mono.set_variation_by_axes([700])
    for text, font, fill, y, stroke in (
        ("I'M UPPING MY", ImageFont.truetype(FONT, 120), (255, 255, 255), 330, 8),
        ("P(DOOM)", ImageFont.truetype(FONT, 230), (255, 42, 42), 600, 10),
        ("feat. Clawd", mono, (217, 119, 87), 1370, 6),
    ):
        d.text((540, y), text, font=font, fill=fill, anchor="ms", stroke_width=stroke,
               stroke_fill=(0, 0, 0))
    img.save(path)


def main():
    src, out = sys.argv[1], sys.argv[2]
    start = float(sys.argv[3]) if len(sys.argv) > 3 else 23.98   # "ChatGPT, please…"
    end = float(sys.argv[4]) if len(sys.argv) > 4 else 46.28     # end of chorus 1
    dur = end - start
    overlay = os.path.join(tempfile.mkdtemp(), "title.png")
    title_overlay(overlay)
    fc = (
        "[0:v]split=2[a][b];"
        "[a]scale=-2:1920,crop=1080:1920,boxblur=28:3,eq=brightness=-0.22:saturation=1.25[bg];"
        "[b]scale=1080:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[base];"
        "[base][1:v]overlay=0:0[v];"
        f"[0:a]afade=t=in:d=0.04,afade=t=out:st={dur - 0.6:.2f}:d=0.6[aud]"
    )
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src, "-i", overlay,
           "-filter_complex", fc, "-map", "[v]", "-map", "[aud]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    print(out)


if __name__ == "__main__":
    main()
