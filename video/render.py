"""Render the "Upping My P (Doom)" music video.

  python render.py --stills 10.5,31.6,70.1          # PNG stills for review
  python render.py --sheet 9.8:30:0.5               # contact sheet of a time range
  python render.py --out out.mp4 [--from 0 --to 30]  # video (parallel workers)
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Process

import imageio_ffmpeg
import numpy as np
from PIL import Image

from mv.core import FPS, H, W, Timeline
from mv.director import build_schedule, render_frame

ROOT = os.path.dirname(os.path.abspath(__file__))
SONG = os.path.join(os.path.dirname(ROOT), "Upping My P (Doom).mp3")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def load():
    from mv.scenes import SCENE_FNS
    tl = Timeline()
    return tl, build_schedule(tl), SCENE_FNS


def worker(f0, f1, path, crf, preset):
    tl, sched, fns = load()
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-threads", "2", "-g", "60", "-bf", "2", "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(f0, f1):
        arr = render_frame(tl, sched, fns, f)
        p.stdin.write(memoryview(np.ascontiguousarray(arr)))
        if (f - f0) % 150 == 0:
            el = time.time() - t0
            print(f"[{os.getpid()}] frame {f} ({f - f0}/{f1 - f0}) {el:.0f}s", flush=True)
    p.stdin.close()
    p.wait()


def render_video(out, t_from, t_to, workers, crf, preset):
    tl = Timeline()
    t_to = min(t_to, tl.duration)
    f_from, f_to = int(round(t_from * FPS)), int(round(t_to * FPS))
    n = f_to - f_from
    tmpdir = out + ".parts"
    os.makedirs(tmpdir, exist_ok=True)
    chunk = (n + workers - 1) // workers
    procs, parts = [], []
    for i in range(workers):
        a = f_from + i * chunk
        b = min(f_to, a + chunk)
        if a >= b:
            break
        path = os.path.join(tmpdir, f"part{i:02d}.mp4")
        parts.append(path)
        pr = Process(target=worker, args=(a, b, path, crf, preset))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            sys.exit(f"worker failed with exit code {pr.exitcode}")
    listfile = os.path.join(tmpdir, "list.txt")
    with open(listfile, "w") as f:
        for pth in parts:
            f.write(f"file '{os.path.abspath(pth)}'\n")
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", listfile,
           "-ss", f"{f_from / FPS:.3f}", "-t", f"{n / FPS:.3f}", "-i", SONG,
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
           "-shortest", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    for pth in parts:
        os.remove(pth)
    os.remove(listfile)
    os.rmdir(tmpdir)


def stills(times, outdir):
    tl, sched, fns = load()
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for t in times:
        f = int(round(t * FPS))
        t0 = time.time()
        arr = render_frame(tl, sched, fns, f)
        p = os.path.join(outdir, f"still_{t:07.2f}.png")
        Image.fromarray(arr).save(p)
        paths.append(p)
        print(f"{p}  ({(time.time() - t0) * 1000:.0f} ms)")
    return paths


def sheet(t0, t1, step, out, cols=4):
    tl, sched, fns = load()
    times = list(np.arange(t0, t1 + 1e-6, step))
    tw, th = W // 4, H // 4
    rows = (len(times) + cols - 1) // cols
    img = Image.new("RGB", (cols * tw, rows * (th + 28)), (0, 0, 0))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for i, t in enumerate(times):
        arr = render_frame(tl, sched, fns, int(round(t * FPS)))
        im = Image.fromarray(arr).resize((tw, th), Image.LANCZOS)
        x, y = (i % cols) * tw, (i // cols) * (th + 28)
        img.paste(im, (x, y + 28))
        d.text((x + 6, y + 6), f"{t:.2f}s", fill=(255, 255, 0))
    img.save(out)
    print(out)


def grid(times, out, cols=2, scale=2):
    """Half-res (scale=2) review grid of arbitrary times."""
    tl, sched, fns = load()
    tw, th = W // scale, H // scale
    rows = (len(times) + cols - 1) // cols
    img = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for i, t in enumerate(times):
        arr = render_frame(tl, sched, fns, int(round(t * FPS)))
        im = Image.fromarray(arr).resize((tw, th), Image.LANCZOS)
        x, y = (i % cols) * tw, (i // cols) * th
        img.paste(im, (x, y))
        d.rectangle((x, y, x + 70, y + 18), fill=(0, 0, 0))
        d.text((x + 4, y + 4), f"{t:.2f}s", fill=(255, 255, 0))
    img.save(out)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills")
    ap.add_argument("--grid")
    ap.add_argument("--name", default="grid")
    ap.add_argument("--sheet")
    ap.add_argument("--out")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=1e9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--dir", default=os.environ.get("MV_OUT", "/tmp/mv_frames"))
    a = ap.parse_args()
    if a.stills:
        stills([float(x) for x in a.stills.split(",")], a.dir)
    elif a.grid:
        os.makedirs(a.dir, exist_ok=True)
        grid([float(x) for x in a.grid.split(",")], os.path.join(a.dir, f"{a.name}.png"))
    elif a.sheet:
        t0, t1, step = (float(x) for x in a.sheet.split(":"))
        sheet(t0, t1, step, os.path.join(a.dir, f"sheet_{t0:06.2f}_{t1:06.2f}.png"))
    elif a.out:
        t = time.time()
        render_video(a.out, a.t_from, a.t_to, a.workers, a.crf, a.preset)
        print(f"done in {time.time() - t:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
