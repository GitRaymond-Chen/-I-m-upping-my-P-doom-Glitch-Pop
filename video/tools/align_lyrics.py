"""Align the known lyrics to ASR word timestamps.

Character-level Needleman-Wunsch between the lyric text and the ASR text (both
reduced to [a-z0-9]), so splits like "upping" <-> "up in" still line up. Every
lyric word gets the time span of the ASR characters it aligned to; words with
no aligned characters are interpolated from their neighbours.

Usage: python align_lyrics.py <lyrics.txt> <asr_words.json> <vocals.wav> <out_timeline.json>
"""
import json
import re
import sys

import librosa
import numpy as np

# How a lyric token is pronounced, for matching purposes only.
SPOKEN = {
    "1e30": "one e thirty",
    "7": "seven",
    "&": "and",
    "p(doom)": "p doom",
    "p(doom),": "p doom",
    "nvda": "nvidia",
    "`cdr`": "cdr",
}
NUMBERS = {"30": "thirty", "7": "seven", "100": "hundred", "1": "one"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse_lyrics(path):
    lines, section = [], None
    for raw in open(path, encoding="utf-8"):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        m = re.match(r"^\[(.+)\]$", raw)
        if m:
            section = m.group(1)
            continue
        words = [w for w in raw.split() if w != "|"]
        lines.append({"section": section, "text": raw, "words": words})
    return lines


def align(a, b, match=2, mismatch=-1, gap=-1):
    """Global alignment of strings a, b. Returns list of (i, j) matched index pairs."""
    n, m = len(a), len(b)
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up (gap in b), 2 left
    ptr[1:, 0] = 1
    ptr[0, 1:] = 2
    bb = np.frombuffer(b.encode(), dtype=np.uint8)
    for i in range(1, n + 1):
        ai = ord(a[i - 1])
        sub = np.where(bb == ai, match, mismatch)
        diag = score[i - 1, :-1] + sub
        up = score[i - 1, 1:] + gap
        row = score[i]
        # left moves depend on the current row, so resolve them sequentially
        best = np.maximum(diag, up)
        p = np.where(diag >= up, 0, 1).astype(np.int8)
        for j in range(1, m + 1):
            v = best[j - 1]
            lv = row[j - 1] + gap
            if lv > v:
                row[j] = lv
                ptr[i, j] = 2
            else:
                row[j] = v
                ptr[i, j] = p[j - 1]
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        d = ptr[i, j]
        if d == 0:
            if a[i - 1] == b[j - 1]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif d == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def main():
    lyr_path, asr_path, vocals, out = sys.argv[1:5]
    lines = parse_lyrics(lyr_path)
    asr = json.load(open(asr_path))

    # lyric chars -> (line, word)
    lchars, lowner = [], []
    for li, line in enumerate(lines):
        for wi, w in enumerate(line["words"]):
            spoken = SPOKEN.get(w.lower(), w)
            for c in norm(spoken):
                lchars.append(c)
                lowner.append((li, wi))
    # asr chars -> time
    achars, atime = [], []
    for w in asr:
        s = NUMBERS.get(w["w"].strip(".,?!"), w["w"])
        s = norm(s)
        if not s:
            continue
        for k, c in enumerate(s):
            achars.append(c)
            atime.append(w["start"] + (w["end"] - w["start"]) * (k + 0.5) / len(s))
    print(f"aligning {len(lchars)} lyric chars to {len(achars)} asr chars")
    pairs = align("".join(lchars), "".join(achars))

    spans = {}
    hits = {}
    for i, j in pairs:
        key = lowner[i]
        t = atime[j]
        lo, hi = spans.get(key, (t, t))
        spans[key] = (min(lo, t), max(hi, t))
        hits[key] = hits.get(key, 0) + 1

    # flatten words, interpolate missing
    flat = []
    for li, line in enumerate(lines):
        for wi, w in enumerate(line["words"]):
            n_chars = max(1, len(norm(SPOKEN.get(w.lower(), w))))
            cov = hits.get((li, wi), 0) / n_chars
            span = spans.get((li, wi)) if cov >= 0.34 else None
            flat.append({"li": li, "wi": wi, "w": w, "span": span, "cov": round(cov, 2)})
    known = [k for k, f in enumerate(flat) if f["span"]]
    for k, f in enumerate(flat):
        if f["span"]:
            continue
        prev = max([q for q in known if q < k], default=None)
        nxt = min([q for q in known if q > k], default=None)
        t0 = flat[prev]["span"][1] if prev is not None else 0.0
        t1 = flat[nxt]["span"][0] if nxt is not None else t0 + 1.0
        lo_k = prev if prev is not None else -1
        hi_k = nxt if nxt is not None else len(flat)
        frac0 = (k - lo_k) / (hi_k - lo_k)
        frac1 = (k - lo_k + 0.8) / (hi_k - lo_k)
        f["span"] = (t0 + (t1 - t0) * frac0 - 0.02, t0 + (t1 - t0) * frac1)
        f["interp"] = True

    # vocal activity to extend held notes: end of word = until vocal energy drops
    y, sr = librosa.load(vocals, sr=16000, mono=True)
    hop = 160
    rms = librosa.feature.rms(y=y, frame_length=640, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    thresh = np.percentile(db, 30) + 6

    def voiced_until(t, limit):
        i = int(t * sr / hop)
        lim = int(limit * sr / hop)
        while i < min(lim, len(db) - 1) and db[i] > thresh:
            i += 1
        return i * hop / sr

    overrides = {}
    ov_path = out.rsplit("/", 1)[0] + "/timing_overrides.json"
    try:
        overrides = {k: v for k, v in json.load(open(ov_path)).items() if not k.startswith("_")}
    except FileNotFoundError:
        pass

    out_lines = []
    for li, line in enumerate(lines):
        ws = [f for f in flat if f["li"] == li]
        ov = dict(overrides.get(str(li + 1), {}))
        words = []
        for k, f in enumerate(ws):
            s, e = f["span"]
            s -= 0.06  # char centres -> onset
            word = {"w": f["w"], "start": round(s, 3), "end": round(e + 0.08, 3), "cov": f["cov"]}
            if f["w"] in ov:
                word["start"] = ov.pop(f["w"])
                word["end"] = max(word["end"], word["start"] + 0.15)
                word["manual"] = True
            elif f.get("interp"):
                word["interp"] = True
            words.append(word)
        # word ends never overlap the next word's start
        for a_, b_ in zip(words[:-1], words[1:]):
            a_["end"] = round(min(max(a_["end"], a_["start"] + 0.1), b_["start"]), 3)
        out_lines.append({"section": line["section"], "text": line["text"], "words": words})
    for li, line in enumerate(out_lines):
        nxt = out_lines[li + 1]["words"][0]["start"] if li + 1 < len(out_lines) else 164.8
        last = line["words"][-1]
        last["end"] = round(max(last["end"], min(voiced_until(last["end"], nxt), nxt - 0.05)), 3)
        line["start"] = line["words"][0]["start"]
        line["end"] = last["end"]

    json.dump({"lines": out_lines}, open(out, "w"), indent=1)
    for li, line in enumerate(out_lines):
        ws = " ".join(
            f"{w['w']}@{w['start']:.2f}{'*' if w.get('interp') else ''}" for w in line["words"]
        )
        print(f"L{li + 1:02d} {line['start']:7.2f}-{line['end']:7.2f} [{line['section']}] {ws}")


if __name__ == "__main__":
    main()
