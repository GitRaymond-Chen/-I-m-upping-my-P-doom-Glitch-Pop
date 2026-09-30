"""Transcribe the isolated vocal stem with NeMo Parakeet-TDT (via sherpa-onnx),
producing word-level timestamps.

Recognition runs on overlapping windows (WIN seconds, hop WIN/2); each window
only contributes the words that start in its central half, so every word is
decoded with context on both sides and no word is cut by a chunk boundary.

Usage: python transcribe.py <vocals.wav> <parakeet_model_dir> <out_words.json>
"""
import json
import sys

import librosa
import numpy as np
import sherpa_onnx

WIN = 20.0


def decode(rec, sr, seg, t_offset):
    stream = rec.create_stream()
    stream.accept_waveform(sr, seg)
    rec.decode_stream(stream)
    r = stream.result
    toks, ts = list(r.tokens), list(r.timestamps)
    durs = list(getattr(r, "durations", []) or [])
    words, cur = [], None
    for k, (tok, t0) in enumerate(zip(toks, ts)):
        t_abs = t_offset + float(t0)
        d = float(durs[k]) if k < len(durs) else 0.08
        if tok.startswith(" ") or cur is None:
            if cur:
                words.append(cur)
            cur = {"w": tok.strip(), "start": t_abs, "end": t_abs + d}
        else:
            cur["w"] += tok
            cur["end"] = t_abs + d
    if cur:
        words.append(cur)
    return r.text, words


def main():
    wav, model_dir, out = sys.argv[1:4]
    y, sr = librosa.load(wav, sr=16000, mono=True)
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    total = len(y) / sr

    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{model_dir}/encoder.int8.onnx",
        decoder=f"{model_dir}/decoder.int8.onnx",
        joiner=f"{model_dir}/joiner.int8.onnx",
        tokens=f"{model_dir}/tokens.txt",
        model_type="nemo_transducer",
        num_threads=4,
    )
    hop = WIN / 2
    words = []
    a = 0.0
    while a < total:
        b = min(a + WIN, total)
        keep_lo = 0.0 if a == 0 else a + WIN / 4
        keep_hi = total + 1 if b >= total else b - WIN / 4
        text, ws = decode(rec, sr, y[int(a * sr): int(b * sr)], a)
        kept = [w for w in ws if keep_lo <= w["start"] < keep_hi]
        print(f"[{a:6.1f}-{b:6.1f}] keep {keep_lo:6.1f}-{min(keep_hi, total):6.1f}: "
              + " ".join(w["w"] for w in kept))
        words.extend(kept)
        if b >= total:
            break
        a += hop
    for w in words:
        w["start"] = round(w["start"], 3)
        w["end"] = round(w["end"], 3)
    with open(out, "w") as f:
        json.dump(words, f, indent=0)
    print(len(words), "words")


if __name__ == "__main__":
    main()
