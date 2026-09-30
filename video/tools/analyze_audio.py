"""Audio analysis for the music video.

Produces video/data/audio_features.npz (per-video-frame energy curves) and
video/data/beats.json (tempo, beat and onset times, section novelty peaks).

Usage: python analyze_audio.py <song.wav> <out_dir> [fps]
"""
import json
import sys

import librosa
import numpy as np


def main():
    wav, out_dir = sys.argv[1], sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

    y, sr = librosa.load(wav, sr=22050, mono=True)
    duration = len(y) / sr
    hop = 256

    # --- beats -------------------------------------------------------------
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop, tightness=120
    )
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    # --- percussive onsets (kick-ish / snare-ish) -----------------------------
    y_h, y_p = librosa.effects.hpss(y)
    S = np.abs(librosa.stft(y_p, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[(freqs >= 30) & (freqs < 150)].sum(axis=0)
    mid = S[(freqs >= 1500) & (freqs < 5000)].sum(axis=0)
    kick_env = np.maximum(0, np.diff(low, prepend=low[0]))
    snare_env = np.maximum(0, np.diff(mid, prepend=mid[0]))
    kick_frames = librosa.util.peak_pick(
        kick_env / (kick_env.max() + 1e-9), pre_max=6, post_max=6, pre_avg=20,
        post_avg=20, delta=0.08, wait=12,
    )
    snare_frames = librosa.util.peak_pick(
        snare_env / (snare_env.max() + 1e-9), pre_max=6, post_max=6, pre_avg=20,
        post_avg=20, delta=0.08, wait=12,
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False
    )

    # --- per-video-frame curves --------------------------------------------------
    n_video = int(np.ceil(duration * fps))
    t_video = np.arange(n_video) / fps
    Sfull = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    t_stft = librosa.frames_to_time(np.arange(Sfull.shape[1]), sr=sr, hop_length=hop)

    def band(lo, hi):
        v = Sfull[(freqs >= lo) & (freqs < hi)].mean(axis=0)
        v = np.interp(t_video, t_stft, v)
        return v / (np.percentile(v, 98) + 1e-9)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    rms_v = np.interp(t_video, t_stft[: len(rms)], rms)
    rms_v = rms_v / (np.percentile(rms_v, 98) + 1e-9)
    onset_v = np.interp(t_video, t_stft[: len(onset_env)], onset_env)
    onset_v = onset_v / (np.percentile(onset_v, 99) + 1e-9)

    # --- structure: novelty on beat-synchronous chroma+mfcc ---------------------------
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
    feat = np.vstack([librosa.util.normalize(chroma, axis=0),
                      librosa.util.normalize(mfcc, axis=1)])
    feat_sync = librosa.util.sync(feat, beat_frames, aggregate=np.median)
    R = librosa.segment.recurrence_matrix(feat_sync, width=3, mode="affinity", sym=True)
    # checkerboard novelty
    k = 8
    g = np.outer(np.r_[-np.ones(k), np.ones(k)], np.r_[-np.ones(k), np.ones(k)])
    g *= np.outer(np.hanning(2 * k), np.hanning(2 * k))
    Rp = np.pad(R, k, mode="constant")
    nov = np.array([np.sum(Rp[i:i + 2 * k, i:i + 2 * k] * -g) for i in range(R.shape[0])])
    nov = nov - nov.min()
    nov /= nov.max() + 1e-9
    peaks = librosa.util.peak_pick(nov, pre_max=8, post_max=8, pre_avg=8, post_avg=8,
                                   delta=0.05, wait=8)
    beat_sync_times = np.r_[0, beat_times]
    section_times = [float(beat_sync_times[min(p, len(beat_sync_times) - 1)]) for p in peaks]

    # --- downbeats: the beat phase (mod 4) carrying the most kick energy, found in a
    # sliding window because the tracker occasionally inserts a beat
    S_k = np.abs(librosa.stft(y, n_fft=2048, hop_length=128))
    f_k = librosa.fft_frequencies(sr=sr, n_fft=2048)
    kick_band = S_k[(f_k >= 35) & (f_k < 110)].sum(axis=0)
    t_k = librosa.frames_to_time(np.arange(S_k.shape[1]), sr=sr, hop_length=128)
    kick_at = np.array([kick_band[max(0, i - 2): i + 4].max()
                        for i in np.searchsorted(t_k, beat_times)])
    downbeats = []
    for i in range(len(beat_times)):
        lo, hi = max(0, i - 12), min(len(beat_times), i + 12)
        idx = np.arange(lo, hi)
        phase = int(np.argmax([kick_at[idx[(idx % 4) == p]].mean() for p in range(4)]))
        if i % 4 == phase:
            downbeats.append(float(beat_times[i]))

    np.savez_compressed(
        f"{out_dir}/audio_features.npz",
        fps=fps, t=t_video,
        rms=rms_v.astype(np.float32), onset=onset_v.astype(np.float32),
        bass=band(30, 160).astype(np.float32), low_mid=band(160, 800).astype(np.float32),
        high_mid=band(800, 4000).astype(np.float32), treble=band(4000, 11000).astype(np.float32),
    )
    out = {
        "duration": duration,
        "tempo": tempo,
        "beats": [round(float(b), 4) for b in beat_times],
        "downbeats": [round(b, 4) for b in downbeats],
        "kicks": [round(float(t), 4) for t in librosa.frames_to_time(kick_frames, sr=sr, hop_length=hop)],
        "snares": [round(float(t), 4) for t in librosa.frames_to_time(snare_frames, sr=sr, hop_length=hop)],
        "onsets": [round(float(t), 4) for t in librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)],
        "section_candidates": [round(t, 3) for t in section_times],
        "novelty_beats": [round(float(v), 4) for v in nov],
    }
    with open(f"{out_dir}/beats.json", "w") as f:
        json.dump(out, f)
    print(f"duration={duration:.2f}s tempo={tempo:.2f} beats={len(beat_times)} "
          f"kicks={len(out['kicks'])} snares={len(out['snares'])}")
    print("section candidates:", out["section_candidates"])


if __name__ == "__main__":
    main()
