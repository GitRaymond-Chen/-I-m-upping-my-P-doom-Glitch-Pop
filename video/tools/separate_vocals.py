"""Isolate the vocal stem with the UVR MDX-Net Voc_FT model (via sherpa-onnx).

Usage: python separate_vocals.py <song44.wav> <uvr_model.onnx> <out_vocals.wav>
"""
import sys
import time

import numpy as np
import sherpa_onnx
import soundfile as sf


def main():
    wav, model, out = sys.argv[1:4]
    samples, sr = sf.read(wav, dtype="float32", always_2d=True)
    samples = np.ascontiguousarray(samples.T)  # (channels, n)

    config = sherpa_onnx.OfflineSourceSeparationConfig(
        model=sherpa_onnx.OfflineSourceSeparationModelConfig(
            uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=model),
            num_threads=4,
        )
    )
    sep = sherpa_onnx.OfflineSourceSeparation(config)
    t0 = time.time()
    result = sep.process(sample_rate=sr, samples=samples)
    print(f"separated in {time.time() - t0:.1f}s")
    print([m for m in dir(result) if not m.startswith("_")])
    stems = result.stems
    print("num stems:", len(stems))
    for i, stem in enumerate(stems):
        data = np.array(stem.data)
        print(i, data.shape)
    vocals = np.array(stems[0].data).T
    sf.write(out, vocals, result.sample_rate)
    if len(stems) > 1:
        sf.write(out.replace(".wav", "_other.wav"), np.array(stems[1].data).T, result.sample_rate)


if __name__ == "__main__":
    main()
