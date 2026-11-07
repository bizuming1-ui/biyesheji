"""子进程:python -m server.vocal_runner in.wav out.wav [cuda|cpu]
用 demucs htdemucs 抽取人声,输出单声道 wav。只用 soundfile 读写(不依赖 torchaudio 的 IO / torchcodec)。"""
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def main():
    src, dst = sys.argv[1], sys.argv[2]
    want = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    dev = "cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu"
    model = get_model("htdemucs")
    model.eval()
    sr_m = int(model.samplerate)
    wav, sr = sf.read(src, always_2d=True, dtype="float32")
    wav = wav.T  # (C, T)
    if sr != sr_m:
        from math import gcd

        g = gcd(sr, sr_m)
        wav = np.stack([resample_poly(ch, sr_m // g, sr // g) for ch in wav]).astype(np.float32)
    ch = int(model.audio_channels)
    if wav.shape[0] == 1 and ch == 2:
        wav = np.repeat(wav, 2, axis=0)
    x = torch.from_numpy(np.ascontiguousarray(wav))[None]
    ref = x.mean(1, keepdim=True)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(model, (x - mean) / std, device=dev, split=True, overlap=0.25, shifts=0, progress=False)[0]
    v = out[model.sources.index("vocals")] * std + mean
    mono = v.mean(0).cpu().numpy().astype(np.float32)
    sf.write(dst, mono, sr_m, subtype="FLOAT")


if __name__ == "__main__":
    main()
