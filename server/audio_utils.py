"""音频工具:解码、参考音频预处理、输出后处理(去沙沙/无缝衔接/微弱呼吸)。"""
import io
import shutil
import subprocess

import numpy as np

from . import config

SR = config.SR


def to_db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-9))


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def decode_audio(data: bytes, sr: int = SR) -> np.ndarray:
    """任意格式(webm/ogg/mp3/wav...)→ 单声道 float32。优先 ffmpeg,失败回退 soundfile。"""
    if shutil.which("ffmpeg"):
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"],
            input=data, capture_output=True,
        )
        if p.returncode == 0 and p.stdout:
            return np.frombuffer(p.stdout, dtype=np.float32).copy()
    import soundfile as sf

    try:
        x, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError("无法解码音频(服务器缺少 ffmpeg 且格式不被 soundfile 支持): %s" % e)
    return resample(x.mean(axis=1), file_sr, sr)


def highpass(x: np.ndarray, sr: int, cutoff: float = 60.0) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def _frame_db(x: np.ndarray, sr: int, ms: int = 20) -> np.ndarray:
    n = max(1, int(sr * ms / 1000))
    k = len(x) // n
    if k == 0:
        return np.array([to_db(rms(x))])
    fr = x[: k * n].reshape(k, n)
    return 20 * np.log10(np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1)) + 1e-9)


def trim_silence(x: np.ndarray, sr: int, margin_db: float = 38.0, keep_ms: int = 60, tail_ms: int = 120) -> np.ndarray:
    if x.size == 0:
        return x
    db = _frame_db(x, sr)
    thr = db.max() - margin_db
    idx = np.where(db > thr)[0]
    if idx.size == 0:
        return x
    n = int(sr * 0.02)
    s = max(0, idx[0] * n - int(sr * keep_ms / 1000))
    e = min(len(x), (idx[-1] + 1) * n + int(sr * tail_ms / 1000))
    return x[s:e]


def analyze_quality(x: np.ndarray, sr: int) -> dict:
    db = _frame_db(x, sr)
    noise = np.percentile(db, 10)
    sig = np.percentile(db, 90)
    return {
        "duration": round(len(x) / sr, 2),
        "snr_db": round(float(sig - noise), 1),
        "rms_db": round(to_db(rms(x)), 1),
        "clip_ratio": round(float(np.mean(np.abs(x) > 0.99)), 4),
    }


def preprocess_reference(x: np.ndarray, sr: int = SR, max_seconds: float = 20.0):
    """参考音频清洗。返回 (处理后音频, 质量报告, 警告列表)。"""
    warnings = []
    x = x.astype(np.float32)
    x = x - float(np.mean(x))
    x = highpass(x, sr, 60.0)
    q0 = analyze_quality(x, sr)
    if q0["snr_db"] < 28:
        try:
            import noisereduce as nr

            strength = 0.85 if q0["snr_db"] < 18 else 0.6
            x = nr.reduce_noise(y=x, sr=sr, stationary=True, prop_decrease=strength).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            warnings.append("降噪不可用: %s" % e)
    x = trim_silence(x, sr, margin_db=40.0, keep_ms=150, tail_ms=250)
    max_n = int(max_seconds * sr)
    if len(x) > max_n:
        # 在 max_seconds 附近找能量最低点截断,避免断在字中间
        lo, hi = int((max_seconds - 2.5) * sr), max_n
        n = int(sr * 0.02)
        seg = x[lo:hi]
        k = len(seg) // n
        fr = np.sqrt(np.mean(seg[: k * n].reshape(k, n) ** 2, axis=1))
        cut = lo + int(np.argmin(fr)) * n
        x = x[:cut]
    peak = float(np.max(np.abs(x)) + 1e-9)
    x = x * (0.9 / peak)
    q = analyze_quality(x, sr)
    if q["duration"] < 3:
        warnings.append("参考音频过短(%.1fs),建议 8~20 秒" % q["duration"])
    elif q["duration"] < 8:
        warnings.append("参考音频偏短(%.1fs),8~20 秒效果最佳" % q["duration"])
    if q0["snr_db"] < 18:
        warnings.append("环境噪声较大(信噪比约 %.0fdB),建议在安静环境重录" % q0["snr_db"])
    if q0["clip_ratio"] > 0.002:
        warnings.append("录音有爆音/削波,请降低音量或远离麦克风")
    q["snr_db_raw"] = q0["snr_db"]
    return x, q, warnings


def make_breath(sr: int, level: float, rng: np.random.Generator) -> np.ndarray:
    """极弱的吸气声(带通噪声 + 起伏包络)。level 0~1,0 表示关闭。"""
    if level <= 0.01:
        return np.zeros(0, dtype=np.float32)
    from scipy.signal import butter, sosfilt

    n = int(sr * rng.uniform(0.22, 0.32))
    noise = rng.standard_normal(n)
    sos = butter(2, [300, 3200], btype="bandpass", fs=sr, output="sos")
    b = sosfilt(sos, noise)
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * t) ** 2 * (0.55 + 0.45 * t)
    b = b * env
    b = b / (np.max(np.abs(b)) + 1e-9)
    amp = 10 ** ((-52.0 + 14.0 * min(level, 1.0)) / 20.0)
    return (b * amp).astype(np.float32)


def postprocess_chunk(
    wav: np.ndarray,
    sr: int,
    target_rms_db: float | None,
    first: bool,
    breath_level: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """对每个 TTS 片段做:去直流 → 60Hz 高通 → 修剪首尾静音 → 统一响度 → 淡入淡出 → 软限幅。"""
    rng = rng or np.random.default_rng()
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    if x.size < int(sr * 0.05):
        return x
    x = x - float(np.mean(x))
    x = highpass(x, sr, 60.0)
    x = trim_silence(x, sr, margin_db=42.0, keep_ms=30 if first else 70, tail_ms=90)
    if target_rms_db is not None:
        cur = to_db(rms(x))
        gain = float(np.clip(10 ** ((target_rms_db - cur) / 20.0), 0.4, 2.5))
        x = x * gain
    fade = int(sr * 0.008)
    if x.size > fade * 2:
        x[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        x[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    x = np.tanh(x * 1.1) / np.tanh(1.1)  # 软限幅,避免削波爆音
    if not first and breath_level > 0.01:
        br = make_breath(sr, breath_level, rng)
        if br.size:
            x = np.concatenate([br, np.zeros(int(sr * 0.04), dtype=np.float32), x])
    return x.astype(np.float32)


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def write_wav(path, x: np.ndarray, sr: int = SR) -> None:
    import soundfile as sf

    sf.write(str(path), x, sr, subtype="PCM_16")
