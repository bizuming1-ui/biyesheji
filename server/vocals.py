"""参考音频「人声分离」:参考语音是歌曲 / 带伴奏 / 背景音乐时,先把人声抠出来再训练。

为什么需要(来自真实运行日志):参考音频是歌曲时 Whisper 把歌词识别成乱码(「晚风吹动着竹林月光拉着了声音音火中…」),
ICL 模式相似度只有 55%,x-vector 也只有 96.6%、信噪比 18dB —— 伴奏被当成「说话人特征」一起克隆了。
分离出干净人声后:声纹更纯、Whisper 文本更准、ICL 才有机会赢。

实现:demucs(htdemucs)在子进程里跑(不污染主进程/不占 TTS 显存),指定 GPU 或 CPU。
任何一步失败 → 返回 None,训练照常使用原音频,绝不因此失败。
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import audio_utils as au
from . import config


def available() -> bool:
    if config.MOCK:
        return True
    try:
        return importlib.util.find_spec("demucs") is not None
    except Exception:  # noqa: BLE001
        return False


def should_run(snr_db: float, song: bool) -> bool:
    mode = config.REF_SEPARATE
    if mode in ("0", "off", "false"):
        return False
    if not available():
        return False
    if song or mode in ("1", "on", "true"):
        return True
    return snr_db < config.REF_SEPARATE_SNR  # auto:信噪比不够高就分离


def _mock_vocals(x: np.ndarray, sr: int) -> np.ndarray:
    """Mock:用带通滤波模拟「去伴奏」,仅用于无 GPU/无 demucs 的链路自检。"""
    from scipy.signal import butter, sosfilt

    sos = butter(4, [80, 7500], btype="band", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def separate(x: np.ndarray, sr: int):
    """返回人声(单声道 float32, 采样率 sr);失败返回 None。会做合理性校验。"""
    try:
        if config.MOCK:
            return _mock_vocals(x, sr)
        with tempfile.TemporaryDirectory(prefix="vocsep_") as td:
            src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
            au.write_wav(src, x, sr)
            env = dict(os.environ)
            dev = config.REF_SEPARATE_DEVICE
            if dev != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = config.REF_SEPARATE_GPU
            cmd = [sys.executable, "-m", "server.vocal_runner", str(src), str(dst), dev]
            r = subprocess.run(cmd, cwd=str(config.ROOT), env=env, capture_output=True, text=True, timeout=config.REF_SEPARATE_TIMEOUT)
            if r.returncode != 0 or not dst.exists():
                print("[vocals] 分离失败,使用原音频:", (r.stderr.strip().splitlines() or ["?"])[-1][:200], flush=True)
                return None
            y = au.decode_audio(dst.read_bytes(), sr)
        if y.size < sr or not np.all(np.isfinite(y)):
            return None
        n = min(x.size, y.size)
        if au.to_db(au.rms(y[:n])) < au.to_db(au.rms(x[:n])) - 30:  # 分离后几乎没声音:当作失败
            print("[vocals] 分离结果能量过低,使用原音频", flush=True)
            return None
        return y.astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print("[vocals] 异常,使用原音频:", repr(e)[:200], flush=True)
        return None
