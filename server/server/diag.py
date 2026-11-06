"""检测日志后端:收集「完整参数」,供浏览器模拟器右上角的检测日志面板读取/导出/上传 GitHub。

提供:
  * 服务端日志环形缓冲(把 stdout/stderr 同时复制一份进缓冲,Kaggle 输出框照常显示)
  * 浏览器页面上报的事件缓冲(/api/clientlog)
  * collect(engine): 版本 / GPU / 进程 / 配置 / 训练任务 / 声音 / 资源占用 的完整快照
所有函数都必须「绝不抛异常」:检测系统自己不能成为故障源。
"""
import collections
import importlib.metadata as md
import os
import platform
import shutil
import sys
import threading
import time

from . import config

START = time.time()
_LOG = collections.deque(maxlen=600)
_CLIENT = collections.deque(maxlen=400)
_LOCK = threading.Lock()
_INSTALLED = False


class _Tee:
    """把写到 stdout/stderr 的内容复制进环形缓冲。"""

    def __init__(self, stream, tag):
        self._s, self._tag, self._buf = stream, tag, ""

    def write(self, data):
        try:
            self._s.write(data)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._buf += data
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    with _LOCK:
                        _LOG.append({"t": round(time.time(), 3), "src": self._tag, "msg": line[-400:]})
        except Exception:  # noqa: BLE001
            self._buf = ""
        return len(data)

    def flush(self):
        try:
            self._s.flush()
        except Exception:  # noqa: BLE001
            pass

    def __getattr__(self, k):
        return getattr(self._s, k)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    sys.stdout = _Tee(sys.stdout, "out")
    sys.stderr = _Tee(sys.stderr, "err")


def add_client(entry: dict):
    try:
        e = {"t": round(time.time(), 3)}
        for k in ("kind", "msg", "url", "status", "ms", "method", "sid", "extra"):
            if k in entry:
                v = entry[k]
                e[k] = v if isinstance(v, (int, float, bool)) or v is None else str(v)[:400]
        with _LOCK:
            _CLIENT.append(e)
    except Exception:  # noqa: BLE001
        pass


def server_log(n: int = 200):
    with _LOCK:
        return list(_LOG)[-n:]


def client_log(n: int = 200):
    with _LOCK:
        return list(_CLIENT)[-n:]


def _ver(name):
    try:
        return md.version(name)
    except Exception:  # noqa: BLE001
        return None


def _mem():
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS", "VmHWM", "Threads")):
                    k, v = line.split(":", 1)
                    out[k] = v.strip()
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal", "MemAvailable")):
                    k, v = line.split(":", 1)
                    out[k] = v.strip()
    except Exception:  # noqa: BLE001
        pass
    return out


def _disk():
    try:
        u = shutil.disk_usage(config.DATA_DIR)
        return {"total_gb": round(u.total / 2**30, 1), "free_gb": round(u.free / 2**30, 1)}
    except Exception:  # noqa: BLE001
        return {}


def collect(engine) -> dict:
    """完整参数快照。"""
    try:
        status = engine.status()
    except Exception as e:  # noqa: BLE001
        status = {"error": repr(e)}
    cfg = {
        k: (str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v)
        for k, v in vars(config).items()
        if k.isupper() and k not in ("SYSTEM_PROMPT", "FILLER_HINT")
    }
    job = dict(status.get("job") or {})
    t0 = job.get("started")
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uptime_s": round(time.time() - START, 1),
        "system": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "kaggle": os.path.exists("/kaggle"),
            "ffmpeg": shutil.which("ffmpeg"),
            "pid": os.getpid(),
            "mem": _mem(),
            "disk": _disk(),
            "public_url": os.environ.get("CLONE_PUBLIC_URL"),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "versions": {
            n: _ver(n)
            for n in (
                "torch", "transformers", "faster-qwen3-tts", "qwen-tts-hf", "qwen-tts", "accelerate",
                "huggingface_hub", "fastapi", "uvicorn", "numpy", "scipy", "soundfile", "noisereduce",
            )
        },
        "config": cfg,
        "gpus": status.get("gpus"),
        "workers": status.get("workers"),
        "voice": status.get("voice"),
        "job": job,
        "job_age_s": round(time.time() - t0, 1) if t0 else None,
        "turn": status.get("turn"),
        "fillers": status.get("fillers"),
        "models": status.get("models"),
        "mock": status.get("mock"),
        "server_log": server_log(120),
        "client_log": client_log(120),
    }
