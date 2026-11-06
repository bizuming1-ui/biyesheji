"""检测日志接口自检(Mock,无需 GPU):  CLONE_MOCK=1 python tests/e2e_diag.py
验证: /api/diag 完整参数 + /api/clientlog 上报 + 训练在 Whisper/另一张卡未就绪时也能立刻开始。"""
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("TEST_PORT", "7862"))
BASE = "http://127.0.0.1:%d" % PORT
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" -> " + str(extra)[:160] if extra else ""))
    if not cond:
        fails.append(name)


env = dict(os.environ, CLONE_MOCK="1", PORT=str(PORT), CLONE_DATA=str(ROOT / "data_test_diag"), PYTHONUNBUFFERED="1")
p = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(PORT), "--no-access-log"],
                     cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    c = httpx.Client(base_url=BASE, timeout=20)
    for _ in range(60):
        try:
            if c.get("/api/health").status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    time.sleep(2.5)  # 等 worker 打印「就绪」日志
    d = c.get("/api/diag").json()
    for k in ("system", "versions", "config", "gpus", "workers", "job", "server_log", "client_log", "uptime_s"):
        check("diag has " + k, k in d)
    check("diag config has TTS_MODEL", "TTS_MODEL" in d["config"])
    check("diag does not leak secrets", "hf_" not in str(d["config"]) and "HF_TOKEN" not in str(d["config"]))
    c.post("/api/clientlog", json=[{"kind": "ui", "msg": "点击 #train"}, {"kind": "http-fail", "url": "/api/x", "ms": 12}])
    d = c.get("/api/diag").json()
    check("clientlog stored", any(e.get("msg") == "点击 #train" for e in d["client_log"]), d["client_log"][-2:])
    check("server log captured", len(d["server_log"]) > 0)
    t = np.arange(24000 * 6) / 24000
    buf = io.BytesIO(); sf.write(buf, (0.2 * np.sin(2 * np.pi * 180 * t)).astype("float32"), 24000, format="WAV")
    r = c.post("/api/voice/train", files={"file": ("ref.wav", buf.getvalue(), "audio/wav")}, data={"text": "测试文本", "lang": "zh"})
    check("train accepted", r.status_code == 200, r.text)
    for _ in range(80):
        j = c.get("/api/voice/job").json()
        if j["state"] in ("done", "error"):
            break
        time.sleep(0.5)
    check("train done", j["state"] == "done", j.get("error"))
    check("job timeline recorded", len(j.get("timeline", [])) >= 3, j.get("timeline"))
finally:
    p.terminate()
print("\n结果:", "全部通过" if not fails else "失败 %s" % fails)
sys.exit(1 if fails else 0)
