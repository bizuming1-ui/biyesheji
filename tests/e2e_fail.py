"""故障场景自检(Mock,无需 GPU):一张卡的 TTS 加载失败时,系统仍然能训练/对话,并且能看到真实原因、能重新加载。
用法:  python tests/e2e_fail.py"""
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


def ref():
    sr = 24000
    t = np.arange(sr * 8) / sr
    x = 0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t)) + 0.003 * np.random.randn(len(t))
    b = io.BytesIO()
    sf.write(b, x.astype(np.float32), sr, format="WAV")
    return b.getvalue()


def main():
    env = dict(os.environ, CLONE_MOCK="1", PORT=str(PORT), MOCK_FAIL_TTS="1", TTS_LOAD_RETRIES="1", CLONE_DATA="/tmp/clone_fail_data")
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(PORT)], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        c = httpx.Client(base_url=BASE, timeout=30)
        for _ in range(60):
            try:
                if c.get("/api/health").status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        st = {}
        for _ in range(60):
            st = c.get("/api/status").json()["workers"]
            if st["tts1"]["state"] == "error" and st["tts0"]["state"] == "ready":
                break
            time.sleep(0.5)
        check("tts1 failed, tts0 ready", st["tts1"]["state"] == "error" and st["tts0"]["state"] == "ready", {k: v["state"] for k, v in st.items()})
        check("real error cause exposed", "模拟的 TTS 加载失败" in (st["tts1"]["error"] or ""))
        r = c.post("/api/voice/train", files={"file": ("ref.wav", ref(), "audio/wav")}, data={"text": ""})
        check("train accepted", r.status_code == 200, r.text)
        job = {}
        for _ in range(120):
            job = c.get("/api/voice/job").json()
            if job.get("state") in ("done", "error"):
                break
            time.sleep(0.5)
        check("train done with one card", job.get("state") == "done", job.get("error") or job.get("stage"))
        v = c.get("/api/status").json()["voice"]
        check("voice uses tts0 only", v and v.get("workers") == ["tts0"], v and v.get("workers"))
        r = c.post("/api/reload")
        check("reload endpoint", r.status_code == 200 and r.json().get("reloading") == ["tts1"], r.text)
        check("favicon no 404", c.get("/favicon.ico").status_code == 204)
    finally:
        p.terminate()
    print("\n结果:", "全部通过" if not fails else "失败 %s" % fails)
    sys.exit(1 if fails else 0)


main()
