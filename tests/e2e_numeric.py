"""训练根因修复自检(Mock,无需 GPU):  CLONE_MOCK=1 python tests/e2e_numeric.py
场景: TTS 在 float16/bfloat16 下采样出 NaN(真实环境里是 CUDA 设备端断言 "probability tensor contains inf/nan")。
验证: 引擎自动重启 TTS 进程 → 降级精度(float16→bfloat16→float32) → 训练自动重试并成功,用户不需要做任何事;
      降级之后对话/合成仍然可用;全部精度都异常时给出清晰的错误而不是卡死。"""
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
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" -> " + str(extra)[:200] if extra else ""))
    if not cond:
        fails.append(name)


def ref():
    sr = 24000
    t = np.arange(sr * 8) / sr
    x = 0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t)) + 0.003 * np.random.randn(len(t))
    b = io.BytesIO()
    sf.write(b, x.astype(np.float32), sr, format="WAV")
    return b.getvalue()


def scenario(name, port, start_dtype, bad_dtypes, expect_state, expect_dtype=None):
    print("\n=== 场景: %s ===" % name)
    env = dict(os.environ, CLONE_MOCK="1", LLM_BACKEND="local", PORT=str(port), TTS_DTYPE=start_dtype, MOCK_NUMERIC_DTYPE=bad_dtypes,
               CLONE_DATA="/tmp/clone_numeric_%d" % port, PYTHONUNBUFFERED="1")
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(port), "--no-access-log"],
                         cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        c = httpx.Client(base_url="http://127.0.0.1:%d" % port, timeout=30)
        for _ in range(80):
            try:
                if c.get("/api/health").status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        for _ in range(60):
            st = c.get("/api/status").json()["workers"]
            if all(v["state"] == "ready" for v in st.values()):
                break
            time.sleep(0.4)
        t0 = time.time()
        r = c.post("/api/voice/train", files={"file": ("ref.wav", ref(), "audio/wav")}, data={"text": ""})
        check("train accepted", r.status_code == 200, r.text)
        job = {}
        for _ in range(240):
            job = c.get("/api/voice/job").json()
            if job.get("state") in ("done", "error"):
                break
            time.sleep(0.5)
        print("   用时 %.1fs, 状态 %s, 日志尾部:" % (time.time() - t0, job.get("state")))
        for l in (job.get("log") or [])[-8:]:
            print("     " + l)
        check("final state == %s" % expect_state, job.get("state") == expect_state, job.get("error") or job.get("stage"))
        if expect_state == "done":
            st = c.get("/api/status").json()
            dts = {k: v["info"].get("dtype") for k, v in st["workers"].items() if k.startswith("tts")}
            check("tts dtype is %s" % expect_dtype, all(d == expect_dtype for d in dts.values()), dts)
            check("voice trained", bool(st.get("voice")))
            check("log mentions auto retry", any("数值异常" in l for l in job.get("log", [])) or start_dtype == expect_dtype)
            # 降级后还能正常对话合成
            sid = "numeric-%d" % port
            rr = c.post("/api/chat", json={"sid": sid, "text": "你好", "speak": True, "opts": {}})
            tid = rr.json()["tid"]
            after, audio, done = 0, 0, False
            t1 = time.time()
            while time.time() - t1 < 40 and not done:
                j = c.get("/api/events", params={"sid": sid, "tid": tid, "after": after, "wait": 5}, timeout=15).json()
                audio += sum(1 for e in j["events"] if e["type"] == "audio")
                after, done = j["next"], j["done"]
            check("chat works after downgrade (got audio)", audio > 0, audio)
        else:
            check("error message is explicit", bool(job.get("error")), job.get("error"))
        return job
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()


scenario("正常(无数值异常)", 7871, "bfloat16", "", "done", "bfloat16")
scenario("bfloat16 数值异常 → 自动升 float32", 7872, "bfloat16", "bfloat16", "done", "float32")
scenario("float16 + bfloat16 都异常 → 连降两级", 7873, "float16", "float16,bfloat16", "done", "float32")
scenario("所有精度都异常 → 明确报错不卡死", 7874, "bfloat16", "bfloat16,float32", "error")
print("\n结果: " + ("全部通过" if not fails else "失败 %d 项: %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)
