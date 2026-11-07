"""v9 自检(Mock,无需 GPU):  CLONE_MOCK=1 python tests/e2e_localllm.py
验证: 1) DeepSeek 已彻底移除(/api/deepseek 404、status 里没有 deepseek 字段、没有 Key 相关界面)
      2) 对话走本地 LLM worker:/api/ping 带 llm 状态;文字流 + 分句 TTS + 每句一个 plan 事件(播放器据此零停顿)
      3) 长文本自动续写:一轮 token 用完后续写,文字流不中断,前端收到 cont 事件,续写标记不会进入文字
      4) LLM 故障必回显、之后仍可正常对话
      5) 令牌桶限速 Pacer:首批不限速,之后严格 ≤ tps"""
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
sys.path.insert(0, str(ROOT))
PORT = int(os.environ.get("TEST_PORT", "7864"))
BASE = "http://127.0.0.1:%d" % PORT
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" -> " + str(extra)[:200] if extra else ""))
    if not cond:
        fails.append(name)


def make_ref():
    sr = 24000
    t = np.arange(sr * 8) / sr
    x = 0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t)) + 0.003 * np.random.randn(len(t))
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def chat(c, text, sid="t9", max_s=60):
    tid = c.post("/api/chat", json={"sid": sid, "text": text, "opts": {"breath": 0.4}}).json()["tid"]
    after, evs, t0 = 0, [], time.time()
    while time.time() - t0 < max_s:
        j = c.get("/api/events", params={"sid": sid, "tid": tid, "after": after, "wait": 5}, timeout=15).json()
        evs += j["events"]
        after = j["next"]
        if j["done"]:
            break
    return evs


# ---- 5) Pacer 单元测试(不需要服务)
from server.workers import Pacer  # noqa: E402

pc = Pacer(50.0, 10)
t0 = time.time()
for _ in range(10):
    pc.tick()
burst_s = time.time() - t0
t1 = time.time()
for _ in range(50):
    pc.tick()
paced_s = time.time() - t1
check("Pacer: burst tokens are not throttled", burst_s < 0.05, "%.3fs" % burst_s)
check("Pacer: afterwards <= tps (50 tokens @50tps >= 0.9s)", 0.9 <= paced_s < 1.6, "%.2fs" % paced_s)
check("Pacer: tps=0 disables", (lambda p: [p.tick() for _ in range(500)] and True)(Pacer(0, 0)))

env = dict(os.environ, CLONE_MOCK="1", PORT=str(PORT), CLONE_DATA="/tmp/clone_localllm_data", PYTHONUNBUFFERED="1")
p = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(PORT), "--no-access-log"],
                     cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    c = httpx.Client(base_url=BASE, timeout=30)
    for _ in range(80):
        try:
            if c.get("/api/health").status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    time.sleep(2.5)
    check("/api/deepseek is gone", c.get("/api/deepseek").status_code in (404, 405))
    st = c.get("/api/status").json()
    check("status has no deepseek field", "deepseek" not in st)
    check("status lists local llm worker", "llm" in st["workers"], list(st["workers"]))
    pg = c.get("/api/ping").json()
    check("ping reports llm state", pg.get("llm") in ("ready", "loading"), pg)
    page = c.get("/").text
    check("page has no DeepSeek/Key UI", "deepseek" not in page.lower() and "dskey" not in page)

    r = c.post("/api/voice/train", files={"file": ("ref.wav", make_ref(), "audio/wav")}, data={"text": "", "lang": "zh"})
    check("train accepted", r.status_code == 200, r.text)
    for _ in range(120):
        j = c.get("/api/voice/job").json()
        if j.get("state") in ("done", "error"):
            break
        time.sleep(0.5)
    check("training done", j.get("state") == "done", j.get("error"))

    evs = chat(c, "请介绍一下你自己")
    kinds = [e["type"] for e in evs]
    n_plan, n_audio = kinds.count("plan"), kinds.count("audio")
    check("short chat: text+audio+done", "text" in kinds and n_audio > 3 and kinds[-1] == "done", kinds[-3:])
    check("plan event per sentence (>=3, last is final)", n_plan >= 3 and any(e.get("final") for e in evs if e["type"] == "plan"), n_plan)
    first_plan = kinds.index("plan")
    check("first plan arrives before the last text", first_plan < len(kinds) - 1 - kinds[::-1].index("text"))

    evs = chat(c, "请讲个长故事 __LLM_LONG__")
    txt = "".join(e["delta"] for e in evs if e["type"] == "text")
    conts = [e for e in evs if e["type"] == "cont"]
    check("long chat: cont events 2 and 3", [e["round"] for e in conts] == [2, 3], conts)
    check("long chat: continuation text present, marker never leaks", "第2段续写" in txt and "第3段续写" in txt and "\x00" not in txt and "CONT" not in txt, txt[-80:])
    check("long chat: finishes with done", evs[-1]["type"] == "done")

    evs = chat(c, "__LLM_FAIL__")
    check("LLM failure is reported", evs[-1]["type"] == "error", evs[-1])
    evs = chat(c, "你好")
    check("chat works after failure", evs[-1]["type"] == "done", evs[-1])
finally:
    p.terminate()
print("\n结果:", "全部通过" if not fails else "失败 %s" % fails)
sys.exit(1 if fails else 0)
