"""DeepSeek 接入端到端自检(Mock TTS + 假 DeepSeek,无需 GPU / 无需真实 Key):  python tests/e2e_deepseek.py
验证: Key 校验(错/对) → 训练 → 对话(流式文本/plan 事件/音频顺序) → thinking 默认关闭 → 长文自动续写 → 429 自动重试
      → 402/401 错误必回显 → Key 不泄露到 /api/status /api/diag /api/perf。"""
import base64
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
DSPORT = int(os.environ.get("FAKE_DS_PORT", "7871"))
GOOD = "sk-test-good-key-1234567890"
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" -> " + str(extra)[:220] if extra else ""))
    if not cond:
        fails.append(name)


def make_ref():
    sr = 24000
    t = np.arange(sr * 8) / sr
    x = 0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t)) + 0.003 * np.random.randn(len(t))
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def chat(c, text, opts=None, sid="ds-test", max_s=40):
    r = c.post("/api/chat", json={"sid": sid, "text": text, "opts": dict({"breath": 0.2}, **(opts or {}))})
    tid = r.json()["tid"]
    after, evs, t0 = 0, [], time.time()
    while time.time() - t0 < max_s:
        j = c.get("/api/events", params={"sid": sid, "tid": tid, "after": after, "wait": 5}, timeout=15).json()
        evs += j["events"]
        after = j["next"]
        if j["done"]:
            break
    return evs


def main():
    env = dict(os.environ, CLONE_MOCK="1", PORT=str(PORT), CLONE_DATA="/tmp/clone_ds_data", DEEPSEEK_BASE_URL="http://127.0.0.1:%d" % DSPORT, PYTHONUNBUFFERED="1")
    fake = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_deepseek.py")], env=dict(env, FAKE_DS_PORT=str(DSPORT)))
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(PORT), "--no-access-log"], cwd=ROOT, env=env)
    try:
        c = httpx.Client(base_url="http://127.0.0.1:%d" % PORT, timeout=30)
        fk = httpx.Client(base_url="http://127.0.0.1:%d" % DSPORT, timeout=10)
        for _ in range(80):
            try:
                if c.get("/api/health").status_code == 200 and fk.get("/_debug").status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            check("servers up", False)
            return
        check("servers up", True)
        st = c.get("/api/status").json()
        check("no local LLM worker (saves ~7.5GiB VRAM)", "llm" not in st["workers"], list(st["workers"]))
        check("deepseek state: no key yet", st["deepseek"]["state"] == "nokey" and not st["deepseek"]["has_key"])
        check("page has key UI + player.js", "dsbtn" in c.get("/").text and c.get("/player.js").status_code == 200)

        r = c.post("/api/deepseek", json={"key": "sk-wrong-key-000000"}).json()
        check("bad key rejected with readable reason", r["ok"] is False and "Key" in r["error"], r)
        r = c.post("/api/deepseek", json={"key": GOOD, "model": ""}).json()
        check("good key accepted, strongest model picked", r["ok"] and r["model"] == "deepseek-v4-pro", r)
        check("key hint only (no full key)", r["key_hint"] == "…7890" and GOOD not in str(r))

        r = c.post("/api/voice/train", files={"file": ("ref.wav", make_ref(), "audio/wav")}, data={"text": "", "lang": "zh"})
        for _ in range(80):
            job = c.get("/api/voice/job").json()
            if job.get("state") in ("done", "error"):
                break
            time.sleep(0.25)
        check("train done", job.get("state") == "done", job.get("error"))

        t0 = time.time()
        evs = chat(c, "你好", {"filler": False})
        types = [e["type"] for e in evs]
        audio = [e for e in evs if e["type"] == "audio"]
        text = "".join(e["delta"] for e in evs if e["type"] == "text")
        done = evs[-1]
        check("chat done", done["type"] == "done", done)
        check("DeepSeek text streamed", "DeepSeek" in text and len(text) > 50, text[:60])
        plans = [e for e in evs if e["type"] == "plan"]
        check("plan events with final flag (drive adaptive prebuffer)", plans and plans[-1].get("final") is True and plans[-1]["est_s"] > 1, plans[-1:] )
        check("plan est_s monotonic", all(a["est_s"] <= b["est_s"] for a, b in zip(plans, plans[1:])))
        idx = [a["idx"] for a in audio]
        check("audio idx contiguous (no hole → no stall)", idx == list(range(len(idx))) and len(idx) >= 4, idx[:12])
        check("audio decodable", all(len(base64.b64decode(a["b64"])) > 100 for a in audio))
        check("both GPUs used", len(done["stats"]["gpu_chunks"]) == 2, done["stats"]["gpu_chunks"])
        check("no filler by default", not done["stats"].get("filler"))
        dbg = fk.get("/_debug").json()
        check("thinking disabled by default (fast first token)", dbg[-1]["thinking"] == {"type": "disabled"}, dbg[-1])
        check("strongest model used", dbg[-1]["model"] == "deepseek-v4-pro", dbg[-1])
        check("first audio quick", done["stats"]["ttfa_ms"] < 2500, done["stats"]["ttfa_ms"])

        n0 = len(dbg)
        evs = chat(c, "给我讲个长故事 __LONG__", {"long": True})
        text = "".join(e["delta"] for e in evs if e["type"] == "text")
        check("long answer auto-continued seamlessly", "老和尚" in text and "一代一代" in text and any(e["type"] == "cont" for e in evs), text[-40:])
        check("continuation = 2 API calls", len(fk.get("/_debug").json()) - n0 == 2, len(fk.get("/_debug").json()) - n0)
        check("long mode asks big max_tokens", fk.get("/_debug").json()[n0]["max_tokens"] == 16384)
        audio = [e for e in evs if e["type"] == "audio"]
        check("continuation audio idx contiguous", [a["idx"] for a in audio] == list(range(len(audio))))

        evs = chat(c, "你好 __429_ONCE__")
        check("429 retried automatically before first token", evs[-1]["type"] == "done" and any(e["type"] == "text" for e in evs), evs[-1])

        evs = chat(c, "你好 __THINK__", {"ds": {"key": GOOD, "model": "deepseek-flash", "thinking": True}})
        check("thinking mode: think event + model override", any(e["type"] == "think" for e in evs) and fk.get("/_debug").json()[-1]["model"] == "deepseek-flash" and fk.get("/_debug").json()[-1]["thinking"]["type"] == "enabled")
        check("thinking text not spoken/leaked", "思考中" not in "".join(e.get("delta", "") for e in evs if e["type"] == "text"))

        evs = chat(c, "你好 __402__")
        check("402 error shown to user", evs[-1]["type"] == "error" and "余额" in evs[-1]["text"], evs[-1])
        evs = chat(c, "你好", {"ds": {"key": "sk-bad-bad-bad-bad"}})
        check("401 error shown to user", evs[-1]["type"] == "error" and "Key" in evs[-1]["text"], evs[-1])
        evs = chat(c, "你好", {"ds": {"key": GOOD}})
        check("recovers after errors", evs[-1]["type"] == "done", evs[-1])

        for path in ("/api/status", "/api/diag", "/api/perf", "/api/deepseek"):
            check("key not leaked in " + path, GOOD not in c.get(path).text)
    finally:
        srv.terminate(); fake.terminate()
        try:
            srv.wait(5); fake.wait(5)
        except Exception:
            srv.kill(); fake.kill()
    print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
