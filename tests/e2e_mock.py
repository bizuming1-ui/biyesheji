"""端到端自检(Mock 模式,无需 GPU):  python tests/e2e_mock.py
验证: 启动 → 训练(双卡) → 流式对话 → 音频顺序/无缺口 → 停止 → ASR → 错误必回显 → 前端页面(无摄像头)。"""
import asyncio
import io
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("TEST_PORT", "7861"))
BASE = "http://127.0.0.1:%d" % PORT
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" -> " + str(extra) if extra else ""))
    if not cond:
        fails.append(name)


def make_ref():
    sr = 24000
    t = np.arange(sr * 8) / sr
    x = 0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    x += 0.003 * np.random.randn(len(t))
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


async def chat(opts_text, ws_url, kind="chat", extra=None):
    frames, texts, done, t0 = [], [], None, time.time()
    first = None
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": kind, "text": opts_text, "opts": dict({"breath": 0.4}, **(extra or {}))}))
        while True:
            m = await asyncio.wait_for(ws.recv(), 30)
            if isinstance(m, bytes):
                turn, idx, sr, _ = struct.unpack("<IIII", m[:16])
                if first is None:
                    first = time.time() - t0
                frames.append((turn, idx, sr, len(m) - 16))
            else:
                j = json.loads(m)
                if j["type"] == "text":
                    texts.append(j["delta"])
                if j["type"] == "done":
                    done = j
                    break
                if j["type"] == "error":
                    done = j
                    break
    return frames, "".join(texts), done, first


async def stop_test(ws_url):
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "chat", "text": "请讲一个很长很长的故事"}))
        got = 0
        t0 = time.time()
        while time.time() - t0 < 1.2:
            try:
                m = await asyncio.wait_for(ws.recv(), 0.5)
                got += 1
            except asyncio.TimeoutError:
                pass
        await ws.send(json.dumps({"type": "stop"}))
        stopped = False
        t1 = time.time()
        while time.time() - t1 < 3:
            try:
                m = await asyncio.wait_for(ws.recv(), 0.5)
                if isinstance(m, str) and json.loads(m)["type"] == "stopped":
                    stopped = True
                    break
            except asyncio.TimeoutError:
                pass
        return got, stopped


def test_chunker():
    sys.path.insert(0, str(ROOT))
    from server.text_chunker import SentenceChunker, clean_for_tts, detect_language

    c = SentenceChunker()
    out = []
    for ch in "好的,我来帮你。今天天气很好,适合出去走走,你觉得呢?最后一句没有标点":
        out += c.feed(ch)
    out += c.flush()
    check("chunker first piece short", len(out[0]) <= 14 and len(out) >= 3, out)
    check("chunker lossless", "".join(out) == "好的,我来帮你。今天天气很好,适合出去走走,你觉得呢?最后一句没有标点", out)
    check("clean markdown/emoji", clean_for_tts("**你好** 😀 [链接](http://a.b) `x`") == "你好 链接 x", clean_for_tts("**你好** 😀 [链接](http://a.b) `x`"))
    check("detect lang", detect_language("你好世界") == "Chinese" and detect_language("hello world") == "English")


def main():
    test_chunker()
    env = dict(os.environ, CLONE_MOCK="1", CLONE_DATA=str(ROOT / "data_test"))
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(PORT)], cwd=ROOT, env=env)
    try:
        c = httpx.Client(base_url=BASE, timeout=30)
        for _ in range(60):
            try:
                if c.get("/api/health").status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            check("server up", False)
            return
        check("server up", True)
        page = c.get("/")
        check("index.html served", page.status_code == 200 and "Voice Clone Studio" in page.text)
        check("iframe allowed", "frame-ancestors *" in page.headers.get("content-security-policy", ""))
        time.sleep(2)
        st = c.get("/api/status").json()
        check("2 tts workers + llm + aux ready", all(w["state"] == "ready" for w in st["workers"].values()) and len([k for k in st["workers"] if k.startswith("tts")]) == 2, {k: v["state"] for k, v in st["workers"].items()})

        r = c.post("/api/voice/train", files={"file": ("ref.wav", make_ref(), "audio/wav")}, data={"text": ""})
        check("train accepted", r.status_code == 200, r.text)
        job = {}
        for _ in range(60):
            job = c.get("/api/voice/job").json()
            if job.get("state") in ("done", "error"):
                break
            time.sleep(0.5)
        check("train done", job.get("state") == "done", job.get("error") or job.get("log", [])[-2:])
        check("both gpus participated", all(g["progress"] == 100 for g in job.get("gpus", [])) and len(job.get("gpus", [])) == 2, job.get("gpus"))
        check("similarity reported", job.get("similarity") is not None, job.get("similarity"))
        check("ref.wav downloadable", c.get("/api/voice/ref.wav").status_code == 200)

        ws_url = "ws://127.0.0.1:%d/ws" % PORT
        frames, text, done, first = asyncio.run(chat("你好,请介绍一下你自己", ws_url))
        check("chat done", done and done["type"] == "done", done)
        idxs = [f[1] for f in frames]
        check("audio idx sequential", idxs == list(range(len(idxs))) and len(idxs) >= 3, idxs)
        check("both GPUs used in chat", len(done["stats"]["gpu_chunks"]) == 2, done["stats"]["gpu_chunks"])
        check("first audio < 3s", first is not None and first < 3, first)
        check("text streamed", len(text) > 40, len(text))
        check("filler played first (0-latency voice)", done["stats"].get("filler") and done["stats"]["ttfa_ms"] < done["stats"]["body_ttfa_ms"], done["stats"])
        frames3, text3, done3, _ = asyncio.run(chat("你好", ws_url, extra={"filler": False}))
        check("filler can be disabled", done3["stats"].get("filler") is None and len(frames3) >= 1, done3["stats"])
        check("chunk count matches", done["stats"]["chunks"] == len(frames), (done["stats"]["chunks"], len(frames)))

        frames2, text2, done2, _ = asyncio.run(chat("这是一句直接朗读的话。第二句也要读出来。", ws_url, "speak"))
        check("speak-only works", done2["type"] == "done" and len(frames2) >= 1, done2)

        got, stopped = asyncio.run(stop_test(ws_url))
        check("stop works", stopped and got > 0, (got, stopped))

        a = c.post("/api/asr", content=make_ref(), headers={"content-type": "application/octet-stream"})
        check("asr works", a.status_code == 200 and a.json().get("text"), a.text[:100])
        check("camera/detect removed", c.post("/api/detect", content=b"x").status_code in (404, 405) and "getUserMedia({ video" not in page.text and "摄像头" not in page.text)
        # 关键回归: LLM 出错时界面必须收到明确的 error,而不是永远转圈
        fr4, tx4, done4, _ = asyncio.run(chat("__LLM_FAIL__ 你好", ws_url))
        check("LLM failure is reported (never silent)", done4 and done4["type"] == "error" and "LLM" in done4.get("text", ""), done4)
        fr5, tx5, done5, _ = asyncio.run(chat("故障之后还能正常聊天", ws_url))
        check("chat works after a failed turn", done5 and done5["type"] == "done", done5)
    finally:
        p.terminate()
        try:
            p.wait(5)
        except Exception:
            p.kill()
    print("\n结果:", "全部通过" if not fails else "失败 %s" % fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
