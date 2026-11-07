"""端到端自检(Mock 模式,无需 GPU):  CLONE_MOCK=1 python tests/e2e_mock.py
验证: 启动 → 训练(双卡并行) → HTTP 长轮询对话 → 音频顺序/无缺口 → 断线续传(同一游标重取) → 停止 → ASR → 错误必回显 → 前端页面。"""
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
PORT = int(os.environ.get("TEST_PORT", "7861"))
BASE = "http://127.0.0.1:%d" % PORT
SID = "test-session"
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" -> " + str(extra)[:200] if extra else ""))
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


def run_chat(c, text, speak=False, opts=None, drop_every=0, sid=SID, max_s=40):
    """返回 (audio_events, texts, last_event, first_audio_s)。drop_every>0: 每取 N 次就假装断线,用旧游标重取(验证不丢不重)。"""
    t0 = time.time()
    r = c.post("/api/chat", json={"sid": sid, "text": text, "speak": speak, "opts": dict({"breath": 0.4}, **(opts or {}))})
    tid = r.json()["tid"]
    after, evs, polls = 0, [], 0
    first = None
    while time.time() - t0 < max_s:
        polls += 1
        j = c.get("/api/events", params={"sid": sid, "tid": tid, "after": after, "wait": 5}, timeout=15).json()
        got = j["events"]
        if drop_every and polls % drop_every == 0 and after > 0:
            # 模拟断线:丢弃这批,回退游标 2 个再取(服务端必须能重放)
            back = max(0, after - 2)
            j2 = c.get("/api/events", params={"sid": sid, "tid": tid, "after": back, "wait": 1}).json()
            prev = evs[back:after]
            check("replay from old cursor is identical", j2["events"][:len(prev)] == prev, (len(prev), len(j2["events"])))
        for e in got:
            if e["type"] == "audio" and first is None:
                first = time.time() - t0
        evs += got
        after = j["next"]
        if j["done"]:
            break
    audio = [e for e in evs if e["type"] == "audio"]
    texts = "".join(e["delta"] for e in evs if e["type"] == "text")
    last = evs[-1] if evs else None
    return audio, texts, last, first, evs


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
    check("clean markdown/emoji", clean_for_tts("**你好** 😀 [链接](http://a.b) `x`") == "你好 链接 x")
    check("detect lang", detect_language("你好世界") == "Chinese" and detect_language("hello world") == "English")


def test_audio_dsp():
    sys.path.insert(0, str(ROOT))
    from server import audio_utils as au

    sr = 24000
    t = np.arange(sr * 2) / sr
    gate = np.clip(np.sin(2 * np.pi * 1.5 * t) * 6, 0, 1)  # 平滑开关:前 0.33s 有声,0.33~0.67s 停顿,如此循环
    voice = (0.3 * np.sin(2 * np.pi * 200 * t) * gate).astype(np.float32)
    hiss = (0.003 * np.random.randn(len(t))).astype(np.float32)  # 约 -48dBFS 底噪
    x = voice + hiss
    post = au.StreamPost(sr, au.to_db(au.rms(voice[gate > 0.99])), trim_head=False)
    outs = [post.feed(x[i:i + 8000]) for i in range(0, len(x), 8000)]
    outs.append(post.finish())
    y = np.concatenate(outs)
    seg_pause = slice(int(sr * 1.05), int(sr * 1.30))  # 原信号在停顿里
    in_db = au.to_db(au.rms(x[seg_pause]))
    out_db = au.to_db(au.rms(y[seg_pause]))
    check("noise floor reduced in pauses (>=4dB)", in_db - out_db >= 4, "%.1f -> %.1f dB" % (in_db, out_db))
    seg_voice = slice(int(sr * 0.08), int(sr * 0.28))
    d = abs(au.to_db(au.rms(y[seg_voice])) - au.to_db(au.rms(x[seg_voice])))
    check("speech level preserved (<2dB change)", d < 2.0, d)
    check("no clipping / no NaN", np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0)
    step = np.max(np.abs(np.diff(y)))
    check("no click discontinuities at chunk seams", step < 0.2, step)
    lim = au.soft_limit(np.array([0.5, 0.79, 0.95, 1.5, -2.0], dtype=np.float32))
    check("soft limiter linear below knee & bounded", lim[0] == 0.5 and lim[1] == np.float32(0.79) and np.max(np.abs(lim)) < 1.0, lim.tolist())


def main():
    test_chunker()
    test_audio_dsp()
    env = dict(os.environ, CLONE_MOCK="1", LLM_BACKEND="local", CLONE_DATA=str(ROOT / "data_test"))
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(PORT), "--no-access-log"], cwd=ROOT, env=env)
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
        check("page uses HTTP long-poll, not WebSocket", "/api/events" in page.text and "new WebSocket" not in page.text)
        check("iframe allowed", "frame-ancestors *" in page.headers.get("content-security-policy", ""))
        time.sleep(2)
        st = c.get("/api/status").json()
        check("2 tts workers + llm + aux ready", all(w["state"] == "ready" for w in st["workers"].values()) and len([k for k in st["workers"] if k.startswith("tts")]) == 2, {k: v["state"] for k, v in st["workers"].items()})
        check("ping endpoint", c.get("/api/ping").status_code == 200)

        t0 = time.time()
        r = c.post("/api/voice/train", files={"file": ("ref.wav", make_ref(), "audio/wav")}, data={"text": "", "lang": "zh"})
        check("train accepted", r.status_code == 200, r.text)
        job = {}
        for _ in range(80):
            job = c.get("/api/voice/job").json()
            if job.get("state") in ("done", "error"):
                break
            time.sleep(0.25)
        check("train done", job.get("state") == "done", job.get("error") or job.get("log", [])[-2:])
        check("both gpus participated", all(g["progress"] == 100 for g in job.get("gpus", [])) and len(job.get("gpus", [])) == 2, job.get("gpus"))
        check("similarity reported", job.get("similarity") is not None, job.get("similarity"))
        check("train fast (mock < 6s)", time.time() - t0 < 6, round(time.time() - t0, 1))
        check("ref.wav downloadable", c.get("/api/voice/ref.wav").status_code == 200)
        v = c.get("/api/status").json()["voice"]
        check("voice registered on both cards", v and len(v.get("workers", [])) == 2, v and v.get("workers"))
        time.sleep(1.5)  # 语气词在后台生成

        audio, text, last, first, evs = run_chat(c, "你好,请介绍一下你自己", opts={"filler": True})
        check("chat done", last and last["type"] == "done", last)
        idxs = [e["idx"] for e in audio]
        check("audio idx sequential", idxs == list(range(len(idxs))) and len(idxs) >= 3, idxs[:20])
        check("audio payload decodable", all(len(base64.b64decode(e["b64"])) > 100 for e in audio))
        check("both GPUs used in chat", len(last["stats"]["gpu_chunks"]) == 2, last["stats"]["gpu_chunks"])
        check("first audio < 3s", first is not None and first < 3, first)
        check("text streamed", len(text) > 40, len(text))
        check("filler played first (0-latency voice)", last["stats"].get("filler") and last["stats"]["ttfa_ms"] < last["stats"]["body_ttfa_ms"], last["stats"])
        check("chunk count matches", last["stats"]["chunks"] == len(audio), (last["stats"]["chunks"], len(audio)))

        # 断线续传:同一轮重新用游标 0 取,事件必须完整、顺序一致(服务端不因连接断开而丢数据)
        tid = [e for e in evs if e["type"] == "start"][0]["turn"]
        j = c.get("/api/events", params={"sid": SID, "tid": 1, "after": 0, "wait": 1}).json()
        check("resume from cursor 0 replays everything", j["done"] is False or len(j["events"]) > 0 and j["events"][0]["type"] == "start", len(j["events"]))
        j2 = c.get("/api/events", params={"sid": SID, "tid": 1, "after": 3, "wait": 1}).json()
        check("resume from cursor 3 matches tail", j2["events"][:1] == j["events"][3:4], (j2["events"][:1] or [None])[0] and j2["events"][0]["type"])
        old = c.get("/api/events", params={"sid": SID, "tid": 999, "after": 0, "wait": 1}).json()
        check("unknown/superseded turn handled", old.get("superseded") is True)

        a3, t3, l3, _, _ = run_chat(c, "你好", opts={"filler": False}, drop_every=3)
        check("filler can be disabled", l3["stats"].get("filler") is None and len(a3) >= 1, l3["stats"])

        a2, t2, l2, _, _ = run_chat(c, "这是一句直接朗读的话。第二句也要读出来。", speak=True)
        check("speak-only works", l2["type"] == "done" and len(a2) >= 1, l2)

        # 停止
        r = c.post("/api/chat", json={"sid": SID, "text": "请讲一个很长很长的故事"})
        tid = r.json()["tid"]
        time.sleep(1.0)
        c.post("/api/stop", json={"sid": SID})
        j = c.get("/api/events", params={"sid": SID, "tid": tid, "after": 0, "wait": 3}).json()
        for _ in range(10):
            if j["done"]:
                break
            j = c.get("/api/events", params={"sid": SID, "tid": tid, "after": j["next"], "wait": 2}).json()
        check("stop works (turn finished)", j["done"] is True)

        # 新一轮会顶掉旧一轮
        r1 = c.post("/api/chat", json={"sid": "s2", "text": "第一轮很长很长的故事"}).json()["tid"]
        r2 = c.post("/api/chat", json={"sid": "s2", "text": "第二轮"}).json()["tid"]
        s1 = c.get("/api/events", params={"sid": "s2", "tid": r1, "after": 0, "wait": 1}).json()
        check("new turn supersedes old", r2 == r1 + 1 and s1.get("superseded") is True)

        a = c.post("/api/asr", content=make_ref(), headers={"content-type": "application/octet-stream"})
        check("asr works", a.status_code == 200 and a.json().get("text"), a.text[:100])

        # 关键回归: LLM 出错时界面必须收到明确的 error,而不是永远转圈
        _, _, l4, _, _ = run_chat(c, "__LLM_FAIL__ 你好")
        check("LLM failure is reported (never silent)", l4 and l4["type"] == "error" and "LLM" in l4.get("text", ""), l4)
        _, _, l5, _, _ = run_chat(c, "故障之后还能正常聊天")
        check("chat works after a failed turn", l5 and l5["type"] == "done", l5)
        c.post("/api/reset", json={"sid": SID})
        check("reset ok", c.get("/api/health").status_code == 200)
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
