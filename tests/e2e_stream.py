"""v8 自检(Mock,无需 GPU):  CLONE_MOCK=1 python tests/e2e_stream.py
验证: 1) /api/chat_stream 二进制帧(hello→事件→end)  2) 音频与 /api/events(base64)逐字节一致
      3) 流中途断开后用 hello.tid + 已收事件数 回退到 /api/events 无丢失无重复  4) 歌曲参考音频触发人声分离且训练成功
      5) 首帧(hello)延迟 < 0.5s(证明不再等 POST→GET 两个往返)"""
import base64
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

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("TEST_PORT", "7863"))
BASE = "http://127.0.0.1:%d" % PORT
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" -> " + str(extra)[:200] if extra else ""))
    if not cond:
        fails.append(name)


def make_ref():
    sr = 24000
    t = np.arange(sr * 8) / sr
    x = 0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    x += 0.02 * np.sin(2 * np.pi * 440 * t) + 0.003 * np.random.randn(len(t))
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def parse_frames(buf: bytearray):
    out = []
    while len(buf) >= 5:
        kind, n = struct.unpack(">BI", bytes(buf[:5]))
        if len(buf) < 5 + n:
            break
        out.append((kind, bytes(buf[5:5 + n])))
        del buf[:5 + n]
    return out


env = dict(os.environ, CLONE_MOCK="1", LLM_BACKEND="local", PORT=str(PORT), CLONE_DATA=str(ROOT / "data_test_stream"), PYTHONUNBUFFERED="1")
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
    # ---- 训练(歌曲标记 → 人声分离)
    r = c.post("/api/voice/train", files={"file": ("ref.wav", make_ref(), "audio/wav")}, data={"text": "", "lang": "zh", "song": "1"})
    check("train accepted", r.status_code == 200, r.text)
    for _ in range(120):
        j = c.get("/api/voice/job").json() if c.get("/api/voice/job").status_code == 200 else {}
        if j.get("state") in ("done", "error"):
            break
        time.sleep(0.5)
    check("training done", j.get("state") == "done", j.get("error") or j.get("stage"))
    logtxt = "\n".join(j.get("log", [])) if isinstance(j.get("log"), list) else str(j.get("log"))
    check("vocal separation ran for song", "人声分离" in logtxt, logtxt[-200:])

    # ---- 流式对话
    SID = "t-stream"
    body = {"sid": SID, "text": "你好,请讲一个很短的故事。", "speak": False, "opts": {"breath": 0.4}}
    t0 = time.time()
    frames, buf, hello_at, first_audio_at = [], bytearray(), None, None
    with c.stream("POST", "/api/chat_stream", json=body, timeout=60) as resp:
        check("stream content-type binary", resp.headers.get("content-type", "").startswith("application/octet-stream"))
        for chunk in resp.iter_raw():
            buf += chunk
            for fr in parse_frames(buf):
                frames.append(fr)
                if fr[0] == 2 and hello_at is None:
                    hello_at = time.time() - t0
                if fr[0] == 1 and first_audio_at is None:
                    first_audio_at = time.time() - t0
    kinds = [k for k, _ in frames]
    check("first frame is hello", kinds and kinds[0] == 2)
    check("hello latency < 0.5s", hello_at is not None and hello_at < 0.5, hello_at)
    check("last frame is end", kinds and kinds[-1] == 4)
    tid = json.loads(frames[0][1])["tid"]
    ev_frames = [(k, d) for k, d in frames if k in (0, 1)]
    check("has audio frames", sum(1 for k, _ in ev_frames if k == 1) > 3, first_audio_at)
    last_json = [json.loads(d) for k, d in ev_frames if k == 0][-1]
    check("done event last", last_json.get("type") == "done", last_json.get("type"))

    # ---- 与 /api/events 一致(逐字节)
    j = c.get("/api/events", params={"sid": SID, "tid": tid, "after": 0, "wait": 1}).json()
    check("same event count", len(j["events"]) == len(ev_frames), (len(j["events"]), len(ev_frames)))
    same = True
    for ev, (k, d) in zip(j["events"], ev_frames):
        if ev["type"] == "audio":
            turn, idx, sr = struct.unpack(">III", d[:12])
            same &= (k == 1 and ev["idx"] == idx and ev["sr"] == sr and base64.b64decode(ev["b64"]) == d[12:])
        else:
            same &= (k == 0 and json.loads(d) == ev)
    check("stream frames == poll events (byte exact)", same)

    # ---- 中途断开 → 回退到 /api/events 游标续传
    body2 = dict(body, text="再讲一个长一点的故事吧,慢慢说。")
    got, buf, tid2 = [], bytearray(), None
    with c.stream("POST", "/api/chat_stream", json=body2, timeout=60) as resp:
        for chunk in resp.iter_raw():
            buf += chunk
            for fr in parse_frames(buf):
                if fr[0] == 2:
                    tid2 = json.loads(fr[1])["tid"]
                elif fr[0] in (0, 1):
                    got.append(fr)
            if len(got) >= 4:
                break  # 假装断线
    after, rest = len(got), []
    for _ in range(2000):
        j = c.get("/api/events", params={"sid": SID, "tid": tid2, "after": after, "wait": 5}, timeout=15).json()
        rest += j["events"]
        after = j["next"]
        if j["done"]:
            break
    full = c.get("/api/events", params={"sid": SID, "tid": tid2, "after": 0, "wait": 1}).json()["events"]
    check("resume after disconnect: no loss/dup", len(got) + len(rest) == len(full), (len(got), len(rest), len(full)))
    check("resumed turn finished with done", rest and rest[-1].get("type") == "done", rest[-1].get("type") if rest else None)

    # ---- 旧接口不受影响
    r = c.post("/api/chat", json={"sid": "t-old", "text": "你好", "speak": False, "opts": {}})
    check("legacy /api/chat still works", r.status_code == 200 and "tid" in r.json())
finally:
    p.terminate()
print("\n结果:", "全部通过" if not fails else "失败 %d 项: %s" % (len(fails), fails))
sys.exit(1 if fails else 0)
