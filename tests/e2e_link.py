"""v10 连接稳定性自检(Mock 模式,无需 GPU):  CLONE_MOCK=1 python tests/e2e_link.py
验证:
  1. /api/state 合并了 ping+status+job,并且在「并发风暴」下仍然 < 1 秒返回(旧 /api/status 每次都现算);
  2. /api/ping /api/state 下发备用线路地址(run.py 写入的 TUNNEL_FILE),且读到新网址后立刻更新;
  3. /link.js 可被页面加载;index.html 引用了它;
  4. 事件循环延迟 lag_ms 存在且很小(以后能区分「服务端阻塞」还是「公网隧道慢」);
  5. 服务端训练 + 对话进行中,/api/state 仍然稳定(P95 延迟 / 零失败)。"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("TEST_PORT", "7871"))
BASE = "http://127.0.0.1:%d" % PORT
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" -> " + str(extra)[:200] if extra else ""))
    if not cond:
        fails.append(name)


def main():
    tun = Path(tempfile.gettempdir()) / ("tun_test_%d.json" % PORT)
    if tun.exists():
        tun.unlink()
    env = dict(os.environ, CLONE_MOCK="1", CLONE_TUNNEL_FILE=str(tun), PORT=str(PORT))
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(PORT), "--no-access-log"], cwd=ROOT, env=env)
    try:
        c = httpx.Client(base_url=BASE, timeout=20)
        for _ in range(60):
            try:
                if c.get("/api/health").status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        r = c.get("/api/state").json()
        check("state has status+job+lag", "status" in r and "lag_ms" in r and "job" in r["status"] and r["code_version"] == "v10", list(r))
        check("state alts empty without tunnels", r["alts"] == [], r["alts"])
        tun.write_text(json.dumps({"primary": "https://a.trycloudflare.com", "urls": ["https://a.trycloudflare.com", "https://b.free.pinggy.link"]}))
        r = c.get("/api/ping").json()
        check("ping publishes alts", r["alts"] == ["https://a.trycloudflare.com", "https://b.free.pinggy.link"] and r["primary"].startswith("https://a."), r)
        tun.write_text("{broken")
        r = c.get("/api/state").json()
        check("broken tunnel file never breaks state", r["alts"] == [], r["alts"])
        js = c.get("/link.js")
        check("/link.js served as javascript", js.status_code == 200 and "javascript" in js.headers["content-type"] and "maybeFailover" in js.text)
        html = c.get("/").text
        check("index.html loads link.js before player", "/link.js" in html and html.index("/link.js") < html.index("/player.js"))
        check("index.html uses /api/state (no ping timer)", "/api/state" in html and "/api/ping\"" not in html)

        # 并发风暴:40 个并发 /api/state + 状态轮询,同时跑训练 + 对话
        lat, errs = [], []

        def hammer(n):
            cc = httpx.Client(base_url=BASE, timeout=20)
            for _ in range(n):
                t0 = time.time()
                try:
                    ok = cc.get("/api/state").status_code == 200
                    lat.append((time.time() - t0) * 1000)
                    if not ok:
                        errs.append("status")
                except Exception as e:  # noqa: BLE001
                    errs.append(str(e))
                time.sleep(0.05)

        ths = [threading.Thread(target=hammer, args=(25,)) for _ in range(40)]
        # 训练 + 对话
        import numpy as np
        import soundfile as sf
        import io

        sr = 24000
        t = np.arange(sr * 8) / sr
        buf = io.BytesIO()
        sf.write(buf, (0.3 * np.sin(2 * np.pi * 180 * t)).astype("float32"), sr, format="WAV")
        for th in ths:
            th.start()
        rt = c.post("/api/voice/train", files={"file": ("ref.wav", buf.getvalue(), "audio/wav")}, data={"text": "你好,这是一个测试。", "lang": "zh"})
        check("training accepted during storm", rt.status_code == 200, rt.text[:100])
        tid = c.post("/api/chat", json={"sid": "lt", "text": "你好", "speak": True, "opts": {}}).json()["tid"]
        after, done, tries = 0, False, 0
        while not done and tries < 60:
            j = c.get("/api/events", params={"sid": "lt", "tid": tid, "after": after, "wait": 3}, timeout=15).json()
            after, done = j["next"], j["done"]
            tries += 1
        check("chat finishes during storm", done)
        for th in ths:
            th.join()
        lat.sort()
        p95 = lat[int(len(lat) * 0.95)] if lat else 1e9
        check("1000 concurrent /api/state: zero failures", not errs and len(lat) >= 1000, (len(lat), errs[:2]))
        check("/api/state P95 < 1000ms under storm", p95 < 1000, "p95=%.0fms max=%.0fms" % (p95, lat[-1] if lat else -1))
        r = c.get("/api/ping").json()
        check("event-loop lag stays small", r["lag_ms"] < 500, r["lag_ms"])
    finally:
        p.terminate()
        try:
            p.wait(10)
        except Exception:  # noqa: BLE001
            p.kill()
        if tun.exists():
            tun.unlink()
    print("\nFAILED: %s" % fails if fails else "\nALL PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
