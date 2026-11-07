"""分块上传 + 运行日志自检(Mock,无需 GPU):  CLONE_MOCK=1 python tests/e2e_upload.py
验证: 分块乱序/重复上传、缺块 422 补传、train_uploaded 幂等、非法 uid 被拒、训练完成、
      /api/diag 含 perf 耗时统计与 analysis 瓶颈分析、/api/perf、旧的一次性上传接口仍可用。"""
import io
import os
import random
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


def wav_bytes(sec=8):
    t = np.arange(24000 * sec) / 24000
    x = 0.25 * np.sin(2 * np.pi * 170 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.01 * np.random.randn(len(t))
    buf = io.BytesIO()
    sf.write(buf, x.astype("float32"), 24000, format="WAV")
    return buf.getvalue()


def wait_job(c, n=120):
    j = {}
    for _ in range(n):
        j = c.get("/api/voice/job").json()
        if j.get("state") in ("done", "error"):
            break
        time.sleep(0.5)
    return j


env = dict(os.environ, CLONE_MOCK="1", PORT=str(PORT), CLONE_DATA=str(ROOT / "data_test_upload"), PYTHONUNBUFFERED="1")
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
    data = wav_bytes(8)
    CH = 50 * 1024
    parts = [data[i:i + CH] for i in range(0, len(data), CH)]
    total = len(parts)
    uid = "testuid1234567890"
    check("multiple chunks", total >= 4, total)
    # 非法
    check("bad uid rejected", c.post("/api/voice/chunk", params=dict(uid="x", idx=0, total=2), content=b"abc").status_code == 400)
    check("bad idx rejected", c.post("/api/voice/chunk", params=dict(uid=uid, idx=99, total=total), content=b"abc").status_code == 400)
    check("empty chunk rejected", c.post("/api/voice/chunk", params=dict(uid=uid, idx=0, total=total), content=b"").status_code == 400)
    # 乱序 + 重复,先故意漏掉一块
    order = list(range(total)); random.shuffle(order)
    skip = order.pop()
    for i in order + order[:2]:
        r = c.post("/api/voice/chunk", params=dict(uid=uid, idx=i, total=total), content=parts[i], headers={"Content-Type": "application/octet-stream"})
        check("chunk %d ok" % i, r.status_code == 200 and r.json()["ok"])
    st = c.get("/api/voice/chunk_status", params=dict(uid=uid)).json()
    check("status lists got chunks (resume)", len(st["got"]) == total - 1 and skip not in st["got"], st)
    r = c.post("/api/voice/train_uploaded", json=dict(uid=uid, total=total, text="", lang="zh"))
    check("missing chunk -> 422 + missing list", r.status_code == 422 and r.json()["missing"] == [skip], r.text)
    c.post("/api/voice/chunk", params=dict(uid=uid, idx=skip, total=total), content=parts[skip])
    r = c.post("/api/voice/train_uploaded", json=dict(uid=uid, total=total, text="测试文本", lang="zh"))
    check("train_uploaded accepted", r.status_code == 200 and r.json().get("bytes") == len(data), r.text)
    r2 = c.post("/api/voice/train_uploaded", json=dict(uid=uid, total=total, text="测试文本", lang="zh"))
    check("train_uploaded idempotent (lost response retry)", r2.status_code == 200 and r2.json().get("dup"), r2.text)
    j = wait_job(c)
    check("chunked training done", j.get("state") == "done", j.get("error"))
    # 客户端上报的上传性能事件 → 分析器能读到
    c.post("/api/clientlog", json=[{"kind": "perf", "msg": "upload", "ms": 52000, "extra": '{"ok":true,"kb":1400,"src_kb":4683,"kbps":27,"retries":3,"chunks":8}'}])
    d = c.get("/api/diag").json()
    check("diag has perf", "perf" in d and any(k.startswith("train/") for k in d["perf"]), list(d.get("perf", {}))[:6])
    check("perf has cpu/gpu spans", "cpu/preprocess_reference" in d["perf"] and any(k.startswith("gpu") for k in d["perf"]), list(d["perf"]))
    check("perf has upload spans", "upload/chunk_recv" in d["perf"] and "upload/assemble" in d["perf"])
    an = d.get("analysis") or []
    check("analysis present", len(an) > 0, an[:2])
    check("analysis flags slow upload", any(a["area"] == "上传" and a["level"] == "warn" for a in an), [a for a in an if a["area"] == "上传"])
    check("analysis names slowest training stage", any(a["area"] == "训练" and "最慢阶段" in a["msg"] for a in an), [a["msg"] for a in an][:4])
    check("diag version v6", d["config"].get("CODE_VERSION") == "v6")
    pr = c.get("/api/perf").json()
    check("/api/perf", "summary" in pr and "recent" in pr)
    # 旧接口兼容
    r = c.post("/api/voice/train", files={"file": ("ref.wav", data, "audio/wav")}, data={"text": "旧接口", "lang": "zh"})
    check("legacy one-shot upload still works", r.status_code == 200, r.text)
    j = wait_job(c)
    check("legacy training done", j.get("state") == "done", j.get("error"))
    # 前端静态检查
    html = c.get("/").text
    for k in ("id=\"logbtn\"", "uploadChunks", "prepareRef", "运行日志", "/api/voice/train_uploaded"):
        check("index.html contains " + k, k in html)
finally:
    p.terminate()
    import shutil
    shutil.rmtree(ROOT / "data_test_upload", ignore_errors=True)
print("\n结果:", "全部通过" if not fails else "失败 %s" % fails)
sys.exit(1 if fails else 0)
