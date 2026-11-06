"""FastAPI 入口。  启动: python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

稳定性设计(v3):
  * 对话不再依赖 WebSocket。隧道/代理悄悄丢掉 WebSocket 时,旧版会「发了没反应」。
    现在是 HTTP: POST /api/chat 创建一轮对话(服务端独立生成,与连接无关),
    GET /api/events?after=N 长轮询取事件(文本/音频/状态)。网络断了就用同一个游标重试,数据一个不丢。
  * 保留 /ws 作为可选通道(不再被前端使用)。
"""
import asyncio
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from . import config
from .engine import Engine, pack_audio

engine = Engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await engine.start()
    yield
    for w in engine.workers:
        w.stop()


app = FastAPI(title="Voice Clone Studio", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def allow_embed(request: Request, call_next):
    resp = await call_next(request)
    # 允许被「浏览器模拟器」iframe 嵌入,并允许麦克风/自动播放/摄像头
    resp.headers["Content-Security-Policy"] = "frame-ancestors *"
    resp.headers["Permissions-Policy"] = "microphone=*, autoplay=*, camera=*"
    resp.headers["Cache-Control"] = "no-store"
    if "x-frame-options" in resp.headers:
        del resp.headers["x-frame-options"]
    return resp


@app.get("/")
async def index():
    return FileResponse(config.WEB_DIR / "index.html", media_type="text/html; charset=utf-8")


@app.get("/api/health")
async def health():
    return {"ok": True, "mock": config.MOCK}


@app.get("/api/ping")
async def ping():
    return {"t": __import__("time").time()}


@app.get("/api/status")
async def status():
    return await asyncio.to_thread(engine.status)


@app.post("/api/reload")
async def reload_models():
    """某个模型加载失败时,界面上的「重新加载模型」按钮调用它(不用重启整个 Kaggle 单元格)。"""
    names = engine.reload_failed()
    return {"ok": True, "reloading": names}


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.post("/api/voice/train")
async def train(file: UploadFile = File(...), text: str = Form(""), lang: str = Form("zh")):
    data = await file.read()
    if len(data) < 2000:
        raise HTTPException(400, "音频文件为空或太小")
    try:
        await engine.start_training(data, text, lang)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.get("/api/voice/job")
async def job():
    return engine.job


@app.get("/api/voice/ref.wav")
async def ref_wav():
    v = engine.voice
    p = Path(v["ref_path"]) if v and v.get("ref_path") else config.VOICE_DIR / "current" / "ref.wav"
    if not p.exists():
        raise HTTPException(404, "尚无参考音频")
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/asr")
async def asr(request: Request):
    data = await request.body()
    try:
        return await engine.transcribe(data)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": (str(e).strip().splitlines() or ["未知错误"])[-1]}, status_code=500)


# ---------------------------------------------------------------- 与连接无关的对话接口
@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    sid = str(body.get("sid") or "anon")[:64]
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(400, "空消息")
    tid = await engine.start_chat(sid, text, dict(body.get("opts") or {}), bool(body.get("speak")))
    return {"ok": True, "tid": tid}


@app.get("/api/events")
async def events(sid: str, tid: int, after: int = 0, wait: float = 15.0):
    return await engine.poll_events(sid[:64], tid, max(0, after), wait)


@app.post("/api/stop")
async def stop(request: Request):
    body = await request.json()
    engine.stop_chat(str(body.get("sid") or "anon")[:64])
    return {"ok": True}


@app.post("/api/reset")
async def reset(request: Request):
    body = await request.json()
    engine.reset_chat(str(body.get("sid") or "anon")[:64])
    return {"ok": True}


# ---------------------------------------------------------------- 可选 WebSocket(兼容旧客户端)
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sess = engine.new_session()
    lock = asyncio.Lock()

    async def send_json(obj):
        async with lock:
            await ws.send_text(json.dumps(obj, ensure_ascii=False))

    async def send_audio(turn, idx, sr, pcm):
        async with lock:
            await ws.send_bytes(pack_audio(turn, idx, sr, pcm))

    async def guarded(coro):
        try:
            await coro
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            try:
                await send_json({"type": "error", "text": "服务端异常: " + (str(e).strip().splitlines() or [repr(e)])[-1]})
            except Exception:  # noqa: BLE001
                pass

    try:
        while True:
            msg = json.loads(await ws.receive_text())
            t = msg.get("type")
            if t == "ping":
                await send_json({"type": "pong"})
            elif t in ("chat", "speak"):
                engine.cancel_turn(sess)
                if sess.task:
                    try:
                        await asyncio.wait_for(asyncio.shield(sess.task), timeout=0.5)
                    except Exception:  # noqa: BLE001
                        pass
                text = str(msg.get("text", "")).strip()
                if not text:
                    continue
                opts = dict(msg.get("opts") or {})
                opts["speak_only"] = t == "speak"
                sess.task = asyncio.create_task(guarded(engine.run_turn(sess, text, opts, send_json, send_audio)))
            elif t == "stop":
                engine.cancel_turn(sess)
                await send_json({"type": "stopped"})
            elif t == "reset":
                engine.cancel_turn(sess)
                sess.history.clear()
                await send_json({"type": "reset"})
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        engine.cancel_turn(sess)
