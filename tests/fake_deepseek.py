"""假的 DeepSeek 服务(只用于离线自检):实现 /models 与流式 /chat/completions,可按触发词模拟 401/402/429/长文续写/思考模式。
用法: FAKE_DS_PORT=7871 python tests/fake_deepseek.py"""
import asyncio
import json
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

GOOD = "sk-test-good-key-1234567890"
app = FastAPI()
LOG: list = []
STATE = {"n429": 0}


def authed(req: Request) -> bool:
    return req.headers.get("authorization", "") == "Bearer " + GOOD


def err(status: int, msg: str):
    return JSONResponse({"error": {"message": msg, "type": "x"}}, status_code=status)


@app.get("/models")
async def models(req: Request):
    if not authed(req):
        return err(401, "Authentication Fails (invalid key)")
    return {"object": "list", "data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}]}


@app.get("/_debug")
async def debug():
    return LOG


def sse(obj):
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()


@app.post("/chat/completions")
async def chat(req: Request):
    if not authed(req):
        return err(401, "Authentication Fails (invalid key)")
    body = await req.json()
    LOG.append({"model": body.get("model"), "thinking": body.get("thinking"), "max_tokens": body.get("max_tokens"), "n_msgs": len(body["messages"])})
    last = body["messages"][-1]["content"]
    first_user = [m for m in body["messages"] if m["role"] == "user"][-1]["content"] if body["messages"][-1]["role"] != "user" else last
    users = " ".join(m["content"] for m in body["messages"] if m["role"] == "user")
    if "__402__" in users:
        return err(402, "Insufficient Balance")
    if "__429_ONCE__" in users and STATE["n429"] < 1:
        STATE["n429"] += 1
        return err(429, "Rate limit reached")

    is_cont = "不要重复已经写过的内容" in last
    if "__LONG__" in users:
        part1 = "好的,我来讲一个长故事。从前有座山,山里有座庙。庙里有个老和尚,每天都在讲故事。他讲的故事特别长,小和尚总是听着听着就睡着了。"
        part2 = "后来小和尚长大了,也成了老和尚,他也开始给新来的小和尚讲同样的故事。故事就这样一代一代传了下去,再也没有人记得是谁讲的第一遍。"
        text, fin = (part2, "stop") if is_cont else (part1, "length")
    elif "__THINK__" in users:
        text, fin = "想好了,答案是四十二。这是我认真思考之后的结论,希望对你有帮助。", "stop"
    else:
        text, fin = "嗯,你好呀。今天过得怎么样呢?我是用 DeepSeek 回答你的,这是一段用来测试的回答,它会被切成很多短句,然后逐句合成语音。希望播放的时候一点停顿都没有。", "stop"

    async def gen():
        yield b": keep-alive\n\n"
        if "__THINK__" in users and body.get("thinking", {}).get("type") == "enabled":
            for i in range(3):
                yield sse({"choices": [{"delta": {"reasoning_content": "思考中%d " % i}}]})
                await asyncio.sleep(0.05)
        for i in range(0, len(text), 3):
            await asyncio.sleep(0.01)
            yield sse({"choices": [{"delta": {"content": text[i:i + 3]}}]})
        yield sse({"choices": [{"delta": {}, "finish_reason": fin}]})
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("FAKE_DS_PORT", "7871")), log_level="warning")
