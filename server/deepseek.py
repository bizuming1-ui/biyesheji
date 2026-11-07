"""DeepSeek API 流式客户端(v7:取代本地 Qwen3-4B 对话模型)。

设计目标
  * 最快首字:一个常驻 httpx 连接池(TLS 握手在填写 Key 时就预热完毕),thinking 默认关闭(思考模式首字要等推理链,慢几秒)。
  * 长文本:finish_reason == "length" 时自动「续写」,最多 DEEPSEEK_MAX_CONT 轮,对上层完全透明(调用方看到的是一条连续的文本流)。
  * 稳定:首字之前遇到 429 / 5xx / 网络错误自动指数退避重试;首字之后不重试(避免重复内容)。
  * 安全:Key 只保存在服务进程内存,绝不写日志、绝不写磁盘、绝不出现在 /api/status /api/diag。
  * 所有错误都翻译成中文可读信息,界面必定能看到真实原因。
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import AsyncIterator, Optional

import httpx

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")  # 当前能力最强;想要更快首字可在界面切到 deepseek-flash
MAX_TOKENS = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "16384"))
MAX_CONT = int(os.environ.get("DEEPSEEK_MAX_CONT", "6"))
CONNECT_TIMEOUT = float(os.environ.get("DEEPSEEK_CONNECT_TIMEOUT", "10"))
RETRIES = int(os.environ.get("DEEPSEEK_RETRIES", "2"))

CONTINUE_PROMPT = "请从上文被截断的地方直接继续往下写,不要重复已经写过的内容,不要加任何开场白或总结性的过渡语。"


class DeepSeekError(Exception):
    """带中文说明的错误。retryable=True 表示首字之前可以自动重试。"""

    def __init__(self, msg: str, status: int = 0, retryable: bool = False):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable


def _explain(status: int, body: str) -> DeepSeekError:
    detail = ""
    try:
        j = json.loads(body)
        detail = (j.get("error") or {}).get("message") or j.get("message") or ""
    except Exception:  # noqa: BLE001
        detail = body.strip()[:200]
    table = {
        400: "请求参数有误(%s)",
        401: "API Key 无效或已被停用,请在右上角重新填写(%s)",
        402: "DeepSeek 账户余额不足,请充值后重试(%s)",
        422: "请求参数无效(%s)",
        429: "请求太频繁,触发 DeepSeek 限流(%s)",
        500: "DeepSeek 服务器内部错误(%s)",
        502: "DeepSeek 网关错误(%s)",
        503: "DeepSeek 服务繁忙(%s)",
    }
    msg = (table.get(status) or "DeepSeek 返回 HTTP %d (%%s)" % status) % (detail or "无详情")
    return DeepSeekError(msg, status, retryable=status in (429, 500, 502, 503, 504))


class DeepSeekClient:
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _http(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._loop = loop
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=httpx.Timeout(connect=CONNECT_TIMEOUT, read=90.0, write=15.0, pool=10.0),
                limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300),
            )
        return self._client

    @staticmethod
    def _headers(key: str) -> dict:
        return {"Authorization": "Bearer " + key.strip(), "Content-Type": "application/json", "Accept": "text/event-stream"}

    # ------------------------------------------------------------------ 校验 / 预热 / 模型列表
    async def check(self, key: str) -> dict:
        """GET /models:同时完成 Key 校验、TLS 预热、可用模型列表。"""
        key = (key or "").strip()
        if len(key) < 8:
            raise DeepSeekError("请先填写 DeepSeek API Key(sk- 开头)")
        t0 = time.perf_counter()
        try:
            r = await self._http().get("/models", headers={"Authorization": "Bearer " + key})
        except httpx.HTTPError as e:
            raise DeepSeekError("无法连接 DeepSeek(%s)。请确认 Kaggle 已开启 Internet。" % (type(e).__name__)) from e
        if r.status_code != 200:
            raise _explain(r.status_code, r.text)
        ids = [m.get("id") for m in (r.json().get("data") or []) if m.get("id")]
        return {"models": ids, "latency_ms": round((time.perf_counter() - t0) * 1000), "best": pick_best(ids)}

    # ------------------------------------------------------------------ 流式对话
    async def stream(
        self,
        key: str,
        messages: list,
        model: str = "",
        thinking: bool = False,
        max_tokens: int = 0,
    ) -> AsyncIterator[tuple[str, str]]:
        """产出 ("text", 片段) / ("think", 片段) / ("cont", 轮次)。自动续写、自动退避重试。"""
        key = (key or "").strip()
        if not key:
            raise DeepSeekError("还没有填写 DeepSeek API Key:请在浏览器模拟器右上角或网页顶部填写后重试。")
        model = (model or DEFAULT_MODEL).strip()
        budget = max_tokens or MAX_TOKENS
        msgs = list(messages)
        produced_any = False
        for round_i in range(MAX_CONT + 1):
            round_text: list[str] = []
            finish = None
            attempt = 0
            while True:
                try:
                    async for kind, piece, fin in self._one_request(key, msgs, model, thinking, budget):
                        if fin is not None:
                            finish = fin
                            continue
                        if kind == "text":
                            produced_any = True
                            round_text.append(piece)
                        yield kind, piece
                    break
                except DeepSeekError as e:
                    # 首字之前:可重试错误 → 退避重试;max_tokens 过大 → 减半重试
                    if produced_any or round_text:
                        raise
                    if e.status == 400 and "max_tokens" in str(e) and budget > 2048:
                        budget //= 2
                        continue
                    if e.retryable and attempt < RETRIES:
                        attempt += 1
                        await asyncio.sleep(0.6 * (2 ** attempt))
                        continue
                    raise
                except (httpx.TransportError, httpx.StreamError) as e:
                    if produced_any or round_text or attempt >= RETRIES:
                        raise DeepSeekError("与 DeepSeek 的连接中断(%s)" % type(e).__name__) from e
                    attempt += 1
                    await asyncio.sleep(0.6 * (2 ** attempt))
            if finish == "length" and round_i < MAX_CONT:
                # 输出被 max_tokens 截断 → 把已生成部分作为 assistant,让模型无缝续写
                msgs = msgs + [
                    {"role": "assistant", "content": "".join(round_text)},
                    {"role": "user", "content": CONTINUE_PROMPT},
                ]
                yield "cont", str(round_i + 1)
                continue
            return

    async def _one_request(self, key, msgs, model, thinking, budget):
        body = {
            "model": model,
            "messages": msgs,
            "stream": True,
            "max_tokens": budget,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        if thinking:
            body["reasoning_effort"] = "high"
        else:
            body["temperature"] = 0.8
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        async with self._http().stream("POST", "/chat/completions", headers=self._headers(key), content=payload) as r:
            if r.status_code != 200:
                raw = (await r.aread()).decode("utf-8", "replace")
                raise _explain(r.status_code, raw)
            async for line in r.aiter_lines():
                if not line or line.startswith(":"):
                    continue  # keep-alive 注释
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    j = json.loads(data)
                except ValueError:
                    continue
                if j.get("error"):
                    raise DeepSeekError("DeepSeek 流中断: %s" % ((j["error"] or {}).get("message") or j["error"]))
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content"):
                        yield "think", d["reasoning_content"], None
                    if d.get("content"):
                        yield "text", d["content"], None
                    if ch.get("finish_reason"):
                        yield "", "", ch["finish_reason"]


def pick_best(ids: list) -> str:
    """从账户可用模型里挑当前能力最强的(环境变量 DEEPSEEK_MODEL 优先)。"""
    if os.environ.get("DEEPSEEK_MODEL"):
        return os.environ["DEEPSEEK_MODEL"]
    for want in ("deepseek-v4-pro", "deepseek-reasoner", "deepseek-chat", "deepseek-flash"):
        if want in ids:
            return want
    pro = [i for i in ids if "pro" in i]
    return (pro or ids or [DEFAULT_MODEL])[0]


client = DeepSeekClient()
