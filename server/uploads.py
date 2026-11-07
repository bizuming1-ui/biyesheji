"""分块上传(v6)。

为什么需要:公网隧道(trycloudflare)对「一次性上传几 MB 的大请求」很不友好 —— 慢、会被中途掐断,
浏览器那边表现就是『已点击开始训练 · 正在上传参考音频…』永远不动、连接显示『重连中』。
现在:浏览器把音频切成小块(≈192KB),每块独立 POST、独立超时、独立重试;已到达的块不会重传(断点续传),
全部到齐后再发一个很小的「开始训练」请求。块写入是幂等的(同一块重复到达没有副作用)。
"""
import re
import threading
import time

MAX_CHUNK = 1024 * 1024
MAX_TOTAL_CHUNKS = 400
MAX_BYTES = 80 * 1024 * 1024
TTL = 1800.0

_UID = re.compile(r"^[A-Za-z0-9_-]{6,48}$")
_LOCK = threading.Lock()
_STORE: dict[str, dict] = {}


class UploadError(ValueError):
    pass


def _gc(now: float) -> None:
    for k in [k for k, v in _STORE.items() if now - v["t"] > TTL]:
        _STORE.pop(k, None)


def put_chunk(uid: str, idx: int, total: int, data: bytes) -> dict:
    if not _UID.match(uid or ""):
        raise UploadError("无效的上传编号")
    if not (1 <= total <= MAX_TOTAL_CHUNKS) or not (0 <= idx < total):
        raise UploadError("无效的分块序号")
    if not data or len(data) > MAX_CHUNK:
        raise UploadError("分块为空或过大")
    now = time.time()
    with _LOCK:
        _gc(now)
        u = _STORE.setdefault(uid, {"t": now, "total": total, "chunks": {}})
        if u["total"] != total:
            raise UploadError("分块总数不一致")
        u["t"] = now
        u["chunks"][idx] = bytes(data)
        if sum(len(c) for c in u["chunks"].values()) > MAX_BYTES:
            _STORE.pop(uid, None)
            raise UploadError("上传内容过大")
        return {"got": len(u["chunks"]), "total": total}


def status(uid: str) -> dict:
    with _LOCK:
        u = _STORE.get(uid)
        if not u:
            return {"got": [], "total": 0}
        return {"got": sorted(u["chunks"]), "total": u["total"]}


def assemble(uid: str, total: int):
    """返回 (bytes, None) 或 (None, 缺失块序号列表)。"""
    with _LOCK:
        u = _STORE.get(uid)
        if not u or u["total"] != total:
            return None, list(range(total))
        missing = [i for i in range(total) if i not in u["chunks"]]
        if missing:
            return None, missing
        u["t"] = time.time()
        return b"".join(u["chunks"][i] for i in range(total)), None


def drop(uid: str) -> None:
    with _LOCK:
        _STORE.pop(uid, None)
