"""性能剖析(v6):记录每个环节的耗时,并给出「哪里慢、怎么提升」的自动分析,供「运行日志」面板使用。

* record(name, seconds)     手动记录一次耗时
* span(name)                with 语句自动计时
* summary()                 每个环节的 次数 / 平均 / P95 / 最大 / 最近
* analyze(diag)             规则引擎:从训练时间线 / 上传速度 / 接口延迟 / 显存 / 首声 里找出瓶颈并给出建议
所有函数都「绝不抛异常」:剖析系统自己不能成为故障源。
"""
import collections
import contextlib
import json
import threading
import time

_LOCK = threading.Lock()
_SPANS = collections.deque(maxlen=1500)  # (wall_time, name, seconds)


def record(name: str, seconds: float) -> None:
    try:
        with _LOCK:
            _SPANS.append((round(time.time(), 3), str(name)[:80], round(float(seconds), 4)))
    except Exception:  # noqa: BLE001
        pass


@contextlib.contextmanager
def span(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record(name, time.perf_counter() - t0)


def recent(n: int = 80):
    with _LOCK:
        return [{"t": t, "name": k, "s": s} for t, k, s in list(_SPANS)[-n:]]


def _pct(vals, q):
    if not vals:
        return 0.0
    v = sorted(vals)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def summary() -> dict:
    by: dict[str, list] = {}
    with _LOCK:
        items = list(_SPANS)
    for _, k, s in items:
        by.setdefault(k, []).append(s)
    out = {}
    for k, v in by.items():
        out[k] = {
            "n": len(v), "avg": round(sum(v) / len(v), 3), "p95": round(_pct(v, 0.95), 3),
            "max": round(max(v), 3), "last": round(v[-1], 3), "total": round(sum(v), 2),
        }
    return out


# ------------------------------------------------------------------ 分析
_STAGE_ADVICE = [
    ("等待模型加载", "模型还在加载(首次运行要下载约 15GB)。让 Kaggle 单元格保持运行,模型常驻显存后这一步为 0;可设置 HF_TOKEN 提高下载速度。"),
    ("参考音频清洗", "CPU 上的降噪/切静音较慢:录音请控制在 8~20 秒、环境安静(信噪比 >28dB 时会跳过降噪)。"),
    ("双卡并行", "GPU 合成校验句是主要耗时:检查 TTS_DTYPE 是否被降级到 float32(慢约 2 倍),必要时重启单元格;看下方『环节耗时』里 gpu0/gpu1 哪张更慢。"),
    ("自动识别", "Whisper 识别较慢:直接在『参考文本』框里填写朗读内容可跳过识别。"),
    ("同步声音", "另一张卡补充声纹较慢:通常是该卡 TTS 刚加载完,属于一次性开销。"),
    ("数值异常", "发生了 NaN/精度回退并重启 TTS 进程:这会多花 1~2 分钟,说明当前精度不稳定,请确认 TTS_DTYPE=bfloat16。"),
]


def _advice_for(stage: str) -> str:
    for k, v in _STAGE_ADVICE:
        if k in stage:
            return v
    return ""


def _parse_extra(e):
    try:
        return json.loads(e.get("extra") or "{}")
    except Exception:  # noqa: BLE001
        return {}


def analyze(d: dict) -> list:
    """返回 [{level: ok|info|warn|bad, area, msg, advice}, ...],按严重程度排序。"""
    out = []

    def add(level, area, msg, advice=""):
        out.append({"level": level, "area": area, "msg": msg, "advice": advice})

    try:
        # 1) 训练各阶段耗时
        job = d.get("job") or {}
        tl = job.get("timeline") or []
        if tl:
            end = job.get("elapsed")
            if end is None:
                end = (d.get("job_age_s") or (tl[-1][0] + 1)) if job.get("state") == "running" else tl[-1][0]
            stages = []
            for i, (t0, name) in enumerate(tl):
                t1 = tl[i + 1][0] if i + 1 < len(tl) else end
                stages.append((max(0.0, float(t1) - float(t0)), name))
            total = sum(s for s, _ in stages) or 1.0
            stages.sort(reverse=True)
            top_s, top_name = stages[0]
            share = top_s / total
            lvl = "bad" if (top_s > 90 and share > 0.5) else "warn" if (top_s > 30 and share > 0.4) else "info"
            add(lvl, "训练", "最慢阶段:%s 用了 %.1fs(占 %.0f%% / 总计 %.1fs)" % (top_name, top_s, share * 100, total), _advice_for(top_name))
            for s, name in stages[1:3]:
                if s > 15:
                    add("info", "训练", "阶段「%s」%.1fs" % (name, s), _advice_for(name))
            if job.get("state") == "running" and (d.get("job_age_s") or 0) > 240:
                add("bad", "训练", "训练已运行 %.0fs 仍未结束" % d.get("job_age_s"), "查看下方服务端日志;若停在『等待模型加载』说明模型还没就绪;停在 GPU 合成说明 CUDA 异常,点『重新加载模型』。")
        if job.get("state") == "error":
            add("bad", "训练", "训练失败:" + str(job.get("error"))[:200], "按日志里的真实原因处理;数值异常会自动降级精度并重试。")
        # 2) 上传(浏览器端上报的 perf 事件)
        ups = [e for e in (d.get("client_log") or []) if e.get("kind") == "perf" and e.get("msg") == "upload"]
        if ups:
            x = _parse_extra(ups[-1])
            kbps = x.get("kbps") or 0
            ms = ups[-1].get("ms") or 0
            retries = x.get("retries") or 0
            if x.get("ok") is False:
                add("bad", "上传", "参考音频上传失败(%s KB,耗时 %.1fs,重试 %s 次)" % (x.get("kb"), ms / 1000, retries), "网络/隧道不稳定:已自动分块续传;请检查 Kaggle 单元格是否仍在运行、Cloudflare 隧道是否被重建。")
            elif kbps and kbps < 150:
                add("warn", "上传", "上传速度只有 %.0f KB/s(%s KB 用了 %.1fs)" % (kbps, x.get("kb"), ms / 1000), "隧道带宽低。已在浏览器端压缩为 24kHz 单声道 WAV 并限制 30 秒;仍慢请换网络。")
            elif retries:
                add("info", "上传", "上传时有 %s 次分块重试(已自动续传,未丢数据)" % retries, "隧道偶发丢包,可忽略。")
            else:
                add("ok", "上传", "上传正常:%s KB / %.1fs(%.0f KB/s)" % (x.get("kb"), ms / 1000, kbps))
            if x.get("src_kb") and x.get("kb"):
                add("info", "上传", "浏览器端压缩 %s KB → %s KB(%.0f%%)" % (x.get("src_kb"), x.get("kb"), 100.0 * x["kb"] / max(1, x["src_kb"])))
        # 3) 接口延迟(浏览器每 10 秒上报的统计)
        stats = [e for e in (d.get("client_log") or []) if e.get("kind") == "stats"]
        if stats:
            sx = _parse_extra(stats[-1])
            for k, v in (sx.get("http") or {}).items():
                if v.get("fail", 0) > 0 and v.get("n", 0) > 0:
                    rate = v["fail"] / v["n"]
                    add("bad" if rate > 0.1 else "warn", "网络", "接口 /api/%s 失败 %d/%d 次" % (k, v["fail"], v["n"]), "隧道丢包或服务端卡顿;失败率高时查看『服务端日志』是否有异常。")
                elif k != "events" and v.get("avg", 0) > 800:
                    add("warn", "网络", "接口 /api/%s 平均 %dms(最大 %dms)" % (k, v["avg"], v["max"]), "延迟高:公网隧道往返慢;不影响 GPU 速度。")
            au = sx.get("audio") or {}
            if au and au.get("underruns", 0) > 0:
                add("warn", "播放", "播放缓冲不足 %d 次(出现停顿)" % au["underruns"], "合成速度跟不上播放:保持『无停顿优先』打开;避免同时训练和对话。")
        # 4) GPU / 显存
        for g in d.get("gpus") or []:
            pct = 100.0 * g["mem_used"] / max(1, g["mem_total"])
            if pct > 93:
                add("bad", "GPU", "GPU%d 显存 %.0f%%(%.1fG/%.1fG),有 OOM 风险" % (g["index"], pct, g["mem_used"] / 1024, g["mem_total"] / 1024), "降低 LLM_MAX_MEM_GIB 或关闭长文模式。")
            elif pct > 85:
                add("warn", "GPU", "GPU%d 显存 %.0f%%" % (g["index"], pct), "接近上限,长文模式可能 OOM。")
        # 5) 首声
        t = d.get("turn") or {}
        ttfa = t.get("ttfa_ms") or t.get("first_audio_ms")
        if ttfa:
            add("bad" if ttfa > 6000 else "warn" if ttfa > 3000 else "ok", "对话", "首声延迟 %.2fs(LLM 首字 %sms)" % (ttfa / 1000, t.get("llm_first_ms")), "首声慢:确认语气词缓存已就绪(训练后台生成);LLM 首字慢说明 GPU 正忙。" if ttfa > 3000 else "")
        # 6) 模型加载耗时
        for name, w in (d.get("workers") or {}).items():
            ls = (w.get("info") or {}).get("load_s")
            if ls and ls > 120:
                add("info", "模型", "%s 加载用了 %.0fs" % (name, ls), "首次需要下载模型;之后同一会话内为 0。")
            if w.get("state") == "error":
                add("bad", "模型", "%s 加载失败:%s" % (name, str(w.get("error") or "")[-160:]), "点界面上『重新加载模型』。")
        # 7) 磁盘
        disk = (d.get("system") or {}).get("disk") or {}
        if disk.get("free_gb") is not None and disk["free_gb"] < 3:
            add("bad", "磁盘", "剩余磁盘仅 %.1fGB" % disk["free_gb"], "清理 /kaggle/working 里不需要的文件。")
        # 8) 各环节耗时(服务端 perf)
        pf = d.get("perf") or {}
        slow = sorted(((v["avg"], k, v) for k, v in pf.items() if k.startswith(("gpu", "aux", "cpu", "upload"))), reverse=True)[:3]
        for avg, k, v in slow:
            if avg > 5:
                add("info", "环节", "%s 平均 %.1fs(最大 %.1fs,%d 次)" % (k, avg, v["max"], v["n"]))
        if not out:
            add("ok", "总览", "暂无足够数据:先上传参考音频并点『开始训练』,再回来看分析。")
        order = {"bad": 0, "warn": 1, "info": 2, "ok": 3}
        out.sort(key=lambda a: order.get(a["level"], 9))
    except Exception as e:  # noqa: BLE001
        out.append({"level": "info", "area": "分析", "msg": "分析器内部错误: %r" % e, "advice": ""})
    return out
