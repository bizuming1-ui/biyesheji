"""引擎:GPU 进程池 + 声音训练流水线 + 对话/语音流水线。"""
import asyncio
import base64
import json
import multiprocessing as mp
import queue as pyqueue
import shutil
import struct
import subprocess
import threading
import time
import uuid

import numpy as np

from . import audio_utils as au
from . import config
from .text_chunker import SentenceChunker, clean_for_tts, detect_language, is_speakable


# ----------------------------------------------------------------------------
# 进程句柄
# ----------------------------------------------------------------------------
class Worker:
    def __init__(self, name: str, kind: str, gpu: int, loop: asyncio.AbstractEventLoop):
        from .workers import worker_main

        self.name, self.kind, self.gpu, self.loop = name, kind, gpu, loop
        self.pending: dict[int, asyncio.Queue] = {}
        self.next_id = 1
        self.inflight = 0
        self.state = "starting"
        self.error = None
        self.info: dict = {}
        self.busy_ms = 0.0
        self._spawn()

    def _spawn(self):
        from .workers import worker_main

        ctx = mp.get_context("spawn")
        self.inq, self.outq = ctx.Queue(), ctx.Queue()
        self.cancel = ctx.Value("q", 0)
        self.proc = ctx.Process(target=worker_main, args=(self.kind, self.gpu, self.inq, self.outq, self.cancel), daemon=True)

    def start(self):
        self.proc.start()
        threading.Thread(target=self._reader, args=(self.proc, self.outq), daemon=True, name="reader-" + self.name).start()

    def respawn(self):
        """进程崩溃(OOM-kill 等)后,重新拉起一个干净的进程。"""
        try:
            if self.proc.is_alive():
                self.proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        self.pending.clear()
        self.inflight = 0
        self.next_id = 1
        self._spawn()
        self.start()

    def _reader(self, proc, outq):
        while True:
            try:
                msg = outq.get(timeout=1.0)
            except pyqueue.Empty:
                if not proc.is_alive():
                    break
                continue
            except (EOFError, OSError):
                break
            try:
                self.loop.call_soon_threadsafe(self._dispatch, msg)
            except RuntimeError:
                return
        try:  # 进程已退出:让所有正在等待它的请求立刻失败,而不是永远等下去
            self.loop.call_soon_threadsafe(self._fail_all, proc)
        except RuntimeError:
            pass

    def _fail_all(self, proc):
        if proc is not self.proc:
            return
        self.state = "error"
        self.error = "工作进程 %s 意外退出(可能显存不足被系统终止),点「重新加载模型」即可恢复" % self.name
        for rid, q in list(self.pending.items()):
            q.put_nowait({"id": rid, "type": "error", "error": self.error})

    def _dispatch(self, msg):
        q = self.pending.get(msg["id"])
        if q is not None:
            q.put_nowait(msg)

    def submit(self, op: str, **args):
        rid = self.next_id
        self.next_id += 1
        q: asyncio.Queue = asyncio.Queue()
        self.pending[rid] = q
        self.inflight += 1
        self.inq.put({"id": rid, "op": op, "args": args})
        return rid, q

    def _done(self, rid):
        self.pending.pop(rid, None)
        self.inflight = max(0, self.inflight - 1)

    async def call(self, op: str, _timeout: float | None = None, **args):
        rid, q = self.submit(op, **args)
        try:
            while True:
                m = await (asyncio.wait_for(q.get(), _timeout) if _timeout else q.get())
                if m["type"] == "error":
                    raise RuntimeError(m["error"])
                if m["type"] == "final":
                    return m["data"]
        finally:
            self._done(rid)

    async def stream(self, rid, q, stall: float = 90.0):
        try:
            while True:
                try:
                    m = await asyncio.wait_for(q.get(), stall)
                except asyncio.TimeoutError:
                    self.cancel.value = max(self.cancel.value, rid)
                    raise RuntimeError("%s 超过 %d 秒没有响应" % (self.name, int(stall)))
                if m["type"] == "error":
                    raise RuntimeError(m["error"])
                if m["type"] == "delta":
                    yield m["data"]
                elif m["type"] == "final":
                    return
        finally:
            self._done(rid)

    def stop(self):
        try:
            self.inq.put(None)
            self.proc.join(timeout=2)
            if self.proc.is_alive():
                self.proc.terminate()
        except Exception:  # noqa: BLE001
            pass


_GPU_CACHE = {"t": 0.0, "v": []}
_GPU_LOCK = threading.Lock()


def gpu_stats():
    """nvidia-smi 在模型加载期间可能很慢,不能让它卡住事件循环 → 1.5 秒缓存 + 加锁。"""
    now = time.time()
    if now - _GPU_CACHE["t"] < 1.5:
        return _GPU_CACHE["v"]
    if not _GPU_LOCK.acquire(blocking=False):
        return _GPU_CACHE["v"]
    try:
        _GPU_CACHE["v"] = _gpu_stats_raw() or _GPU_CACHE["v"]
        _GPU_CACHE["t"] = time.time()
        return _GPU_CACHE["v"]
    finally:
        _GPU_LOCK.release()


def _gpu_stats_raw():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        res = []
        for line in out:
            p = [s.strip() for s in line.split(",")]
            res.append({"index": int(p[0]), "name": p[1], "mem_used": int(p[2]), "mem_total": int(p[3]), "util": int(p[4])})
        return res
    except Exception:  # noqa: BLE001
        return []


class TurnLog:
    """一轮对话的事件日志(文本/音频/状态)。与任何网络连接解耦:连接断了,服务端照样生成;
    客户端重连后用游标 after=N 继续取,绝不丢数据 —— 这是「100% 稳定连接」的核心。"""

    def __init__(self, tid: int):
        self.tid = tid
        self.events: list[dict] = []
        self.done = False
        self.ev = asyncio.Event()
        self.created = time.time()

    def push(self, obj: dict):
        self.events.append(obj)
        self.ev.set()

    def finish(self):
        if not self.done:
            self.done = True
            self.ev.set()


class Session:
    def __init__(self, sid: str = ""):
        self.sid = sid
        self.history: list[dict] = []
        self.task: asyncio.Task | None = None
        self.turn = 0
        self.llm_rid = 0
        self.log: TurnLog | None = None
        self.tid_seq = 0
        self.touched = time.time()

    def trim(self):
        total, keep = 0, []
        for m in reversed(self.history):
            total += len(m["content"])
            if total > config.HISTORY_CHARS:
                break
            keep.append(m)
        self.history = list(reversed(keep))
        while self.history and self.history[0]["role"] != "user":
            self.history.pop(0)


# ----------------------------------------------------------------------------
# 引擎
# ----------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.tts: list[Worker] = []
        self.llm: Worker | None = None
        self.aux: Worker | None = None
        self.voice: dict | None = None
        self.job: dict = {"state": "idle"}
        self.gpu_count = 0
        self.rng = np.random.default_rng()
        self.turn_stats: dict = {}
        self.fillers: list[dict] = []
        self._last_filler = -1
        self.sessions: dict[str, Session] = {}

    # ---------- 启动 ----------
    async def start(self):
        self.loop = asyncio.get_running_loop()
        n = 2 if config.MOCK else len(gpu_stats())
        self.gpu_count = n
        if n == 0:
            raise RuntimeError("未检测到 GPU。请在 Kaggle 右侧 Settings → Accelerator 选择 GPU T4 x2。")
        tts_gpus = [0, 1] if n >= 2 else [0]
        for i, g in enumerate(tts_gpus):
            self.tts.append(Worker("tts%d" % i, "tts", g, self.loop))
        # 对话模型按层均分到两张卡("0,1"),避免 GPU0 同时放 TTS+整个 LLM 而 OOM
        self.llm = Worker("llm", "llm", "0,1" if (n >= 2 and config.LLM_SPLIT) else 0, self.loop)
        self.aux = Worker("aux", "aux", 1 if n >= 2 else 0, self.loop)
        for w in self.workers:
            w.start()
            asyncio.create_task(self._load(w))
        asyncio.create_task(self._restore_voice())

    @property
    def workers(self):
        return [*self.tts, *( [self.llm] if self.llm else [] ), *( [self.aux] if self.aux else [] )]

    async def _load(self, w: Worker):
        w.state = "loading"
        w.error = None
        t0 = time.time()
        tries = 1 + max(0, config.TTS_LOAD_RETRIES)
        for attempt in range(tries):
            try:
                w.info = await w.call("load")
                w.state = "ready"
                w.info["load_s"] = round(time.time() - t0, 1)
                print("[engine] worker %s 就绪 (%.0fs) %s" % (w.name, time.time() - t0, w.info), flush=True)
                if w.kind == "tts" and self.voice:
                    asyncio.create_task(self._prep_late(w))
                return
            except Exception as e:  # noqa: BLE001
                w.error = str(e)[-1500:]
                print("[engine] worker %s 加载失败(第 %d/%d 次):\n%s" % (w.name, attempt + 1, tries, e), flush=True)
                if attempt + 1 < tries:
                    w.state = "loading"
                    await asyncio.sleep(4)  # 等其它进程加载完、显存峰值过去后重试
        w.state = "error"

    async def _prep_late(self, w: Worker):
        """某张卡比声音训练晚就绪(或重载成功)时,自动给它补上当前声音,让双卡一起干活。"""
        v = self.voice
        if not v:
            return
        try:
            await w.call("prepare", _timeout=300, voice_id=v["id"], ref_path=v["ref_path"], ref_text=v.get("ref_text", ""),
                         modes=[v["mode"]] if v.get("mode") else None)
            if w.name not in v.setdefault("workers", []):
                v["workers"].append(w.name)
            print("[engine] %s 已补齐当前声音" % w.name, flush=True)
        except Exception as e:  # noqa: BLE001
            print("[engine] %s 补齐声音失败: %s" % (w.name, e), flush=True)

    def error_summary(self, ws=None) -> str:
        out = []
        for w in ws or self.workers:
            if w.state == "error" and w.error:
                out.append("%s: %s" % (w.name, (w.error.strip().splitlines() or ["?"])[-1][:240]))
        return " | ".join(out)

    def reload_failed(self) -> list:
        """重新加载所有失败的模型(界面「重新加载模型」按钮 / 开始训练时自动调用)。"""
        names = []
        for w in self.workers:
            if w.state == "error":
                names.append(w.name)
                if not w.proc.is_alive():
                    w.respawn()
                asyncio.create_task(self._load(w))
        return names

    async def _wait_loaded(self, ws=None, timeout=1200):
        ws = ws or self.workers
        t0 = time.time()
        while any(w.state in ("starting", "loading") for w in ws):
            if time.time() - t0 > timeout:
                raise TimeoutError("模型加载超时")
            await asyncio.sleep(0.5)

    async def _ensure_ready(self):
        """训练前:失败的模型自动重试一次;**只要有 1 张卡的 TTS 就绪就立刻开始训练**。
        旧版会等「两张 TTS + Whisper + 相似度」全部加载完,首次运行要等十几分钟,界面像「点了没反应」。
        现在: 另一张卡/Whisper 还在加载时不等待,它们就绪后自动参与(见 _prep_late / _train 里的 aux_task)。"""
        failed = [w for w in [*self.tts, self.aux] if w and w.state == "error"]
        for w in failed:
            self._log("重新加载 %s …" % w.name)
            if not w.proc.is_alive():
                w.respawn()
            asyncio.create_task(self._load(w))
        t0 = time.time()
        last = 0.0
        while not self.ready_tts():
            if self.tts and all(w.state == "error" for w in self.tts):
                break
            if time.time() - t0 > 1200:
                raise TimeoutError("模型加载超时(>20 分钟),请查看 Kaggle 输出日志")
            if time.time() - last >= 2.0:
                last = time.time()
                st = ", ".join("%s=%s" % (w.name, w.state) for w in [*self.tts, self.aux] if w)
                self._set(stage="等待 TTS 模型加载 %ds (%s)" % (time.time() - t0, st))
            await asyncio.sleep(0.4)
        ws = self.ready_tts()
        if not ws:
            raise RuntimeError("TTS 模型加载失败 — " + (self.error_summary(self.tts) or "未知原因,请查看 Kaggle 输出日志"))
        bad = [w.name for w in self.tts if w.state != "ready"]
        if bad:
            self._log("ℹ %s 仍在加载/不可用,先用已就绪的卡训练,其余就绪后自动加入" % ",".join(bad))
        return ws

    async def _wait_ready(self, ws, timeout=900):
        t0 = time.time()
        while any(w.state in ("starting", "loading") for w in ws):
            if time.time() - t0 > timeout:
                raise TimeoutError("模型加载超时")
            await asyncio.sleep(0.5)
        bad = [w for w in ws if w.state != "ready"]
        if bad:
            raise RuntimeError("模型加载失败: " + ", ".join(w.name for w in bad))

    def ready_tts(self):
        return [w for w in self.tts if w.state == "ready"]

    def status(self):
        return {
            "mock": config.MOCK,
            "gpus": gpu_stats(),
            "workers": {
                w.name: {"gpu": w.gpu, "state": w.state, "inflight": w.inflight, "info": w.info, "error": w.error}
                for w in self.workers
            },
            "voice": self.voice_public(),
            "job": self.job,
            "turn": self.turn_stats,
            "fillers": len(self.fillers),
            "models": {"tts": config.TTS_MODEL, "llm": config.LLM_MODEL, "asr": config.ASR_MODEL},
        }

    def voice_public(self):
        if not self.voice:
            return None
        return {k: v for k, v in self.voice.items() if k not in ("ref_path",)}

    # ---------- 声音 ----------
    async def _restore_voice(self):
        meta = config.VOICE_DIR / "current" / "meta.json"
        if not meta.exists():
            return
        try:
            v = json.loads(meta.read_text("utf-8"))
            v["ref_path"] = str(config.VOICE_DIR / "current" / v.get("ref_file", "ref.wav"))
            await self._wait_loaded(self.tts)
            if not self.ready_tts():
                return
            await asyncio.gather(*[
                w.call("prepare", voice_id=v["id"], ref_path=v["ref_path"], ref_text=v.get("ref_text", ""))
                for w in self.ready_tts()
            ])
            v["workers"] = [w.name for w in self.ready_tts()]
            self.voice = v
            await self._make_fillers()
            print("[engine] 已恢复已训练的声音 %s" % v["id"], flush=True)
        except Exception as e:  # noqa: BLE001
            print("[engine] 恢复声音失败: %s" % e, flush=True)

    def _log(self, msg: str):
        self.job.setdefault("log", []).append("[%s] %s" % (time.strftime("%H:%M:%S"), msg))
        self.job["log"] = self.job["log"][-40:]
        print("[train] " + msg, flush=True)

    def _set(self, **kw):
        if kw.get("stage") and kw.get("stage") != self.job.get("stage"):
            tl = self.job.setdefault("timeline", [])
            tl.append([round(time.time() - self.job.get("started", time.time()), 1), kw["stage"][:80]])
            del tl[:-30]
        self.job.update(kw)

    async def start_training(self, audio: bytes, ref_text: str, lang: str = "zh"):
        if self.job.get("state") == "running" and time.time() - self.job.get("started", 0) < 900:
            raise RuntimeError("训练正在进行中")
        n = max(1, len(self.tts))
        self.job = {
            "state": "running", "stage": "准备中", "progress": 1, "log": [], "warnings": [],
            "gpus": [{"gpu": w.gpu, "progress": 0, "msg": "等待"} for w in self.tts[:n]],
            "similarity": None, "quality": None, "started": time.time(),
        }
        asyncio.create_task(self._train(audio, ref_text.strip(), lang))

    def _gpu_msg(self, i, progress, msg):
        gs = self.job.get("gpus", [])
        if i < len(gs):
            gs[i].update(progress=progress, msg=msg)
        if gs and self.job.get("state") == "running":
            avg = sum(g["progress"] for g in gs) / len(gs)
            self.job["progress"] = max(self.job.get("progress", 0), int(15 + 0.8 * avg))

    async def _get_ref_text(self, ref, ref_text, aux_task, lang):
        if ref_text:
            return ref_text
        if not await aux_task:
            self._log("⚠ 语音识别模型未就绪,本次只用 x-vector 模式(不需要文本)")
            return ""
        self._set(stage="自动识别参考语音文本 (Whisper @ GPU%d)" % self.aux.gpu)
        try:
            r = await self.aux.call("asr", _timeout=120, samples=au.resample(ref, config.SR, 16000), lang=lang)
            self._log("识别文本: %s" % r["text"])
            return r["text"]
        except Exception as e:  # noqa: BLE001
            self._log("⚠ 语音识别失败,本次只用 x-vector 模式: %s" % (str(e).strip().splitlines() or ["?"])[-1])
            return ""

    async def _train(self, audio: bytes, ref_text: str, lang_hint: str = "zh"):
        """加速版流水线: 两张卡从第一秒起就同时干活。
        GPU0: 提取 x-vector 声纹 → 合成校验句 → 相似度;   aux(GPU1): Whisper 识别文本
        GPU1: (等文本) 提取 ICL 上下文 → 合成校验句 → 相似度;   最后把胜出的模式补到另一张卡。
        语气词缓存放到后台,不再占用训练时间。"""
        try:
            self._set(stage="等待模型加载", progress=3)
            tts_ws = await self._ensure_ready()

            async def _aux_wait():  # Whisper/相似度模型:后台等它,绝不阻塞训练起步
                if self.aux is None:
                    return False
                try:
                    await self._wait_loaded([self.aux], timeout=300)
                except Exception:  # noqa: BLE001
                    return False
                if self.aux.state != "ready":
                    self._log("⚠ Whisper/相似度模型未就绪:" + (self.error_summary([self.aux]) or "?"))
                return self.aux.state == "ready"

            aux_task = asyncio.create_task(_aux_wait())
            # 1. 参考音频清洗
            self._set(stage="参考音频清洗(解码/降噪/切静音/归一化)", progress=8)
            x = await asyncio.to_thread(au.decode_audio, audio, config.SR)
            if x.size < config.SR * 1.5:
                raise ValueError("参考音频太短(<1.5s),请至少录制 5 秒以上")
            ref, quality, warns = await asyncio.to_thread(au.preprocess_reference, x, config.SR)
            self._set(quality=quality, warnings=warns)
            self._log("参考音频 %.1fs, 信噪比≈%.0fdB" % (quality["duration"], quality.get("snr_db_raw", 0)))
            vid = uuid.uuid4().hex[:8]
            cur = config.VOICE_DIR / "current"
            if cur.exists():
                shutil.rmtree(cur, ignore_errors=True)
            cur.mkdir(parents=True, exist_ok=True)
            ref_path = cur / ("ref_%s.wav" % vid)  # 每次训练换新文件名,避免 TTS 内部按路径缓存到旧声音
            await asyncio.to_thread(au.write_wav, ref_path, ref, config.SR)
            ref16 = au.resample(ref, config.SR, 16000)
            ref_db = au.to_db(au.rms(ref))
            zh = lang_hint != "en"
            if ref_text:
                zh = detect_language(ref_text) == "Chinese"
            test = "今天天气真不错,我们一起去公园散步吧。" if zh else "The weather is lovely today, let's take a walk in the park."
            lang = "Chinese" if zh else "English"

            # 2. 双卡并行
            self._set(stage="双卡并行:声纹提取 + 文本识别 + 合成校验", progress=15)
            for w_ in tts_ws:
                self._gpu_msg(self.tts.index(w_), 5, "等待")
            w0 = tts_ws[0]
            w1 = tts_ws[1] if len(tts_ws) > 1 else tts_ws[0]
            text_task = asyncio.create_task(self._get_ref_text(ref, ref_text, aux_task, "zh" if zh else "en"))
            prepared: dict[str, set] = {}
            scores: dict[str, float] = {}

            async def eval_mode(w, mode, text):
                i = self.tts.index(w)
                self._gpu_msg(i, 15, "提取声纹(%s)" % mode)
                await w.call("prepare", _timeout=300, voice_id=vid, ref_path=str(ref_path), ref_text=text, modes=[mode])
                prepared.setdefault(w.name, set()).add(mode)
                self._gpu_msg(i, 50, "合成校验句(%s)" % mode)
                wav, sr = await self.synth_collect(w, vid, test, lang, mode)
                wav = await asyncio.to_thread(au.postprocess_chunk, wav, sr, ref_db, True, 0.0)
                aux_ok = await aux_task
                if not aux_ok:
                    self._gpu_msg(i, 90, "%s 就绪" % mode)
                    return mode, None
                s_ = await self.aux.call("sim", _timeout=120, a=ref16, b=au.resample(wav, sr, 16000))
                self._gpu_msg(i, 90, "%s 相似度 %.1f%%" % (mode, s_["cosine"] * 100))
                return mode, s_["cosine"]

            async def path_x():
                return await eval_mode(w0, "xvec", "")

            async def path_i():
                text = await text_task
                if not text:
                    return None
                return await eval_mode(w1, "icl", text)

            res = await asyncio.gather(path_x(), path_i(), return_exceptions=True)
            text = await text_task
            for r in res:
                if isinstance(r, Exception):
                    self._log("模式评估失败: %s" % (str(r).strip().splitlines() or [repr(r)])[-1])
                elif r is not None:
                    self._log("模式 %s 说话人相似度 %s" % (r[0], "%.2f%%" % (r[1] * 100) if r[1] is not None else "N/A"))
                    if r[1] is not None:
                        scores[r[0]] = float(r[1])
            ok_modes = [m for ws_ in prepared.values() for m in ws_]
            if not ok_modes:
                raise RuntimeError("所有克隆模式都失败了:" + "; ".join(str(r) for r in res if isinstance(r, Exception))[-300:])
            best = max(scores, key=scores.get) if scores else ("icl" if "icl" in ok_modes else "xvec")
            sim = scores.get(best)
            # 3. 把胜出的模式补到另一张卡,保证双卡都能合成
            self._set(stage="同步声音到双卡", progress=90)

            async def ensure(w):
                i = self.tts.index(w)
                if best in prepared.get(w.name, set()):
                    self._gpu_msg(i, 100, "完成")
                    return w.name
                try:
                    self._gpu_msg(i, 92, "同步 %s 声纹" % best)
                    await w.call("prepare", _timeout=300, voice_id=vid, ref_path=str(ref_path), ref_text=text, modes=[best])
                    self._gpu_msg(i, 100, "完成")
                    return w.name
                except Exception as e:  # noqa: BLE001
                    self._log("%s 同步失败: %s" % (w.name, (str(e).strip().splitlines() or ["?"])[-1]))
                    return None

            names = [n for n in await asyncio.gather(*[ensure(w) for w in tts_ws]) if n]
            if not names:
                raise RuntimeError("声音同步失败")
            self.voice = {
                "id": vid, "ref_path": str(ref_path), "ref_file": ref_path.name, "ref_text": text, "mode": best,
                "similarity": sim, "scores": scores, "target_rms_db": ref_db,
                "quality": quality, "trained_at": time.time(), "modes": sorted(set(ok_modes)),
                "lang": "Chinese" if zh else "English", "workers": names,
            }
            (cur / "meta.json").write_text(
                json.dumps({k: v for k, v in self.voice.items() if k != "ref_path"}, ensure_ascii=False), "utf-8"
            )
            self.fillers = []
            self._set(state="done", stage="训练完成", progress=100, similarity=sim, mode=best, ref_text=text,
                      elapsed=round(time.time() - self.job.get("started", time.time()), 1))
            self._log("完成(%.1fs),选用模式 %s,相似度 %s" % (self.job["elapsed"], best, "%.2f%%" % (sim * 100) if sim is not None else "N/A"))
            for w_ in self.ready_tts():  # 训练期间才加载完的卡:后台补上当前声音,双卡一起干活
                if w_.name not in names:
                    asyncio.create_task(self._prep_late(w_))
            asyncio.create_task(self._make_fillers())  # 语气词缓存放后台,训练完立刻可用
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
            self._set(state="error", stage="失败", error=msg)
            self._log("失败: " + msg)
            print(e, flush=True)

    # ---------- 语气词缓存 ----------
    async def _make_fillers(self):
        v = self.voice
        fl: list[dict] = []
        texts = config.FILLERS.get((v or {}).get("lang", ""), [])
        if not texts or not self.ready_tts():
            return

        async def one(t):
            w = self._pick_tts()
            wav, sr = await self.synth_collect(w, v["id"], t, v["lang"], v["mode"])
            wav = await asyncio.to_thread(au.postprocess_chunk, wav, sr, v.get("target_rms_db"), True, 0.0, self.rng)
            return {"text": t, "pcm": au.to_pcm16(wav), "sr": sr, "dur": len(wav) / sr}

        res = await asyncio.gather(*[one(t) for t in texts], return_exceptions=True)
        for t, r in zip(texts, res):
            if isinstance(r, dict) and r["dur"] > 0.15:
                fl.append(r)
        if self.voice is v:  # 期间没有被新的训练替换
            self.fillers = fl
        print("[engine] 语气词缓存 %d/%d 就绪" % (len(fl), len(texts)), flush=True)

    def pick_filler(self):
        if not self.fillers:
            return None
        n = len(self.fillers)
        i = np.random.randint(n)
        if n > 1 and i == self._last_filler:
            i = (i + 1) % n
        self._last_filler = i
        return self.fillers[i]

    # ---------- 语音合成 ----------
    def _pick_tts(self) -> Worker:
        ws = self.ready_tts()
        if self.voice and self.voice.get("workers"):
            ws = [w for w in ws if w.name in self.voice["workers"]]
        if not ws:
            raise RuntimeError("TTS 模型尚未就绪")
        return min(ws, key=lambda w: (w.inflight, w.busy_ms))

    def _cancel_tts(self):
        """让所有 TTS 卡立刻放弃排队中/进行中的合成(用户点停止或开始新一轮时)。"""
        for w in self.tts:
            w.cancel.value = max(w.cancel.value, w.next_id - 1)

    async def synth_collect(self, w: Worker, voice_id: str, text: str, lang: str, mode: str):
        """整句合成(训练评估/语气词缓存用):收集所有流式块 → (wav, sr)。"""
        rid, q = w.submit("synth", voice_id=voice_id, text=text, lang=lang, mode=mode, chunk=config.TTS_CHUNK_NEXT)
        parts, sr = [], config.SR
        async for d in w.stream(rid, q):
            parts.append(d["pcm"])
            sr = d["sr"]
        if not parts:
            raise RuntimeError("TTS 没有产出音频")
        return np.concatenate(parts).astype(np.float32), sr

    async def synth_stream(self, text: str, first: bool, out: asyncio.Queue):
        """合成一句话:每个流式块经「带状态后处理」后放入 out;结束放入 None。出错放入 {"error"}。"""
        v = self.voice
        try:
            w = self._pick_tts()
            chunk = config.TTS_CHUNK_FIRST if first else config.TTS_CHUNK_NEXT
            rid, q = w.submit("synth", voice_id=v["id"], text=text, lang=detect_language(text), mode=v["mode"], chunk=chunk)
        except Exception as e:  # noqa: BLE001
            await out.put({"error": str(e)})
            await out.put(None)
            return
        post, t0 = None, time.time()
        try:
            async for d in w.stream(rid, q, stall=60):
                if post is None:
                    post = au.StreamPost(d["sr"], v.get("target_rms_db"), trim_head=True)
                pcm = post.feed(d["pcm"])
                if pcm.size:
                    await out.put({"pcm": au.to_pcm16(pcm), "sr": d["sr"], "dur": pcm.size / d["sr"], "gpu": w.gpu})
            if post is not None:
                tail = post.finish()
                if tail.size:
                    await out.put({"pcm": au.to_pcm16(tail), "sr": post.sr, "dur": tail.size / post.sr, "gpu": w.gpu})
        except asyncio.CancelledError:
            w.cancel.value = max(w.cancel.value, rid)
            raise
        except Exception as e:  # noqa: BLE001
            await out.put({"error": (str(e).strip().splitlines() or [repr(e)])[-1]})
        finally:
            w.busy_ms += (time.time() - t0) * 1000
            await out.put(None)

    # ---------- 对话 ----------
    def new_session(self) -> Session:
        return Session()

    # ---------- 与连接无关的对话接口(HTTP 长轮询) ----------
    def get_session(self, sid: str) -> Session:
        now = time.time()
        for k in [k for k, v in self.sessions.items() if now - v.touched > 6 * 3600 and not (v.task and not v.task.done())]:
            self.sessions.pop(k, None)
        sess = self.sessions.get(sid)
        if sess is None:
            sess = self.sessions[sid] = Session(sid)
        sess.touched = now
        return sess

    async def start_chat(self, sid: str, text: str, opts: dict, speak_only: bool) -> int:
        sess = self.get_session(sid)
        self.cancel_turn(sess)
        if sess.task and not sess.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(sess.task), timeout=0.6)
            except Exception:  # noqa: BLE001
                pass
        sess.tid_seq += 1
        log = TurnLog(sess.tid_seq)
        sess.log = log
        opts = dict(opts or {})
        opts["speak_only"] = speak_only

        def push_json(obj):
            log.push(obj)

        async def emit_json(obj):
            push_json(obj)

        async def emit_audio(turn, idx, sr, pcm):
            log.push({"type": "audio", "turn": turn, "idx": idx, "sr": sr, "b64": base64.b64encode(pcm).decode("ascii")})

        async def guarded():
            try:
                await self.run_turn(sess, text, opts, emit_json, emit_audio)
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001  任何意外都必须让界面看到
                log.push({"type": "error", "text": "服务端异常: " + (str(e).strip().splitlines() or [repr(e)])[-1]})
            finally:
                log.finish()

        sess.task = asyncio.create_task(guarded())
        return log.tid

    async def poll_events(self, sid: str, tid: int, after: int, wait: float = 15.0) -> dict:
        sess = self.get_session(sid)
        log = sess.log
        if log is None or log.tid != tid:
            return {"events": [], "next": after, "done": True, "superseded": True}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, min(wait, 25.0))
        while len(log.events) <= after and not log.done:
            log.ev.clear()
            if len(log.events) > after or log.done:
                break
            rem = deadline - loop.time()
            if rem <= 0:
                break
            try:
                await asyncio.wait_for(log.ev.wait(), rem)
            except asyncio.TimeoutError:
                break
        if len(log.events) > after and not log.done and len(log.events) - after < 3:
            await asyncio.sleep(0.02)  # 合并相邻事件,减少请求数
        evs = log.events[after:after + 200]
        nxt = after + len(evs)
        return {"events": evs, "next": nxt, "done": log.done and nxt >= len(log.events)}

    def stop_chat(self, sid: str):
        sess = self.get_session(sid)
        self.cancel_turn(sess)
        if sess.log and not sess.log.done:
            sess.log.push({"type": "stopped"})
            sess.log.finish()

    def reset_chat(self, sid: str):
        sess = self.get_session(sid)
        self.stop_chat(sid)
        sess.history.clear()

    async def run_turn(self, sess: Session, text: str, opts: dict, emit_json, emit_audio):
        t0 = time.time()
        sess.turn += 1
        turn = sess.turn
        speak_only = opts.get("speak_only", False)
        breath = float(opts.get("breath", 0.4))
        long_mode = bool(opts.get("long", False))
        has_voice = self.voice is not None and bool([w for w in self.ready_tts() if w.name in self.voice.get("workers", [w.name])])
        use_filler = bool(opts.get("filler", True)) and not speak_only
        stats = {"turn": turn, "chunks": 0, "audio_s": 0.0, "ttfa_ms": None, "body_ttfa_ms": None, "llm_first_ms": None, "gpu_chunks": {}, "filler": None}
        filler = self.pick_filler() if (use_filler and has_voice) else None
        base_idx = 1 if filler else 0
        await emit_json({"type": "start", "turn": turn, "voice": has_voice, "filler": bool(filler)})
        if not has_voice:
            await emit_json({"type": "notice", "text": "还没有训练声音,本次只输出文字。请先录制/上传参考语音并点击「开始训练」。"})

        chunker = SentenceChunker()
        order: asyncio.Queue = asyncio.Queue()

        async def emitter():
            idx, n_sent = base_idx, 0
            while True:
                sq = await order.get()
                if sq is None:
                    return
                first_piece = True
                while True:
                    item = await sq.get()
                    if item is None:
                        break
                    if "error" in item:
                        await emit_json({"type": "warn", "text": "语音片段合成失败: " + item["error"][:200]})
                        continue
                    if first_piece and n_sent > 0 and n_sent % 2 == 1 and breath > 0.01:
                        br = au.make_breath(item["sr"], breath, self.rng)
                        if br.size:
                            br = np.concatenate([br, np.zeros(int(item["sr"] * 0.03), dtype=np.float32)])
                            await emit_audio(turn, idx, item["sr"], au.to_pcm16(br))
                            idx += 1
                            stats["chunks"] += 1
                    first_piece = False
                    if stats["body_ttfa_ms"] is None:
                        stats["body_ttfa_ms"] = round((time.time() - t0) * 1000)
                    if stats["ttfa_ms"] is None:
                        stats["ttfa_ms"] = stats["body_ttfa_ms"]
                    await emit_audio(turn, idx, item["sr"], item["pcm"])
                    stats["chunks"] += 1
                    stats["audio_s"] += item["dur"]
                    g = "gpu%d" % item["gpu"]
                    stats["gpu_chunks"][g] = stats["gpu_chunks"].get(g, 0) + 1
                    idx += 1
                n_sent += 1

        em_task = asyncio.create_task(emitter())
        if filler:
            stats["ttfa_ms"] = round((time.time() - t0) * 1000)
            stats["filler"] = filler["text"]
            stats["chunks"] += 1
            stats["audio_s"] += filler["dur"]
            await emit_audio(turn, 0, filler["sr"], filler["pcm"])
            await emit_json({"type": "text", "delta": filler["text"].rstrip(",") + ",", "filler": True})
        synth_tasks: list[asyncio.Task] = []
        n_sub = 0

        def submit(piece: str):
            nonlocal n_sub
            if not has_voice:
                return
            cleaned = clean_for_tts(piece)
            if not is_speakable(cleaned):
                return
            sq: asyncio.Queue = asyncio.Queue()
            synth_tasks.append(asyncio.create_task(self.synth_stream(cleaned, n_sub == 0, sq)))
            n_sub += 1
            order.put_nowait(sq)

        def abort():
            for t in synth_tasks:
                t.cancel()
            em_task.cancel()
            self._cancel_tts()

        try:
            if speak_only:
                for p in _split_all(text):
                    submit(p)
                await emit_json({"type": "text", "delta": text})
            else:
                sys_prompt = config.SYSTEM_PROMPT
                msgs = [{"role": "system", "content": sys_prompt}, *sess.history, {"role": "user", "content": text}]
                sess.history.append({"role": "user", "content": text})
                max_new = config.LLM_LONG_TOKENS if long_mode else config.LLM_DEFAULT_TOKENS
                if filler:
                    msgs[0]["content"] += "\n" + config.FILLER_HINT % filler["text"].rstrip(",")
                if long_mode:
                    msgs[0]["content"] += "\n本次请尽可能详尽、完整、有层次地长篇回答。"
                if self.llm is None or self.llm.state != "ready":
                    detail = (self.llm.error or "") if self.llm else ""
                    raise RuntimeError("对话模型尚未加载完成或加载失败。" + (detail.strip().splitlines()[-1] if detail.strip() else "请稍候或查看 Kaggle 输出日志"))
                rid, q = self.llm.submit("chat", messages=msgs, max_new_tokens=max_new)
                sess.llm_rid = rid
                full = []
                try:
                    async for delta in self.llm.stream(rid, q, stall=config.LLM_FIRST_TOKEN_TIMEOUT + 30):
                        if stats["llm_first_ms"] is None:
                            stats["llm_first_ms"] = round((time.time() - t0) * 1000)
                        full.append(delta)
                        await emit_json({"type": "text", "delta": delta})
                        for p in chunker.feed(delta):
                            submit(p)
                finally:
                    if full:
                        sess.history.append({"role": "assistant", "content": "".join(full)})
                        sess.trim()
                    elif sess.history and sess.history[-1]["role"] == "user":
                        sess.history.pop()  # 没有产出内容 → 不把这条用户消息留在历史里
                for p in chunker.flush():
                    submit(p)
                if not full:
                    raise RuntimeError("对话模型没有返回任何内容")
            order.put_nowait(None)
            await em_task
            stats["total_ms"] = round((time.time() - t0) * 1000)
            stats["audio_s"] = round(stats["audio_s"], 2)
            self.turn_stats = stats
            await emit_json({"type": "done", "turn": turn, "stats": stats})
        except asyncio.CancelledError:
            abort()
            if sess.llm_rid and self.llm:
                self.llm.cancel.value = max(self.llm.cancel.value, sess.llm_rid)
            raise
        except Exception as e:  # noqa: BLE001
            abort()
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
            await emit_json({"type": "error", "text": msg})
            print("[turn] 失败:\n%s" % e, flush=True)

    def cancel_turn(self, sess: Session):
        if sess.llm_rid and self.llm:
            self.llm.cancel.value = max(self.llm.cancel.value, sess.llm_rid)
        self._cancel_tts()
        if sess.task and not sess.task.done():
            sess.task.cancel()

    # ---------- ASR ----------
    async def transcribe(self, data: bytes) -> dict:
        if self.aux is None or self.aux.state != "ready":
            raise RuntimeError("语音识别模型尚未就绪")
        x = await asyncio.to_thread(au.decode_audio, data, 16000)
        if x.size < 1600:
            return {"text": ""}
        return await self.aux.call("asr", samples=x)


def _split_all(text: str):
    c = SentenceChunker()
    out = c.feed(text)
    out += c.flush()
    return out


def pack_audio(turn: int, idx: int, sr: int, pcm: bytes) -> bytes:
    return struct.pack("<IIII", turn, idx, sr, 0) + pcm
