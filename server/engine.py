"""引擎:GPU 进程池 + 声音训练流水线 + 对话/语音流水线。"""
import asyncio
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

        ctx = mp.get_context("spawn")
        self.name, self.kind, self.gpu, self.loop = name, kind, gpu, loop
        self.inq, self.outq = ctx.Queue(), ctx.Queue()
        self.cancel = ctx.Value("q", 0)
        self.proc = ctx.Process(target=worker_main, args=(kind, gpu, self.inq, self.outq, self.cancel), daemon=True)
        self.pending: dict[int, asyncio.Queue] = {}
        self.next_id = 1
        self.inflight = 0
        self.state = "starting"
        self.error = None
        self.info: dict = {}
        self.busy_ms = 0.0

    def start(self):
        self.proc.start()
        threading.Thread(target=self._reader, daemon=True, name="reader-" + self.name).start()

    def _reader(self):
        while True:
            try:
                msg = self.outq.get(timeout=1.0)
            except pyqueue.Empty:
                if not self.proc.is_alive():
                    break
                continue
            except (EOFError, OSError):
                break
            try:
                self.loop.call_soon_threadsafe(self._dispatch, msg)
            except RuntimeError:
                break

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

    async def call(self, op: str, **args):
        rid, q = self.submit(op, **args)
        try:
            while True:
                m = await q.get()
                if m["type"] == "error":
                    raise RuntimeError(m["error"])
                if m["type"] == "final":
                    return m["data"]
        finally:
            self._done(rid)

    async def stream(self, rid, q):
        try:
            while True:
                m = await q.get()
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


def gpu_stats():
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


class Session:
    def __init__(self):
        self.history: list[dict] = []
        self.task: asyncio.Task | None = None
        self.turn = 0
        self.llm_rid = 0

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
        self.llm = Worker("llm", "llm", 0, self.loop)
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
        t0 = time.time()
        try:
            w.info = await w.call("load")
            w.state = "ready"
            w.info["load_s"] = round(time.time() - t0, 1)
        except Exception as e:  # noqa: BLE001
            w.state = "error"
            w.error = str(e)[-1500:]
            print("[engine] worker %s 加载失败:\n%s" % (w.name, e), flush=True)

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
            await self._wait_ready(self.ready_tts() or self.tts)
            await asyncio.gather(*[
                w.call("prepare", voice_id=v["id"], ref_path=v["ref_path"], ref_text=v.get("ref_text", ""))
                for w in self.ready_tts()
            ])
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
        self.job.update(kw)

    async def start_training(self, audio: bytes, ref_text: str):
        if self.job.get("state") == "running":
            raise RuntimeError("训练正在进行中")
        n = max(1, len(self.tts))
        self.job = {
            "state": "running", "stage": "准备中", "progress": 1, "log": [], "warnings": [],
            "gpus": [{"gpu": w.gpu, "progress": 0, "msg": "等待"} for w in self.tts[:n]],
            "similarity": None, "quality": None, "started": time.time(),
        }
        asyncio.create_task(self._train(audio, ref_text.strip()))

    def _gpu_msg(self, i, progress, msg):
        if i < len(self.job.get("gpus", [])):
            self.job["gpus"][i].update(progress=progress, msg=msg)

    async def _train(self, audio: bytes, ref_text: str):
        try:
            self._set(stage="等待模型加载", progress=3)
            await self._wait_ready([*self.tts, self.aux])
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
            # 2. 自动转写
            if not ref_text:
                self._set(stage="自动识别参考语音文本 (Whisper @ GPU%d)" % self.aux.gpu, progress=18)
                r = await self.aux.call("asr", samples=au.resample(ref, config.SR, 16000))
                ref_text = r["text"]
                self._log("识别文本: %s" % ref_text)
            # 3. 双卡同时构建声音 prompt
            self._set(stage="双卡并行提取声纹/声学特征", progress=30)
            for i in range(len(self.tts)):
                self._gpu_msg(i, 10, "提取中")

            async def prep(i, w):
                r = await w.call("prepare", voice_id=vid, ref_path=str(ref_path), ref_text=ref_text)
                self._gpu_msg(i, 45, "特征就绪 %s" % "+".join(r["modes"]))
                return r

            results = await asyncio.gather(*[prep(i, w) for i, w in enumerate(self.tts)])
            modes = results[0]["modes"]
            # 4. 双卡并行合成测试句:不同克隆模式分配到不同的卡上同时评估,并择优
            self._set(stage="双卡并行合成校验 & 相似度评估", progress=55)
            zh = detect_language(ref_text or "你好") == "Chinese"
            test = "今天天气真不错,我们一起去公园散步吧。" if zh else "The weather is lovely today, let's take a walk in the park."
            lang = "Chinese" if zh else "English"
            ref16 = au.resample(ref, config.SR, 16000)
            scores: dict[str, float] = {}

            async def one(mi, mode):
                i = mi % len(self.tts)
                w = self.tts[i]
                self._gpu_msg(i, 60, "合成测试句(%s)" % mode)
                wav, sr = await self.synth_collect(w, vid, test, lang, mode)
                wav = au.postprocess_chunk(wav, sr, au.to_db(au.rms(ref)), True, 0.0)
                s_ = await self.aux.call("sim", a=ref16, b=au.resample(wav, sr, 16000))
                self._gpu_msg(i, 90, "%s 相似度 %.1f%%" % (mode, s_["cosine"] * 100))
                return mode, s_["cosine"]

            res = await asyncio.gather(*[one(mi, m) for mi, m in enumerate(modes)], return_exceptions=True)
            for r in res:
                if isinstance(r, Exception):
                    self._log("模式评估失败: %s" % (str(r).strip().splitlines() or [repr(r)])[-1])
                else:
                    scores[r[0]] = float(r[1])
                    self._log("模式 %s 说话人相似度 %.2f%%" % (r[0], r[1] * 100))
            self._set(progress=88)
            if not scores:
                best = modes[0]
            else:
                best = max(scores, key=scores.get)
            sim = scores.get(best)
            # 5. 生效
            self.voice = {
                "id": vid, "ref_path": str(ref_path), "ref_file": ref_path.name, "ref_text": ref_text, "mode": best,
                "similarity": sim, "scores": scores, "target_rms_db": au.to_db(au.rms(ref)),
                "quality": quality, "trained_at": time.time(), "modes": modes,
                "lang": "Chinese" if zh else "English",
            }
            (cur / "meta.json").write_text(
                json.dumps({k: v for k, v in self.voice.items() if k != "ref_path"}, ensure_ascii=False), "utf-8"
            )
            self._set(stage="双卡并行生成「思考语气词」缓存(0 延迟首声)", progress=94)
            await self._make_fillers()
            for i in range(len(self.tts)):
                self._gpu_msg(i, 100, "完成")
            self._set(state="done", stage="训练完成", progress=100, similarity=sim, mode=best, ref_text=ref_text)
            self._log("完成,选用模式 %s,相似度 %s" % (best, "%.2f%%" % (sim * 100) if sim is not None else "N/A"))
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
            self._set(state="error", stage="失败", error=msg)
            self._log("失败: " + msg)
            print(e, flush=True)

    # ---------- 语气词缓存 ----------
    async def _make_fillers(self):
        self.fillers = []
        v = self.voice
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
                self.fillers.append(r)
        print("[engine] 语气词缓存 %d/%d 就绪" % (len(self.fillers), len(texts)), flush=True)

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
            async for d in w.stream(rid, q):
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

    async def run_turn(self, sess: Session, text: str, opts: dict, emit_json, emit_audio):
        t0 = time.time()
        sess.turn += 1
        turn = sess.turn
        speak_only = opts.get("speak_only", False)
        breath = float(opts.get("breath", 0.4))
        long_mode = bool(opts.get("long", False))
        has_voice = self.voice is not None and bool(self.ready_tts())
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
                    async for delta in self.llm.stream(rid, q):
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
