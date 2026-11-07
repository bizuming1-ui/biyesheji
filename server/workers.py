"""GPU 工作进程。每个进程独占一张卡(CUDA_VISIBLE_DEVICES),互不抢 GIL,从而真正实现双卡并行。

进程划分 (Kaggle T4 x2) —— 全部常驻显存,服务不退出就不会卸载:
  GPU0: tts0 (Qwen3-TTS-1.7B, CUDA Graph 流式)  +  llm (Qwen3-4B-Instruct-2507, 长输出)
  GPU1: tts1 (Qwen3-TTS-1.7B, CUDA Graph 流式)  +  aux (Whisper ASR + WavLM 说话人相似度)

协议: inq 收到 {"id","op","args"};outq 回复 {"id","type": delta|final|error, "data"}。
  - chat : delta 为文本片段
  - synth: delta 为 {"pcm": float32 ndarray, "sr": int} 音频块(流式)
"""
import os
import sys
import threading
import time
import traceback

import numpy as np

from . import config
from .errors import is_numeric_error, next_dtype


# ----------------------------------------------------------------------------
# Mock 实现 (CLONE_MOCK=1) —— 不依赖 torch,用于验证整条链路
# ----------------------------------------------------------------------------
def _fake_voice(text: str, sr: int, seed: int = 0) -> np.ndarray:
    dur = min(12.0, 0.35 + 0.17 * len(text))
    t = np.arange(int(sr * dur)) / sr
    f0 = 140 + 25 * np.sin(2 * np.pi * 0.7 * t + seed)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = sum(np.sin(k * phase) / k for k in range(1, 6))
    env = 0.5 * (1 + np.sin(2 * np.pi * 3.2 * t)) ** 0.7
    x = 0.25 * sig * env
    return x.astype(np.float32)


class MockTTS:
    def __init__(self, gpu):
        self.gpu, self.voices = gpu, {}
        self.dtype = config.TTS_DTYPE

    def _numeric_check(self):
        """测试用:MOCK_NUMERIC_DTYPE=float16,bfloat16 → 以这些精度运行时模拟「probability tensor contains inf/nan」崩溃,
        用来验证引擎会自动重启进程、降级精度、重试训练。"""
        bad = [x for x in os.environ.get("MOCK_NUMERIC_DTYPE", "").split(",") if x]
        if self.dtype in bad:
            raise RuntimeError("TTS_NUMERIC(%s): probability tensor contains either inf, nan or element < 0" % self.dtype)

    def load(self, args, emit):
        time.sleep(0.4)
        if args and args.get("dtype"):
            self.dtype = args["dtype"]
        if os.environ.get("MOCK_FAIL_TTS") == str(self.gpu):  # 测试用:模拟某张卡 TTS 加载失败
            raise RuntimeError("模拟的 TTS 加载失败(MOCK_FAIL_TTS=%d)" % self.gpu)
        return {"device": "mock:%d" % self.gpu, "engine": "mock-stream", "dtype": self.dtype}

    def prepare(self, args, emit):
        time.sleep(0.3)
        self._numeric_check()
        self.voices[args["voice_id"]] = True
        want = args.get("modes") or ["icl", "xvec"]
        modes = [m for m in want if m != "icl" or (args.get("ref_text") or "").strip()]
        return {"modes": modes, "dtype": self.dtype}

    def synth(self, args, emit, cancelled=lambda: False):
        self._numeric_check()
        text = args["text"]
        seed = 0 if args.get("mode") == "icl" else 1
        wav = _fake_voice(text, config.SR, seed)
        step = int(config.SR * 0.33)
        time.sleep(0.12)
        total = 0
        for i in range(0, len(wav), step):
            if cancelled():
                break
            time.sleep(float(os.environ.get("MOCK_CHUNK_SLEEP", "0.05")))  # 模拟合成速度(默认 RTF≈0.15;MOCK_CHUNK_SLEEP=0.3 → 单卡 RTF≈0.9)
            emit({"pcm": wav[i : i + step], "sr": config.SR})
            total += min(step, len(wav) - i)
        return {"samples": total, "sr": config.SR}


class MockLLM:
    def load(self, args, emit):
        time.sleep(0.3)
        return {"device": "mock"}

    def chat(self, args, emit, cancelled):
        last = args["messages"][-1]["content"]
        reply = (
            "好的,我听到了你说的:%s。这是一段用来测试的模拟回答,它会被切成很多短句。"
            "第一句很短,所以首声会很快出现。后面的句子会稍微长一点,双卡会交替合成,"
            "播放的时候不应该出现任何停顿。我们再多说几句,用来验证长文本输出能力,"
            "看看流式分句、乱序到达和顺序播放是不是都正常。最后,祝你使用愉快。" % last[:30]
        )
        if "__LLM_FAIL__" in last:
            raise RuntimeError("模拟的 LLM 故障(用于验证错误必定回显,不会静默卡死)")
        for i in range(0, len(reply), 2):
            if cancelled():
                break
            time.sleep(0.03)
            emit(reply[i : i + 2])
        return {"text": reply}


class MockAux:
    def load(self, args, emit):
        time.sleep(0.2)
        return {"device": "mock", "asr": True, "sv": True}

    def asr(self, args, emit):
        return {"text": "这是模拟的语音识别结果", "language": "zh"}

    def sim(self, args, emit):
        a, b = args["a"], args["b"]
        return {"cosine": 0.93 if len(a) and len(b) else 0.0}


# ----------------------------------------------------------------------------
# 真实实现  (每个进程只看得到一张卡 → 一律使用 cuda:0)
# ----------------------------------------------------------------------------
def _from_pretrained(cls, name, dtype, **kw):
    """兼容 transformers 4.x(torch_dtype) 与 5.x(dtype)。"""
    try:
        return cls.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(name, torch_dtype=dtype, **kw)


def free_cuda():
    """加载失败/回退前必须释放半加载的模型,否则显存泄漏会让第二次加载 OOM。"""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _last_line(e) -> str:
    return (str(e).strip().splitlines() or [repr(e)])[-1][:300]


class RealTTS:
    def __init__(self, gpu):
        self.gpu, self.model, self.fast = gpu, None, None
        self.dtype_name = config.TTS_DTYPE
        self.preps: dict[str, dict] = {}
        self.engine_name = "?"

    # ---- 加载 ----
    def _load_model(self, dtype_name):
        import torch

        dt = getattr(torch, dtype_name)
        self.fast = None
        fast_err = None
        try:
            from faster_qwen3_tts import FasterQwen3TTS

            self.fast = FasterQwen3TTS.from_pretrained(
                config.TTS_MODEL, device="cuda", dtype=dt, attn_implementation="sdpa", max_seq_len=1536
            )
            self.model = self.fast.model
            self.engine_name = "faster-qwen3-tts(CUDA Graph 流式)"
            try:
                self.fast.warmup(prefill_len=120)
            except Exception as e:  # noqa: BLE001
                print("[tts gpu%d] CUDA Graph 预热失败(首次合成时再捕获): %s" % (self.gpu, e), flush=True)
        except Exception as e:  # noqa: BLE001
            fast_err = e
            self.fast = self.model = None
            free_cuda()  # 关键:先释放半加载的模型,再尝试回退,否则会 OOM
            print("[tts gpu%d] faster-qwen3-tts 加载失败: %s" % (self.gpu, _last_line(e)), flush=True)
        if self.fast is None:
            if fast_err is not None and "out of memory" in str(fast_err).lower():
                raise RuntimeError("TTS 显存不足(GPU%d): %s" % (self.gpu, _last_line(fast_err)))
            try:
                from qwen_tts import Qwen3TTSModel

                self.model = _from_pretrained(Qwen3TTSModel, config.TTS_MODEL, dt, device_map="cuda", attn_implementation="sdpa")
                self.engine_name = "qwen-tts(官方,非流式)"
            except Exception as e2:  # noqa: BLE001
                free_cuda()
                raise RuntimeError(
                    "TTS 模型加载失败。faster-qwen3-tts: %s | qwen-tts: %s。"
                    "请确认依赖: transformers==5.15.1 + qwen-tts-hf(不要同时装 qwen-tts)" % (_last_line(fast_err), _last_line(e2))
                )
        self.dtype_name = dtype_name

    def load(self, args, emit):
        want = (args or {}).get("dtype") or self.dtype_name
        self._load_model(want)
        return {"device": "cuda(GPU%d)" % self.gpu, "dtype": self.dtype_name, "engine": self.engine_name}

    def _reload(self, dtype_name):
        """数值溢出(输出 NaN/全零)时,换更稳的精度重新加载,并重建所有声音(进程上下文仍健康时才可用)。"""
        import gc

        import torch

        self.model = self.fast = None
        self.__dict__.pop("_prompt_cache", None)
        keep = dict(self.preps)
        self.preps = {}
        gc.collect()
        torch.cuda.empty_cache()
        self._load_model(dtype_name)
        for a in list(keep.values()):
            self.prepare(dict(a, warm=False), None)

    # ---- 声音 prompt ----
    def prepare(self, args, emit):
        """构建可复用的声音克隆 prompt(ICL / x-vector)。
        warm=False(训练时): 只提取并缓存声纹,不再多合成一句「嗯。」(校验句本身就会预热) → 训练更快。
        warm=True(恢复声音/补卡时): 真实合成一小句,让 CUDA Graph 与解码器完全热起来。"""
        vid = args["voice_id"]
        old = self.preps.get(vid, {})
        text = (args.get("ref_text") or old.get("ref_text") or "").strip()
        self.preps[vid] = {**old, **{k: v for k, v in args.items() if k != "warm"}, "ref_text": text}
        want = args.get("modes") or (["icl"] if text else []) + ["xvec"]
        modes = [m for m in want if m != "icl" or text]
        lang = "Chinese" if any("\u4e00" <= c <= "\u9fff" for c in text) or not text else "English"
        warm = bool(args.get("warm", True))
        try:
            for m in modes:
                if not warm and self._prime(vid, m):
                    continue
                for _ in self._gen(vid, "嗯。" if lang == "Chinese" else "Hmm.", lang, m, chunk=8, max_new=36):
                    pass
        except Exception as e:  # noqa: BLE001
            if is_numeric_error(e):
                raise RuntimeError("TTS_NUMERIC(%s): %s" % (self.dtype_name, _last_line(e)))
            raise
        return {"modes": modes, "dtype": self.dtype_name}

    def _prime(self, vid, mode) -> bool:
        """只构建并缓存声纹 prompt(不生成音频)。成功返回 True;任何不确定的情况返回 False → 走真实合成预热。"""
        if self.fast is None:
            return False
        try:
            a = self.preps[vid]
            ref_text = (a.get("ref_text") or "").strip()
            if mode == "icl" and not ref_text:
                mode = "xvec"
            self.fast._resolve_voice_clone_prompt_from_reference(
                input_ids=[None], ref_audio=a["ref_path"], ref_text=ref_text, xvec_only=(mode == "xvec"), append_silence=True,
            )
            return True
        except Exception as e:  # noqa: BLE001
            if is_numeric_error(e):
                raise
            print("[tts gpu%d] 仅提取声纹失败,改用合成预热: %s" % (self.gpu, _last_line(e)), flush=True)
            return False

    # ---- 合成 ----
    def _gen(self, vid, text, lang, mode, chunk, cancelled=lambda: False, max_new=None):
        a = self.preps[vid]
        ref, ref_text = a["ref_path"], (a.get("ref_text") or "").strip()
        if mode == "icl" and not ref_text:
            mode = "xvec"
        max_new = int(max_new or min(1200, max(72, len(text) * 6 + 48)))
        kw = dict(temperature=config.TTS_TEMPERATURE, top_k=config.TTS_TOP_K, top_p=config.TTS_TOP_P, repetition_penalty=1.05)
        if self.fast is not None:
            for audio, sr, _timing in self.fast.generate_voice_clone_streaming(
                text=text, language=lang, ref_audio=ref, ref_text=ref_text,
                xvec_only=(mode == "xvec"), chunk_size=chunk, max_new_tokens=max_new, **kw,
            ):
                if cancelled():
                    return
                yield np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)
        else:
            key = (vid, mode)
            cache = self.__dict__.setdefault("_prompt_cache", {})
            if key not in cache:
                cache[key] = (
                    self.model.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text, x_vector_only_mode=False)
                    if mode == "icl"
                    else self.model.create_voice_clone_prompt(ref_audio=ref, ref_text=None, x_vector_only_mode=True)
                )
            wavs, sr = self.model.generate_voice_clone(
                text=text, language=lang, voice_clone_prompt=cache[key], max_new_tokens=max_new
            )
            w = wavs[0]
            if hasattr(w, "detach"):
                w = w.detach().float().cpu().numpy()
            yield np.asarray(w, dtype=np.float32).reshape(-1), int(sr)

    def synth(self, args, emit, cancelled=lambda: False):
        text, vid = args["text"], args["voice_id"]
        lang, mode = args.get("lang", "Chinese"), args.get("mode", "icl")
        chunk = int(args.get("chunk", config.TTS_CHUNK_NEXT))
        for attempt in range(3):
            total, sr_out, bad = 0, config.SR, False
            try:
                for w, sr in self._gen(vid, text, lang, mode, chunk, cancelled):
                    if not np.all(np.isfinite(w)):
                        bad = True
                        break
                    if total == 0 and w.size and float(np.max(np.abs(w))) < 1e-5 and w.size > sr * 0.2:
                        bad = True
                        break
                    total += w.size
                    sr_out = sr
                    emit({"pcm": w, "sr": sr})
            except Exception as e:  # noqa: BLE001
                # CUDA 设备端断言(probability tensor contains inf/nan)会让本进程的 CUDA 上下文报废,
                # 这里不能原地恢复 → 抛出带 TTS_NUMERIC 标记的错误,由引擎重启进程并降级精度。
                if is_numeric_error(e):
                    raise RuntimeError("TTS_NUMERIC(%s): %s" % (self.dtype_name, _last_line(e)))
                raise
            if not bad:
                return {"samples": total, "sr": sr_out}
            nxt = next_dtype(self.dtype_name)
            if nxt is None or attempt == 2:
                raise RuntimeError("TTS_NUMERIC(%s): 输出异常(NaN/静音)" % self.dtype_name)
            print("[tts gpu%d] %s 输出异常,自动回退 %s" % (self.gpu, self.dtype_name, nxt), flush=True)
            self._reload(nxt)
        raise RuntimeError("unreachable")


class RealLLM:
    def __init__(self, gpu):
        self.gpu, self.model, self.tok = gpu, None, None
        self.dtype_name = config.LLM_DTYPE

    def load(self, args, emit):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        n = torch.cuda.device_count()
        if config.LLM_SPLIT and n >= 2:
            # 两张卡各放一半层(顺序执行,速度≈单卡,但每卡显存占用减半)
            mm = {i: "%.1fGiB" % config.LLM_MAX_MEM_GIB for i in range(n)}
            self.model = _from_pretrained(
                AutoModelForCausalLM, config.LLM_MODEL, getattr(torch, self.dtype_name),
                device_map="auto", max_memory=mm,
            ).eval()
            where = "GPU0+GPU1(各一半)"
        else:
            self.model = _from_pretrained(AutoModelForCausalLM, config.LLM_MODEL, getattr(torch, self.dtype_name)).to("cuda").eval()
            where = "GPU0"
        self.dev = "cuda:0"
        self._warm()
        return {"device": where, "dtype": self.dtype_name}

    def _warm(self):
        import torch

        ids = self.tok(["你好"], return_tensors="pt").to(self.dev)
        with torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=6, do_sample=False)
        txt = self.tok.decode(out[0], skip_special_tokens=True)
        if self.dtype_name == "float16" and ("!!!!" in txt or not txt.strip()):
            print("[llm] fp16 预热输出异常(%r),自动换 bfloat16" % txt, flush=True)
            self._to_bf16()

    def _to_bf16(self):
        import torch

        self.model = self.model.to(torch.bfloat16)
        self.dtype_name = "bfloat16"

    def _run_once(self, kw, streamer, cancelled, emit):
        import queue

        err = {}

        def run():
            import torch

            try:
                with torch.inference_mode():
                    self.model.generate(**kw)
            except BaseException as e:  # noqa: BLE001  必须捕获,否则 streamer 永远等不到结束信号 → 界面「没反应」
                err["e"] = e
                err["tb"] = traceback.format_exc()
                try:
                    streamer.end()
                except Exception:  # noqa: BLE001
                    streamer.text_queue.put(streamer.stop_signal)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        pieces, head, held, released = [], "", [], False
        try:
            for piece in streamer:
                if not piece:
                    continue
                pieces.append(piece)
                if not released:
                    # 前 4 个字先扣住:fp16 溢出时模型会吐 "!!!!",不能让它出现在界面/语音里
                    head += piece
                    held.append(piece)
                    if len(head) < 4:
                        continue
                    if self.dtype_name == "float16" and head.startswith("!!!!"):
                        kw["stopping_criteria"][0].force = True
                        break
                    released = True
                    for h in held:
                        emit(h)
                    held = []
                    continue
                emit(piece)
            if held and not (self.dtype_name == "float16" and head.startswith("!!!!")):
                for h in held:
                    emit(h)
        except queue.Empty:
            kw["stopping_criteria"][0].force = True
            raise RuntimeError("LLM 超过 %d 秒没有输出(可能显存不足或模型卡死)" % int(config.LLM_STALL_TIMEOUT))
        th.join(timeout=5)
        return pieces, err, head

    def chat(self, args, emit, cancelled):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        msgs = args["messages"]
        try:
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok([prompt], return_tensors="pt").to(self.dev)

        class Cancel(StoppingCriteria):
            force = False

            def __call__(self, input_ids, scores, **kw):
                return torch.full((input_ids.shape[0],), bool(cancelled() or self.force), dtype=torch.bool, device=input_ids.device)

        for attempt in (0, 1):
            streamer = TextIteratorStreamer(
                self.tok, skip_prompt=True, skip_special_tokens=True, timeout=config.LLM_FIRST_TOKEN_TIMEOUT
            )
            kw = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=int(args.get("max_new_tokens", config.LLM_DEFAULT_TOKENS)),
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
                stopping_criteria=StoppingCriteriaList([Cancel()]),
            )
            pieces, err, head = self._run_once(kw, streamer, cancelled, emit)
            garbage = self.dtype_name == "float16" and head.startswith("!!!!")
            if (err or garbage) and not pieces_ok(pieces, garbage) and attempt == 0 and self.dtype_name == "float16":
                print("[llm] fp16 生成失败/输出异常,换 bfloat16 重试: %s" % (err.get("tb") or head), flush=True)
                self._to_bf16()
                continue
            if err:
                raise RuntimeError("LLM 生成失败: %s" % (str(err["e"]).strip().splitlines() or [repr(err["e"])])[-1])
            return {"text": "".join(pieces)}
        raise RuntimeError("LLM 生成失败")


def pieces_ok(pieces, garbage):
    """已经流出的正常文字不应因重试而重复:只有「还没输出任何有效内容」时才允许重试。"""
    if garbage:
        return False
    return bool("".join(pieces).strip())


class RealAux:
    def __init__(self, gpu):
        self.gpu = gpu
        self.asr_pipe = self.sv = self.sv_fe = None
        self.errors = {}

    def load(self, args, emit):
        import torch

        self.dev = "cuda"
        try:
            from transformers import pipeline

            try:
                self.asr_pipe = pipeline("automatic-speech-recognition", model=config.ASR_MODEL, dtype=torch.float16, device=self.dev)
            except TypeError:
                self.asr_pipe = pipeline("automatic-speech-recognition", model=config.ASR_MODEL, torch_dtype=torch.float16, device=self.dev)
        except Exception as e:  # noqa: BLE001
            self.errors["asr"] = repr(e)
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector

            self.sv_fe = AutoFeatureExtractor.from_pretrained(config.SV_MODEL)
            self.sv = WavLMForXVector.from_pretrained(config.SV_MODEL).to(self.dev).eval()
        except Exception as e:  # noqa: BLE001
            self.errors["sv"] = repr(e)
        try:  # 预热:第一次真实使用时不再卡 cuDNN/内核编译
            z = np.zeros(16000, dtype=np.float32) + 1e-4 * np.random.randn(16000).astype(np.float32)
            if self.asr_pipe is not None:
                self.asr_pipe({"raw": z, "sampling_rate": 16000})
            if self.sv is not None:
                self._embed(z)
        except Exception as e:  # noqa: BLE001
            print("[aux] 预热失败(不影响使用): %s" % _last_line(e), flush=True)
        return {"device": "cuda(GPU%d)" % self.gpu, "asr": self.asr_pipe is not None, "sv": self.sv is not None, "errors": self.errors}

    def asr(self, args, emit):
        if self.asr_pipe is None:
            raise RuntimeError("ASR 未加载: %s" % self.errors.get("asr"))
        x = np.asarray(args["samples"], dtype=np.float32)  # 16k
        kw = {}
        if x.size > 16000 * 29:  # 短音频不分块,更快、更准
            kw.update(chunk_length_s=30, return_timestamps=False)
        lang = {"zh": "chinese", "en": "english"}.get(str(args.get("lang") or ""), None)
        if lang:
            kw["generate_kwargs"] = {"language": lang, "task": "transcribe"}
        try:
            out = self.asr_pipe({"raw": x, "sampling_rate": 16000}, **kw)
        except Exception:  # noqa: BLE001  语言参数在个别版本不被接受 → 退回自动检测
            out = self.asr_pipe({"raw": x, "sampling_rate": 16000}, chunk_length_s=30)
        return {"text": (out.get("text") or "").strip()}

    def _embed(self, x16):
        import torch

        inp = self.sv_fe(x16, sampling_rate=16000, return_tensors="pt", padding=True)
        inp = {k: v.to(self.dev) for k, v in inp.items()}
        with torch.inference_mode():
            e = self.sv(**inp).embeddings
        e = torch.nn.functional.normalize(e.float(), dim=-1)
        return e[0]

    def sim(self, args, emit):
        import torch

        if self.sv is None:
            raise RuntimeError("说话人模型未加载: %s" % self.errors.get("sv"))
        ea = self._embed(np.asarray(args["a"], dtype=np.float32))
        eb = self._embed(np.asarray(args["b"], dtype=np.float32))
        return {"cosine": float(torch.dot(ea, eb).item())}


# ----------------------------------------------------------------------------
# 进程入口
# ----------------------------------------------------------------------------
def _make(kind: str, gpu: int):
    if config.MOCK:
        return {"tts": lambda: MockTTS(gpu), "llm": MockLLM, "aux": MockAux}[kind]()
    return {"tts": lambda: RealTTS(gpu), "llm": lambda: RealLLM(gpu), "aux": lambda: RealAux(gpu)}[kind]()


def worker_main(kind: str, gpu: int, inq, outq, cancel_val) -> None:
    # 必须在 import torch 之前:让本进程只看到自己的那张卡 → 双卡真并行、互不干扰
    if not config.MOCK:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    handler = None
    while True:
        msg = inq.get()
        if msg is None:
            break
        rid, op, args = msg["id"], msg["op"], msg.get("args", {})

        def emit(piece, _rid=rid):
            outq.put({"id": _rid, "type": "delta", "data": piece})

        def cancelled(_rid=rid):
            return cancel_val.value >= _rid

        try:
            t0 = time.time()
            if handler is None:
                handler = _make(kind, gpu)
            if op in ("chat", "synth") and cancelled():
                data = {"cancelled": True}
            elif op == "chat":
                data = handler.chat(args, emit, cancelled)
            elif op == "synth":
                data = handler.synth(args, emit, cancelled)
            else:
                data = getattr(handler, op)(args, emit)
            if isinstance(data, dict):
                data["_ms"] = round((time.time() - t0) * 1000, 1)
            outq.put({"id": rid, "type": "final", "data": data})
        except Exception:  # noqa: BLE001
            outq.put({"id": rid, "type": "error", "error": traceback.format_exc()})
            if op == "load":
                handler = None
                if not config.MOCK:
                    free_cuda()
        sys.stdout.flush()
