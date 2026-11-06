"""GPU 工作进程。每个进程独占一张卡,互不抢 GIL,从而真正实现双卡并行。

进程划分 (Kaggle T4 x2):
  GPU0: tts0 (Qwen3-TTS-1.7B)  +  llm (Qwen3-4B-Instruct-2507, 长输出)
  GPU1: tts1 (Qwen3-TTS-1.7B)  +  aux (Whisper ASR + WavLM 说话人相似度 + OWL-ViT 开放词汇检测)

协议: inq 收到 {"id","op","args"};outq 回复 {"id","type": delta|final|error, "data"}。
"""
import io
import os
import sys
import threading
import time
import traceback

import numpy as np

from . import config


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
    fade = int(sr * 0.02)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return x.astype(np.float32)


class MockTTS:
    def __init__(self, gpu):
        self.gpu, self.voices = gpu, {}

    def load(self, args, emit):
        time.sleep(0.4)
        return {"device": "mock:%d" % self.gpu}

    def prepare(self, args, emit):
        time.sleep(0.3)
        self.voices[args["voice_id"]] = True
        modes = ["icl", "xvec"] if (args.get("ref_text") or "").strip() else ["xvec"]
        return {"modes": modes}

    def synth(self, args, emit):
        text = args["text"]
        time.sleep(0.12 + 0.012 * len(text))
        seed = 0 if args.get("mode") == "icl" else 1
        return {"wav": _fake_voice(text, config.SR, seed), "sr": config.SR}


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
        for i in range(0, len(reply), 2):
            if cancelled():
                break
            time.sleep(0.03)
            emit(reply[i : i + 2])
        return {"text": reply}


class MockAux:
    def load(self, args, emit):
        time.sleep(0.2)
        return {"device": "mock", "asr": True, "sv": True, "det": True}

    def asr(self, args, emit):
        return {"text": "这是模拟的语音识别结果", "language": "zh"}

    def sim(self, args, emit):
        a, b = args["a"], args["b"]
        return {"cosine": 0.93 if len(a) and len(b) else 0.0}

    def detect(self, args, emit):
        return {"boxes": [{"label": "person", "score": 0.91, "box": [0.2, 0.2, 0.6, 0.9]}], "ms": 5.0}


# ----------------------------------------------------------------------------
# 真实实现
# ----------------------------------------------------------------------------
class RealTTS:
    def __init__(self, gpu):
        self.gpu, self.model, self.prompts = gpu, None, {}
        self.dtype_name = config.TTS_DTYPE
        self.prep_args = {}

    def _load_model(self, dtype_name):
        import torch
        from qwen_tts import Qwen3TTSModel

        dev = "cuda:%d" % self.gpu
        dt = getattr(torch, dtype_name)
        # T4 不支持 flash-attn2:使用 sdpa
        try:
            self.model = Qwen3TTSModel.from_pretrained(config.TTS_MODEL, device_map=dev, dtype=dt, attn_implementation="sdpa")
        except TypeError:
            self.model = Qwen3TTSModel.from_pretrained(config.TTS_MODEL, device_map=dev, torch_dtype=dt, attn_implementation="sdpa")
        self.dtype_name = dtype_name

    def load(self, args, emit):
        self._load_model(self.dtype_name)
        return {"device": "cuda:%d" % self.gpu, "dtype": self.dtype_name}

    def _fallback_bf16(self):
        """fp16 数值溢出(输出 NaN/全零)时,换 bf16 重新加载并重建所有声音 prompt。"""
        import gc

        import torch

        self.model = None
        self.prompts = {}
        gc.collect()
        torch.cuda.empty_cache()
        self._load_model("bfloat16")
        for a in list(self.prep_args.values()):
            self.prepare(a, None)

    def prepare(self, args, emit):
        """构建可复用的声音克隆 prompt(ICL 与 x-vector 两种模式都构建,后续择优)。"""
        vid, path, text = args["voice_id"], args["ref_path"], args.get("ref_text") or ""
        self.prep_args[vid] = args
        entry = {}
        if text.strip():
            entry["icl"] = self.model.create_voice_clone_prompt(
                ref_audio=path, ref_text=text, x_vector_only_mode=False
            )
        entry["xvec"] = self.model.create_voice_clone_prompt(
            ref_audio=path, ref_text=None, x_vector_only_mode=True
        )
        self.prompts[vid] = entry
        return {"modes": list(entry.keys())}

    def synth(self, args, emit):
        text = args["text"]
        max_new = int(min(1800, max(96, len(text) * 9 + 64)))
        for attempt in (0, 1):
            entry = self.prompts[args["voice_id"]]
            mode = args.get("mode", "icl")
            prompt = entry.get(mode) or next(iter(entry.values()))
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=args.get("lang", "Auto"),
                voice_clone_prompt=prompt,
                max_new_tokens=max_new,
            )
            w = wavs[0]
            if hasattr(w, "detach"):
                w = w.detach().float().cpu().numpy()
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            bad = (not np.all(np.isfinite(w))) or float(np.max(np.abs(w)) if w.size else 0.0) < 1e-4
            if not bad or attempt == 1 or self.dtype_name != "float16":
                if bad:
                    raise RuntimeError("TTS 输出异常(NaN/静音),请把环境变量 TTS_DTYPE 设为 bfloat16 再试")
                return {"wav": w, "sr": int(sr)}
            print("[tts gpu%d] fp16 输出异常,自动回退 bfloat16" % self.gpu, flush=True)
            self._fallback_bf16()
        raise RuntimeError("unreachable")


class RealLLM:
    def __init__(self, gpu):
        self.gpu, self.model, self.tok = gpu, None, None

    def load(self, args, emit):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = "cuda:%d" % self.gpu
        self.tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, torch_dtype=torch.float16).to(dev)
        self.model.eval()
        self.dev = dev
        # 预热,避免第一句话慢
        ids = self.tok(["你好"], return_tensors="pt").to(dev)
        with torch.no_grad():
            self.model.generate(**ids, max_new_tokens=4, do_sample=False)
        return {"device": dev}

    def chat(self, args, emit, cancelled):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        msgs = args["messages"]
        try:
            prompt = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok([prompt], return_tensors="pt").to(self.dev)

        class Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return torch.full((input_ids.shape[0],), bool(cancelled()), dtype=torch.bool, device=input_ids.device)

        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
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

        def run():
            with torch.no_grad():
                self.model.generate(**kw)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        full = []
        for piece in streamer:
            if piece:
                full.append(piece)
                emit(piece)
        th.join()
        return {"text": "".join(full)}


class RealAux:
    def __init__(self, gpu):
        self.gpu = gpu
        self.asr_pipe = self.sv = self.sv_fe = self.det = self.det_proc = None
        self.errors = {}

    def load(self, args, emit):
        import torch

        dev = "cuda:%d" % self.gpu
        self.dev = dev
        try:
            from transformers import pipeline

            self.asr_pipe = pipeline(
                "automatic-speech-recognition", model=config.ASR_MODEL, torch_dtype=torch.float16, device=dev
            )
        except Exception as e:  # noqa: BLE001
            self.errors["asr"] = repr(e)
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector

            self.sv_fe = AutoFeatureExtractor.from_pretrained(config.SV_MODEL)
            self.sv = WavLMForXVector.from_pretrained(config.SV_MODEL).to(dev).eval()
        except Exception as e:  # noqa: BLE001
            self.errors["sv"] = repr(e)
        try:
            from transformers import OwlViTForObjectDetection, OwlViTProcessor

            self.det_proc = OwlViTProcessor.from_pretrained(config.DET_MODEL)
            self.det = OwlViTForObjectDetection.from_pretrained(config.DET_MODEL, torch_dtype=torch.float16).to(dev).eval()
        except Exception as e:  # noqa: BLE001
            self.errors["det"] = repr(e)
        return {
            "device": dev,
            "asr": self.asr_pipe is not None,
            "sv": self.sv is not None,
            "det": self.det is not None,
            "errors": self.errors,
        }

    def asr(self, args, emit):
        if self.asr_pipe is None:
            raise RuntimeError("ASR 未加载: %s" % self.errors.get("asr"))
        x = np.asarray(args["samples"], dtype=np.float32)  # 16k
        out = self.asr_pipe({"raw": x, "sampling_rate": 16000}, chunk_length_s=30, return_timestamps=False)
        return {"text": (out.get("text") or "").strip()}

    def _embed(self, x16):
        import torch

        inp = self.sv_fe(x16, sampling_rate=16000, return_tensors="pt", padding=True)
        inp = {k: v.to(self.dev) for k, v in inp.items()}
        with torch.no_grad():
            e = self.sv(**inp).embeddings
        e = torch.nn.functional.normalize(e, dim=-1)
        return e[0]

    def sim(self, args, emit):
        import torch

        if self.sv is None:
            raise RuntimeError("说话人模型未加载: %s" % self.errors.get("sv"))
        ea = self._embed(np.asarray(args["a"], dtype=np.float32))
        eb = self._embed(np.asarray(args["b"], dtype=np.float32))
        return {"cosine": float(torch.dot(ea, eb).item())}

    def detect(self, args, emit):
        import torch
        from PIL import Image

        if self.det is None:
            raise RuntimeError("检测模型未加载: %s" % self.errors.get("det"))
        t0 = time.time()
        img = Image.open(io.BytesIO(args["jpeg"])).convert("RGB")
        w, h = img.size
        labels = args["labels"]
        queries = ["a photo of a %s" % l for l in labels]
        inputs = self.det_proc(text=[queries], images=img, return_tensors="pt")
        inputs = {k: (v.to(self.dev).half() if v.dtype == torch.float32 else v.to(self.dev)) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.det(**inputs)
        ts = torch.tensor([[h, w]], device=self.dev)
        res = self.det_proc.post_process_object_detection(out, threshold=float(args.get("thr", 0.15)), target_sizes=ts)[0]
        boxes = []
        for s, l, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            boxes.append({
                "label": labels[l],
                "score": round(s, 3),
                "box": [max(0.0, b[0] / w), max(0.0, b[1] / h), min(1.0, b[2] / w), min(1.0, b[3] / h)],
            })
        boxes = _nms(boxes)
        return {"boxes": boxes, "ms": round((time.time() - t0) * 1000, 1)}


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(boxes, thr=0.5):
    boxes = sorted(boxes, key=lambda d: -d["score"])
    keep = []
    for b in boxes:
        if all(_iou(b["box"], k["box"]) < thr for k in keep):
            keep.append(b)
    return keep[:30]


# ----------------------------------------------------------------------------
# 进程入口
# ----------------------------------------------------------------------------
def _make(kind: str, gpu: int):
    if config.MOCK:
        return {"tts": lambda: MockTTS(gpu), "llm": MockLLM, "aux": MockAux}[kind]()
    return {"tts": lambda: RealTTS(gpu), "llm": lambda: RealLLM(gpu), "aux": lambda: RealAux(gpu)}[kind]()


def worker_main(kind: str, gpu: int, inq, outq, cancel_val) -> None:
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
            if op == "chat":
                data = handler.chat(args, emit, cancelled)
            else:
                data = getattr(handler, op)(args, emit)
            if isinstance(data, dict):
                data["_ms"] = round((time.time() - t0) * 1000, 1)
            outq.put({"id": rid, "type": "final", "data": data})
        except Exception:  # noqa: BLE001
            outq.put({"id": rid, "type": "error", "error": traceback.format_exc()})
            if op == "load":
                handler = None
        sys.stdout.flush()
