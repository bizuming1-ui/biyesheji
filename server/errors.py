"""数值异常识别 + 精度回退链(v5 训练修复的核心)。

根因:Qwen3-TTS 声音克隆模型在 T4(Turing,无原生 bf16)上用 float16 推理时,logits 会出现 inf/NaN,
采样处 torch.multinomial 触发 CUDA 设备端断言:
    "probability tensor contains either inf, nan or element < 0"  /  "device-side assert triggered"
(见 QwenLM/Qwen3-TTS issue #43)。这个错误发生在「提取声纹 / 合成校验句」里 —— 正好是点击「开始训练」时,
而且 CUDA 设备端断言会让整个 GPU 进程的上下文报废,只能重启该进程。

处理: 默认 bfloat16(faster-qwen3-tts 官方在 Tesla T4 上的基准就是 bf16, RTF≈0.93);
一旦仍出现数值异常 → 引擎自动重启 TTS 进程并换下一档精度(bfloat16 → float32),训练自动重试。
"""

NUMERIC_MARKERS = (
    "tts_numeric",
    "probability tensor",
    "device-side assert",
    "inf, nan",
    "nan or element",
    "illegal memory access",
    "cuda error",
    "cublas_status",
    "acceleratorerror",
)

# 精度回退链:float16 → bfloat16 → float32(最稳、最占显存)
DTYPE_CHAIN = {"float16": "bfloat16", "bfloat16": "float32", "float32": None}


def is_numeric_error(e) -> bool:
    s = str(e).lower()
    return any(m in s for m in NUMERIC_MARKERS)


def next_dtype(cur):
    return DTYPE_CHAIN.get(cur or "", None)


class NumericError(RuntimeError):
    """TTS 数值异常(NaN/inf/CUDA 断言):需要重启进程并降级精度后重试。"""
