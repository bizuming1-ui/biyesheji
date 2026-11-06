"""全局配置。所有项都可以用环境变量覆盖。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CLONE_DATA", ROOT / "data"))
VOICE_DIR = DATA_DIR / "voices"
WEB_DIR = ROOT / "web"
VOICE_DIR.mkdir(parents=True, exist_ok=True)

# CLONE_MOCK=1 时不加载任何真实模型,用于在没有 GPU 的机器上验证整条链路。
MOCK = os.environ.get("CLONE_MOCK", "0") == "1"
PORT = int(os.environ.get("PORT", "7860"))

# ---- 模型(全部常驻显存:服务进程不退出,模型就不会卸载) ----
TTS_MODEL = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-large-v3-turbo")
SV_MODEL = os.environ.get("SV_MODEL", "microsoft/wavlm-base-plus-sv")

SR = 24000  # Qwen3-TTS 输出采样率
LLM_DEFAULT_TOKENS = int(os.environ.get("LLM_DEFAULT_TOKENS", "1024"))
LLM_LONG_TOKENS = int(os.environ.get("LLM_LONG_TOKENS", "4096"))
HISTORY_CHARS = int(os.environ.get("HISTORY_CHARS", "7000"))
LLM_FIRST_TOKEN_TIMEOUT = float(os.environ.get("LLM_FIRST_TOKEN_TIMEOUT", "60"))  # 秒,超时必给出错误,绝不静默卡死
LLM_STALL_TIMEOUT = float(os.environ.get("LLM_STALL_TIMEOUT", "45"))

# ---- 流式 TTS ----
# faster-qwen3-tts: CUDA Graph + 流式输出。chunk=4 步 ≈ 0.33s 音频,首块越小首声越快。
TTS_STREAM = os.environ.get("TTS_STREAM", "1") == "1"
TTS_CHUNK_FIRST = int(os.environ.get("TTS_CHUNK_FIRST", "4"))
TTS_CHUNK_NEXT = int(os.environ.get("TTS_CHUNK_NEXT", "8"))

# ---- 分句策略 (决定首声延迟) ----
FIRST_MIN_CHARS = int(os.environ.get("FIRST_MIN_CHARS", "3"))
FIRST_MAX_CHARS = int(os.environ.get("FIRST_MAX_CHARS", "14"))
NEXT_MIN_CHARS = int(os.environ.get("NEXT_MIN_CHARS", "14"))
NEXT_MAX_CHARS = int(os.environ.get("NEXT_MAX_CHARS", "48"))

SYSTEM_PROMPT = (
    "你是一个温暖、自然、有情绪的中文语音助手,你的回答会被直接转成语音朗读。"
    "请使用口语化的纯文本回答:不要使用 Markdown、列表符号、表情符号、代码块和网址。"
    "句子要自然分段,多用逗号和句号,适当使用语气词(嗯、啊、呢、吧)表达情绪。"
    "每次回答的第一个分句要非常短(不超过 8 个字),然后再自然展开。"
    "如果用户使用其他语言,就用对应语言回答。"
    "当用户要求长篇内容(故事、文章、讲解)时,请完整、连贯、有条理地充分展开,不要中途停止。"
)

# 对话大模型(Qwen3-4B fp16 ≈ 7.5GiB)按层均分到两张卡,每卡只占约 3.8GiB,
# 这样每张 T4(≈14.5GiB)都能同时放下:TTS(≈4.5GiB) + LLM 一半 + 其它,不再 OOM。
LLM_SPLIT = os.environ.get("LLM_SPLIT", "1") == "1"
LLM_MAX_MEM_GIB = float(os.environ.get("LLM_MAX_MEM_GIB", "4.6"))
TTS_LOAD_RETRIES = int(os.environ.get("TTS_LOAD_RETRIES", "2"))

# 采样参数:温度越低越稳(音色更一致、杂音更少),越高越有情绪起伏。0.75 是实测的折中。
TTS_TEMPERATURE = float(os.environ.get("TTS_TEMPERATURE", "0.75"))
TTS_TOP_K = int(os.environ.get("TTS_TOP_K", "40"))
TTS_TOP_P = float(os.environ.get("TTS_TOP_P", "0.95"))

TTS_DTYPE = os.environ.get("TTS_DTYPE", "float16")  # T4 无原生 bf16;若输出异常会自动回退 bfloat16
LLM_DTYPE = os.environ.get("LLM_DTYPE", "float16")

# 预合成并缓存的「思考语气词」(用克隆声音生成,每轮对话 0 延迟先播,正文在后台同步合成)
FILLERS = {
    "Chinese": ["嗯,", "好的,", "让我想想,", "哦,", "嗯嗯,"],
    "English": ["Hmm,", "Well,", "Okay,", "Sure,", "Let me think,"],
}
FILLER_HINT = "(系统已经替你先说了一个简短的语气词「%s」,你的正文请直接接着回答,不要再用语气词开头。)"
