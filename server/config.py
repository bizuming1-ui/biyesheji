"""全局配置。所有项都可以用环境变量覆盖。"""
import os
from pathlib import Path

CODE_VERSION = "v9"  # 浏览器模拟器的「检测日志」会读取它,用来判断 Kaggle 上跑的是不是最新代码

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
# v9: 对话大模型直接部署在 Kaggle GPU 上(Qwen3-4B-Instruct-2507,原生 262K 上下文),不再依赖任何外部 API / Key。
LLM_DEFAULT_TOKENS = int(os.environ.get("LLM_DEFAULT_TOKENS", "6144"))     # 普通模式一次回答最多生成的 token(超出自动续写,播放无感)
LLM_LONG_TOKENS = int(os.environ.get("LLM_LONG_TOKENS", "20000"))          # 「长文」模式的总 token 预算
LLM_ROUND_TOKENS = int(os.environ.get("LLM_ROUND_TOKENS", "2048"))         # 每一轮 generate 的上限;撞到上限且没有自然结束 → 自动续写下一轮,直到预算用完
LLM_PACE_TPS = float(os.environ.get("LLM_PACE_TPS", "12"))                 # 首批之后把生成速度限制在每秒多少 token:语音只消耗约 3~6 token/s,多生成只会和 TTS 抢 GPU;0=不限速
LLM_PACE_BURST = int(os.environ.get("LLM_PACE_BURST", "48"))               # 每轮对话开头不限速的 token 数(保证首句、首声最快)
LLM_CONT_MARK = "\x00CONT"                                                 # 续写信号(worker→引擎,不会进入文字流)
LLM_LOAD_WAIT = float(os.environ.get("LLM_LOAD_WAIT", "900"))              # 对话模型还在加载时,一次对话最多等多少秒(而不是直接报错)
HISTORY_CHARS = int(os.environ.get("HISTORY_CHARS", "24000"))
LLM_FIRST_TOKEN_TIMEOUT = float(os.environ.get("LLM_FIRST_TOKEN_TIMEOUT", "45"))  # 秒,超时必给出错误,绝不静默卡死
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
LLM_MAX_MEM_GIB = float(os.environ.get("LLM_MAX_MEM_GIB", "5.0"))
TTS_LOAD_RETRIES = int(os.environ.get("TTS_LOAD_RETRIES", "2"))

# 采样参数:温度越低越稳(音色更一致、杂音更少),越高越有情绪起伏。0.75 是实测的折中。
TTS_TEMPERATURE = float(os.environ.get("TTS_TEMPERATURE", "0.65"))
TTS_TOP_K = int(os.environ.get("TTS_TOP_K", "40"))
TTS_TOP_P = float(os.environ.get("TTS_TOP_P", "0.95"))

TTS_DTYPE = os.environ.get("TTS_DTYPE", "bfloat16")  # v5: T4 上 float16 会让 Qwen3-TTS 采样出 NaN(训练失败的根因);bfloat16 是官方 T4 基准用的精度,出现数值异常会自动升到 float32
LLM_DTYPE = os.environ.get("LLM_DTYPE", "float16")

# 预合成并缓存的「思考语气词」(用克隆声音生成,每轮对话 0 延迟先播,正文在后台同步合成)
FILLERS = {
    "Chinese": ["嗯,", "好的,", "让我想想,", "哦,", "嗯嗯,"],
    "English": ["Hmm,", "Well,", "Okay,", "Sure,", "Let me think,"],
}
FILLER_HINT = "(系统已经替你先说了一个简短的语气词「%s」,你的正文请直接接着回答,不要再用语气词开头。)"

# ---- 训练流水线(v5) ----
TRAIN_AUX_WAIT = float(os.environ.get("TRAIN_AUX_WAIT", "30"))      # 合成完后最多再等 Whisper/相似度模型多少秒(不再无限等)
TRAIN_SYNTH_STALL = float(os.environ.get("TRAIN_SYNTH_STALL", "240"))  # 训练里首次合成含声纹提取/图捕获,放宽无响应超时
TRAIN_CHUNK = int(os.environ.get("TRAIN_CHUNK", "16"))              # 训练校验句用更大的流式块 → 更少的解码调用 → 更快
TRAIN_MAX_SECONDS = float(os.environ.get("TRAIN_MAX_SECONDS", "900"))  # 超过这个时间的训练任务视为卡死,可被新训练取代
TRAIN_ASR_WAIT = float(os.environ.get("TRAIN_ASR_WAIT", "90"))      # 自动识别参考文本最多等 Whisper 就绪多少秒


# ---- v8: 参考音频人声分离(歌曲/伴奏) ----
REF_SEPARATE = os.environ.get("REF_SEPARATE", "auto").lower()          # auto: 信噪比低或用户勾选「歌曲」时分离; 1: 总是; 0: 关闭
REF_SEPARATE_SNR = float(os.environ.get("REF_SEPARATE_SNR", "30"))      # auto 模式:信噪比低于此值(dB)就分离
REF_SEPARATE_DEVICE = os.environ.get("REF_SEPARATE_DEVICE", "cuda")     # cuda / cpu
REF_SEPARATE_GPU = os.environ.get("REF_SEPARATE_GPU", "1")              # 用哪张卡跑分离(GPU1 上只有 Whisper,比较空)
REF_SEPARATE_TIMEOUT = float(os.environ.get("REF_SEPARATE_TIMEOUT", "120"))

# ---- v8: 对话流式通道(单个 HTTP 响应推送二进制帧,无 base64、无轮询间隙) ----
STREAM_HEARTBEAT = float(os.environ.get("STREAM_HEARTBEAT", "4"))
