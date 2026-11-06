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

# ---- 模型 ----
TTS_MODEL = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-large-v3-turbo")
SV_MODEL = os.environ.get("SV_MODEL", "microsoft/wavlm-base-plus-sv")
DET_MODEL = os.environ.get("DET_MODEL", "google/owlvit-base-patch32")

SR = 24000  # Qwen3-TTS 输出采样率
LLM_DEFAULT_TOKENS = int(os.environ.get("LLM_DEFAULT_TOKENS", "1024"))
LLM_LONG_TOKENS = int(os.environ.get("LLM_LONG_TOKENS", "4096"))
HISTORY_CHARS = int(os.environ.get("HISTORY_CHARS", "7000"))

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

TTS_DTYPE = os.environ.get("TTS_DTYPE", "float16")  # T4 无原生 bf16;若输出异常会自动回退 bfloat16

# 预合成并缓存的「思考语气词」(用克隆声音生成,每轮对话 0 延迟先播,正文在后台同步合成)
FILLERS = {
    "Chinese": ["嗯,", "好的,", "让我想想,", "哦,", "嗯嗯,"],
    "English": ["Hmm,", "Well,", "Okay,", "Sure,", "Let me think,"],
}
FILLER_HINT = "(系统已经替你先说了一个简短的语气词「%s」,你的正文请直接接着回答,不要再用语气词开头。)"

# 开放词汇检测的默认词表 (英文, 中文)
DETECT_LABELS = [
    ("person", "人"), ("face", "脸"), ("hand", "手"), ("cat", "猫"), ("dog", "狗"),
    ("bird", "鸟"), ("horse", "马"), ("cow", "牛"), ("sheep", "羊"), ("fish", "鱼"),
    ("car", "汽车"), ("bus", "公交车"), ("truck", "卡车"), ("bicycle", "自行车"),
    ("motorcycle", "摩托车"), ("airplane", "飞机"), ("boat", "船"), ("train", "火车"),
    ("traffic light", "红绿灯"), ("chair", "椅子"), ("sofa", "沙发"), ("bed", "床"),
    ("table", "桌子"), ("desk", "书桌"), ("door", "门"), ("window", "窗户"),
    ("laptop", "笔记本电脑"), ("computer monitor", "显示器"), ("keyboard", "键盘"),
    ("computer mouse", "鼠标"), ("cell phone", "手机"), ("tablet", "平板"),
    ("television", "电视"), ("remote control", "遥控器"), ("headphones", "耳机"),
    ("camera", "相机"), ("clock", "时钟"), ("lamp", "台灯"), ("book", "书"),
    ("notebook", "笔记本"), ("pen", "笔"), ("pencil", "铅笔"), ("scissors", "剪刀"),
    ("backpack", "背包"), ("handbag", "手提包"), ("umbrella", "雨伞"), ("wallet", "钱包"),
    ("glasses", "眼镜"), ("watch", "手表"), ("hat", "帽子"), ("shoe", "鞋"),
    ("bottle", "瓶子"), ("cup", "杯子"), ("mug", "马克杯"), ("bowl", "碗"),
    ("plate", "盘子"), ("fork", "叉子"), ("knife", "刀"), ("spoon", "勺子"),
    ("apple", "苹果"), ("banana", "香蕉"), ("orange", "橙子"), ("pizza", "披萨"),
    ("sandwich", "三明治"), ("cake", "蛋糕"), ("potted plant", "盆栽"), ("flower", "花"),
    ("tree", "树"), ("vase", "花瓶"), ("pillow", "枕头"), ("blanket", "毯子"),
    ("toy", "玩具"), ("teddy bear", "泰迪熊"), ("ball", "球"), ("guitar", "吉他"),
    ("microphone", "麦克风"), ("speaker", "音箱"), ("refrigerator", "冰箱"),
    ("microwave", "微波炉"), ("oven", "烤箱"), ("sink", "水槽"), ("toilet", "马桶"),
    ("mirror", "镜子"), ("painting", "画"), ("bag", "袋子"), ("box", "盒子"),
    ("key", "钥匙"), ("coin", "硬币"), ("charger", "充电器"), ("cable", "数据线"),
    ("water bottle", "水壶"), ("tissue box", "纸巾盒"), ("trash can", "垃圾桶"),
]
ZH2EN = {zh: en for en, zh in DETECT_LABELS}
EN2ZH = {en: zh for en, zh in DETECT_LABELS}
