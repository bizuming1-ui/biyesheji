"""流式文本 → 适合 TTS 的短句。

首句尽量短,以换取 3 秒内出首声;后续句子适当变长,让韵律更自然。
"""
import re

from . import config

STRONG = "。！？!?；;\n…"
WEAK = ",，、:：—"

_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0000FE0F\U0000200D]+",
    flags=re.UNICODE,
)
_CODE = re.compile(r"```.*?```", re.S)
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_MD = re.compile(r"[*_#>`~|]+")
_BULLET = re.compile(r"^\s*(?:[-+•]|\d+[.、)])\s+", re.M)
_SPEAKABLE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def clean_for_tts(text: str) -> str:
    text = _CODE.sub("。", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMOJI.sub("", text)
    text = _MD.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "。", text) if "\n" in text else text
    text = re.sub(r"[。]{2,}", "。", text)
    return text.strip()


def is_speakable(text: str) -> bool:
    return bool(_SPEAKABLE.search(text))


def detect_language(text: str) -> str:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    kana = len(re.findall(r"[\u3040-\u30ff]", text))
    hangul = len(re.findall(r"[\uac00-\ud7af]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if kana > 0 and kana >= cjk * 0.3:
        return "Japanese"
    if hangul > 0:
        return "Korean"
    if cjk >= max(1, latin // 3):
        return "Chinese"
    return "English"


class SentenceChunker:
    def __init__(self) -> None:
        self.buf = ""
        self.count = 0  # 已产出的片段数

    def _limits(self):
        if self.count == 0:
            return config.FIRST_MIN_CHARS, config.FIRST_MAX_CHARS
        return config.NEXT_MIN_CHARS, config.NEXT_MAX_CHARS

    def _cut(self, final: bool = False):
        buf = self.buf
        if not buf.strip():
            return None
        min_len, max_len = self._limits()
        first = self.count == 0
        # 1) 强标点
        for i, ch in enumerate(buf):
            if ch in STRONG and i + 1 >= min_len:
                return i + 1
        # 2) 首句允许弱标点,抢首声
        if first:
            for i, ch in enumerate(buf):
                if ch in WEAK and i + 1 >= min_len:
                    return i + 1
        # 3) 超长强制切分
        if len(buf) >= max_len:
            cut = -1
            for i in range(min(len(buf), max_len) - 1, min_len - 1, -1):
                if buf[i] in WEAK or buf[i] == " ":
                    cut = i + 1
                    break
            return cut if cut > 0 else max_len
        if final:
            return len(buf)
        return None

    def feed(self, delta: str):
        self.buf += delta
        out = []
        while True:
            cut = self._cut()
            if not cut:
                break
            piece, self.buf = self.buf[:cut], self.buf[cut:]
            piece = piece.strip()
            if piece:
                self.count += 1
                out.append(piece)
        return out

    def flush(self):
        out = []
        while self.buf.strip():
            cut = self._cut(final=True) or len(self.buf)
            piece, self.buf = self.buf[:cut], self.buf[cut:]
            piece = piece.strip()
            if piece:
                self.count += 1
                out.append(piece)
        self.buf = ""
        return out
