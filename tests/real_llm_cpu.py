"""用真实的小模型(Qwen2.5-0.5B-Instruct, CPU)验证 RealLLM.chat 的「长文本自动续写 + 限速 + 取消」逻辑:
   python tests/real_llm_cpu.py      (需要 torch + transformers;首次会下载约 1GB 模型,仅供开发自检,Kaggle 上不会运行)"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["LLM_ROUND_TOKENS"] = "24"
os.environ["LLM_PACE_TPS"] = "0"
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from server import config  # noqa: E402
from server.workers import RealLLM  # noqa: E402

NAME = os.environ.get("TEST_LLM", "Qwen/Qwen2.5-0.5B-Instruct")
fails = []


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" -> " + str(extra)[:160] if extra else ""))
    if not cond:
        fails.append(name)


llm = RealLLM(0)
llm.tok = AutoTokenizer.from_pretrained(NAME)
llm.model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
llm.dev, llm.dtype_name = "cpu", "float32"
msgs = [{"role": "user", "content": "请用中文写一个很长很长的故事,至少写五百字。"}]

# 1) 预算 80、每轮 24 → 至少 3 轮续写,文字连续,标记不进文字
out = []
r = llm.chat({"messages": msgs, "max_new_tokens": 80}, out.append, lambda: False)
text = "".join(x for x in out if not x.startswith(config.LLM_CONT_MARK))
conts = [x for x in out if x.startswith(config.LLM_CONT_MARK)]
check("multi-round continuation happened", r["rounds"] >= 3 and len(conts) == r["rounds"] - 1, (r["rounds"], conts))
check("budget respected (<=80+rounds slack)", r["tokens"] <= 80, r["tokens"])
check("text is the concatenation of all rounds", text == r["text"] and len(text) > 40, text[:80])
check("no marker leak", "\x00" not in text)
print("   生成文本:", text[:100].replace("\n", " "))

# 2) 自然结束 → 不续写
out = []
r = llm.chat({"messages": [{"role": "user", "content": "只回答一个字:好"}], "max_new_tokens": 200}, out.append, lambda: False)
check("natural stop => single round", r["rounds"] == 1, r)

# 3) 取消 → 立刻停止,不续写
out, n = [], [0]
def cancelled():
    n[0] += 1
    return n[0] > 30
t0 = time.time()
r = llm.chat({"messages": msgs, "max_new_tokens": 400}, out.append, cancelled)
check("cancel stops generation & continuation", r["rounds"] <= 2 and r["tokens"] < 100, r)

# 4) 限速
config.LLM_PACE_TPS = 20.0
config.LLM_PACE_BURST = 5
out = []
t0 = time.time()
r = llm.chat({"messages": msgs, "max_new_tokens": 45}, out.append, lambda: False)
dt = time.time() - t0
check("pacing: ~40 paced tokens take >= 1.5s", dt >= 1.5, "%.2fs for %d tokens" % (dt, r["tokens"]))
print("\n结果:", "全部通过" if not fails else "失败 %s" % fails)
sys.exit(1 if fails else 0)
