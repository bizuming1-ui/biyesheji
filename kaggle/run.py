"""Kaggle 一键启动器 (Accelerator 请选 GPU T4 x2,Internet 打开)。

流程: 检查双卡 → 装依赖 → 并行预下载模型 → 启动服务 → 建立公网隧道 → 在输出框打印可点击网址。
用法(Kaggle 单元格):  !python /kaggle/working/clone-system/kaggle/run.py
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("PORT", "7860"))
MARK = Path("/tmp/.clone_deps_ok_v7")


def log(*a):
    print("[kaggle-run]", *a, flush=True)


def sh(cmd, check=True):
    log("$", cmd)
    return subprocess.run(cmd, shell=True, check=check)


def load_secrets():
    try:
        from kaggle_secrets import UserSecretsClient

        c = UserSecretsClient()
        for name in ("HF_TOKEN", "NGROK_TOKEN", "NGROK_AUTHTOKEN"):
            try:
                os.environ[name] = c.get_secret(name)
                log("已读取 Kaggle Secret:", name)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


def check_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
    except FileNotFoundError:
        out = []
    log("检测到 GPU 数量:", len(out))
    for l in out:
        log("  ", l)
    if len(out) < 2:
        log("⚠ 少于 2 张 GPU!请在 Settings → Accelerator 选择 'GPU T4 x2'。将降级为单卡运行(速度会变慢)。")
    return len(out)


def _torch_constraints():
    """锁定 Kaggle 自带的 torch 系列版本,防止 pip 升级 torch 后与 GPU 驱动不匹配。"""
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    keep = [l for l in out if re.match(r"^(torch|torchvision|torchaudio)==", l, re.I)]
    f = Path("/tmp/torch_constraints.txt")
    f.write_text("\n".join(keep) + "\n")
    log("锁定:", ", ".join(keep) or "(无)")
    return f


SMOKE = r"""
import importlib.metadata as m, sys
import transformers
print("transformers", transformers.__version__, "| qwen-tts-hf", m.version("qwen-tts-hf") if True else "")
from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as mod
from qwen_tts.core.tokenizer_12hz import configuration_qwen3_tts_tokenizer_v2 as cfgm
cfg = cfgm.Qwen3TTSTokenizerV2Config()
mod.Qwen3TTSTokenizerV2Encoder._from_config(cfg.encoder_config)   # 日志里 upsample/rope_theta 报错就出在这一步
import faster_qwen3_tts
print("SMOKE_OK")
"""


def smoke_test():
    """真实构造一次 Qwen3-TTS 的编码器(CPU,几秒):版本不兼容(upsample / rope_theta)会在这里暴露,而不是等到加载 GPU 模型才失败。"""
    p = subprocess.run([sys.executable, "-c", SMOKE], capture_output=True, text=True)
    ok = "SMOKE_OK" in p.stdout
    tail = (p.stderr.strip().splitlines() or p.stdout.strip().splitlines() or ["?"])[-1]
    return ok, tail


def install_deps():
    base = "fastapi 'uvicorn[standard]' python-multipart soundfile scipy noisereduce accelerate huggingface_hub"
    c = _torch_constraints()
    if MARK.exists():
        log("依赖已安装,跳过")
    else:
        # ★ 根因修复:官方 qwen-tts 与 transformers 5.x 不兼容(报错 property 'upsample' ... has no setter),
        #   而 transformers>5.15.1 又会报 MimiEncoderConfig 没有 rope_theta。
        #   正确组合 = faster-qwen3-tts + qwen-tts-hf + transformers==5.15.1,且不能同时装 qwen-tts。
        sh("pip uninstall -y -q qwen-tts", check=False)
        r = sh("pip install -q -c %s faster-qwen3-tts 'qwen-tts-hf>=0.1.1.post1' 'transformers==5.15.1' %s" % (c, base), check=False)
        if r.returncode != 0:
            log("⚠ 依赖安装返回非 0,继续做兼容性自检")
        MARK.write_text("ok")
    ok, msg = smoke_test()
    log(("✓ 兼容性自检通过: " if ok else "✗ 兼容性自检失败: ") + msg)
    if not ok:
        log("尝试强制对齐版本 …")
        sh("pip uninstall -y -q qwen-tts", check=False)
        sh("pip install -q -c %s --force-reinstall --no-deps 'qwen-tts-hf>=0.1.1.post1' 'transformers==5.15.1'" % c, check=False)
        sh("pip install -q -c %s 'huggingface_hub>=1.5,<2' faster-qwen3-tts" % c, check=False)
        ok, msg = smoke_test()
        log(("✓ 第二次自检通过: " if ok else "✗ 第二次自检仍失败: ") + msg)
    if not ok:
        log("⚠ 回退方案:官方 qwen-tts + transformers 4.57.3(整句合成,首声会慢,但能用)")
        sh("pip uninstall -y -q faster-qwen3-tts qwen-tts-hf", check=False)
        sh("pip install -q -c %s qwen-tts 'transformers==4.57.3' 'accelerate==1.12.0' %s" % (c, base), check=False)
    chk = subprocess.run([sys.executable, "-c", "import faster_qwen3_tts"], capture_output=True, text=True)
    if chk.returncode == 0:
        log("✓ faster-qwen3-tts 可用(CUDA Graph 流式)")
    else:
        log("faster-qwen3-tts 不可用,将使用官方 qwen-tts:", (chk.stderr.strip().splitlines() or ["?"])[-1])
    # v8: 人声分离(参考音频是歌曲/带伴奏时用)。独立于 MARK,失败不影响主流程;-c 约束文件保证不会升级/降级 torch。
    if subprocess.run([sys.executable, "-c", "import demucs"], capture_output=True).returncode != 0:
        log("安装 demucs(参考音频人声分离,约 1 分钟)…")
        sh("pip install -q -c %s demucs" % c, check=False)
    if subprocess.run([sys.executable, "-c", "import demucs"], capture_output=True).returncode == 0:
        log("✓ demucs 可用(歌曲参考音频将自动分离人声)")
    else:
        log("⚠ demucs 不可用:歌曲参考音频将不做人声分离(不影响其它功能)")
    if not shutil.which("sox"):  # 只是消除 'SoX could not be found' 警告,失败不影响使用
        sh("(apt-get install -y -qq sox libsox-fmt-all > /dev/null 2>&1) || true", check=False)
    if not shutil.which("ffmpeg"):
        log("未找到 ffmpeg,尝试安装(浏览器录音 webm 需要它)")
        sh("apt-get install -y -qq ffmpeg > /dev/null 2>&1", check=False)
    if not shutil.which("ffmpeg"):
        sh("pip install -q imageio-ffmpeg", check=False)
        try:
            import imageio_ffmpeg

            d = Path("/tmp/ffbin")
            d.mkdir(exist_ok=True)
            link = d / "ffmpeg"
            if not link.exists():
                link.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]
        except Exception as e:  # noqa: BLE001
            log("ffmpeg 安装失败:", e)
    log("ffmpeg:", shutil.which("ffmpeg"))


def predownload():
    sys.path.insert(0, str(ROOT))
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import snapshot_download

    from server import config

    repos = [config.TTS_MODEL, config.ASR_MODEL, config.SV_MODEL] + ([] if config.USE_DEEPSEEK else [config.LLM_MODEL])  # v7: 对话走 DeepSeek API,不再下载本地 LLM(省 ~8GB)

    def dl(r):
        t = time.time()
        try:
            snapshot_download(r, token=os.environ.get("HF_TOKEN") or None)
            log("✓ 模型就绪 %s (%.0fs)" % (r, time.time() - t))
        except Exception as e:  # noqa: BLE001
            log("✗ 下载失败 %s: %s" % (r, e))

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(dl, repos))


def pump(proc, prefix):
    for line in iter(proc.stdout.readline, ""):
        print(prefix + line.rstrip(), flush=True)


def kill_stale():
    """重复运行单元格时,上一次遗留的服务/隧道进程还占着端口和显存,必须先清掉。"""
    sh("pkill -f 'uvicorn server.app' || true", check=False)
    sh("pkill -f cloudflared || true", check=False)
    sh("(fuser -k %d/tcp > /dev/null 2>&1) || true" % PORT, check=False)
    time.sleep(3)
    try:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True
        ).stdout.strip().splitlines()
        for l in used:
            idx, mb = [x.strip() for x in l.split(",")]
            if int(mb) > 1500:
                log("⚠ GPU%s 已被占用 %s MiB(可能是上一次运行残留)。如果后面出现显存不足,请 Kaggle 菜单 Run → Restart session 后重跑。" % (idx, mb))
    except Exception:  # noqa: BLE001
        pass


def start_server():
    env = dict(os.environ, PYTHONUNBUFFERED="1", PORT=str(PORT))
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", str(PORT), "--no-access-log", "--timeout-keep-alive", "75", "--ws-ping-interval", "20", "--ws-ping-timeout", "120"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    threading.Thread(target=pump, args=(p, "[server] "), daemon=True).start()
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/api/health" % PORT, timeout=2)
            log("服务已启动 (端口 %d)" % PORT)
            return p
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise RuntimeError("服务启动超时")


_CF = {"proc": None}


def start_cloudflared():
    exe = Path("/tmp/cloudflared")
    if not exe.exists():
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        log("下载 cloudflared …")
        urllib.request.urlretrieve(url, exe)
        exe.chmod(0o755)
    for attempt in range(3):
        old = _CF.get("proc")
        if old and old.poll() is None:
            old.kill()
        # --protocol http2: Kaggle 对 QUIC(UDP) 不稳定,http2(TCP) 更稳;--edge-ip-version 4 避免 IPv6 黑洞
        p = subprocess.Popen(
            [str(exe), "tunnel", "--url", "http://127.0.0.1:%d" % PORT, "--no-autoupdate",
             "--protocol", "http2", "--edge-ip-version", "4", "--retries", "8"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        _CF["proc"] = p
        found = {}

        def rd(p=p, found=found):
            for line in iter(p.stdout.readline, ""):
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if m and "url" not in found:
                    found["url"] = m.group(0)

        threading.Thread(target=rd, daemon=True).start()
        for _ in range(45):
            if "url" in found:
                return found["url"]
            if p.poll() is not None:
                break
            time.sleep(1)
        log("cloudflared 第 %d 次没拿到网址,重试 …" % (attempt + 1))
    return None


def start_localhost_run():
    """第三备选:localhost.run(只需 ssh,无需账号)。"""
    if not shutil.which("ssh"):
        return None
    p = subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=20",
         "-R", "80:127.0.0.1:%d" % PORT, "nokey@localhost.run"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    found = {}

    def rd():
        for line in iter(p.stdout.readline, ""):
            m = re.search(r"https://[a-z0-9]+\.lhr\.life", line)
            if m and "url" not in found:
                found["url"] = m.group(0)

    threading.Thread(target=rd, daemon=True).start()
    for _ in range(30):
        if "url" in found:
            return found["url"]
        time.sleep(1)
    return None


def tunnel_ok(url, timeout=12):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_ngrok():
    tok = os.environ.get("NGROK_TOKEN") or os.environ.get("NGROK_AUTHTOKEN")
    if not tok:
        return None
    sh("pip install -q pyngrok", check=False)
    from pyngrok import ngrok

    ngrok.set_auth_token(tok)
    return ngrok.connect(PORT, "http").public_url


def banner(url):
    line = "=" * 74
    print("\n" + line)
    print("  ✅ 克隆语音系统已就绪,请打开下面的网址(粘贴到「浏览器模拟器」地址栏即可):")
    print("\n      " + url + "\n")
    print("  🔎 检测日志(完整参数):" + url + "/api/diag   (浏览器模拟器右上角「检测日志」会自动读取)")
    print(line + "\n", flush=True)
    try:
        from IPython.display import HTML, display

        display(HTML('<h2>🎙 <a href="%s" target="_blank" style="color:#7c5cff">打开克隆语音系统</a></h2><p>%s</p>' % (url, url)))
    except Exception:  # noqa: BLE001
        pass


def open_tunnel():
    for fn in (start_cloudflared, start_ngrok, start_localhost_run):
        try:
            url = fn()
        except Exception as e:  # noqa: BLE001
            log("隧道失败", fn.__name__, e)
            url = None
        if url:
            return url
    return None


def main():
    log("工作目录:", ROOT)
    load_secrets()
    check_gpus()
    if os.environ.get("CLONE_MOCK") != "1":
        kill_stale()
        install_deps()
        predownload()
    server = start_server()
    url = open_tunnel()
    if url:
        banner(url)
    else:
        log("⚠ 未能建立公网隧道。可在 Kaggle Secrets 添加 NGROK_TOKEN 后重试。")
    # ---- 看门狗:服务进程/公网隧道任何一个挂了都自动拉起,保证「一直连得上」 ----
    bad = 0
    try:
        while True:
            time.sleep(20)
            if server.poll() is not None:
                log("⚠ 服务进程退出(code=%s),正在重启 …" % server.returncode)
                server = start_server()
                bad = 0
                continue
            if not url:
                continue
            if tunnel_ok(url):
                bad = 0
                continue
            bad += 1
            log("⚠ 公网隧道检测失败 (%d/3)" % bad)
            if bad >= 3:
                log("重建公网隧道 …(网址会变化,请使用下面的新网址)")
                new = open_tunnel()
                if new:
                    url = new
                    banner(url)
                bad = 0
    except KeyboardInterrupt:
        server.terminate()


if __name__ == "__main__":
    main()
