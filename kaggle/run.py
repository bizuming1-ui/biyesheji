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
MARK = Path("/tmp/.clone_deps_ok_v3")


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


def install_deps():
    base = "fastapi 'uvicorn[standard]' python-multipart soundfile scipy noisereduce accelerate huggingface_hub"
    if MARK.exists():
        log("依赖已安装,跳过")
    else:
        c = _torch_constraints()
        # 首选 faster-qwen3-tts: CUDA Graph + 流式输出 (T4 上 RTF 约 2x 提升,首块音频约 1 秒)
        r = sh("pip install -q -c %s faster-qwen3-tts %s" % (c, base), check=False)
        if r.returncode != 0:
            log("⚠ faster-qwen3-tts 安装失败,回退官方 qwen-tts(整句合成,首声会慢)")
            sh("pip install -q -c %s qwen-tts %s" % (c, base))
        MARK.write_text("ok")
    chk = subprocess.run([sys.executable, "-c", "import faster_qwen3_tts"], capture_output=True, text=True)
    if chk.returncode == 0:
        log("✓ faster-qwen3-tts 可用(CUDA Graph 流式)")
    else:
        log("faster-qwen3-tts 不可用,将使用官方 qwen-tts:", (chk.stderr.strip().splitlines() or ["?"])[-1])
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

    repos = [config.TTS_MODEL, config.LLM_MODEL, config.ASR_MODEL, config.SV_MODEL]

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


def start_server():
    env = dict(os.environ, PYTHONUNBUFFERED="1", PORT=str(PORT))
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", str(PORT), "--ws-ping-interval", "20"],
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


def start_cloudflared():
    exe = Path("/tmp/cloudflared")
    if not exe.exists():
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        log("下载 cloudflared …")
        urllib.request.urlretrieve(url, exe)
        exe.chmod(0o755)
    p = subprocess.Popen(
        [str(exe), "tunnel", "--url", "http://127.0.0.1:%d" % PORT, "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    found = {}

    def rd():
        for line in iter(p.stdout.readline, ""):
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m and "url" not in found:
                found["url"] = m.group(0)

    threading.Thread(target=rd, daemon=True).start()
    for _ in range(60):
        if "url" in found:
            return found["url"]
        time.sleep(1)
    return None


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
    print(line + "\n", flush=True)
    try:
        from IPython.display import HTML, display

        display(HTML('<h2>🎙 <a href="%s" target="_blank" style="color:#7c5cff">打开克隆语音系统</a></h2><p>%s</p>' % (url, url)))
    except Exception:  # noqa: BLE001
        pass


def main():
    log("工作目录:", ROOT)
    load_secrets()
    check_gpus()
    install_deps()
    if os.environ.get("CLONE_MOCK") != "1":
        predownload()
    server = start_server()
    url = None
    for fn in (start_cloudflared, start_ngrok):
        try:
            url = fn()
        except Exception as e:  # noqa: BLE001
            log("隧道失败", fn.__name__, e)
        if url:
            break
    if url:
        banner(url)
    else:
        log("⚠ 未能建立公网隧道。可在 Kaggle Secrets 添加 NGROK_TOKEN 后重试。")
    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()


if __name__ == "__main__":
    main()
