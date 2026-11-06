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
MARK = Path("/tmp/.clone_deps_ok_v2")


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


def install_deps():
    if MARK.exists():
        log("依赖已安装,跳过")
    else:
        sh(
            "pip install -q qwen-tts fastapi 'uvicorn[standard]' python-multipart soundfile scipy "
            "noisereduce accelerate pillow huggingface_hub"
        )
        MARK.write_text("ok")
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

    repos = [config.TTS_MODEL, config.LLM_MODEL, config.ASR_MODEL, config.SV_MODEL, config.DET_MODEL]

    def dl(r):
        t = time.time()
        try:
            snapshot_download(r, token=os.environ.get("HF_TOKEN") or None)
            log("✓ 模型就绪 %s (%.0fs)" % (r, time.time() - t))
        except Exception as e:  # noqa: BLE001
            log("✗ 下载失败 %s: %s" % (r, e))

    with ThreadPoolExecutor(5) as ex:
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
