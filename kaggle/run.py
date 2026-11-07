"""Kaggle 一键启动器 (Accelerator 请选 GPU T4 x2,Internet 打开)。

流程: 检查双卡 → 装依赖 → 并行预下载模型 → 启动服务 → 建立公网隧道 → 在输出框打印可点击网址。
用法(Kaggle 单元格):  !python /kaggle/working/clone-system/kaggle/run.py
"""
import json
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

    repos = [config.TTS_MODEL, config.ASR_MODEL, config.SV_MODEL] + [config.LLM_MODEL]  # v9: 对话大模型直接部署在 Kaggle(约 8GB)

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
    sh("pkill -f 'free.pinggy.io' || true; pkill -f 'nokey@localhost.run' || true", check=False)
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


# ---------------------------------------------------------------- v10: 双线路 + 线路地址发布
# 免费公网隧道(trycloudflare)没有 SLA,偶尔会整条卡住几十秒。第一性原理:单线路不可能「不断连」,
# 只能用冗余:主线路 cloudflared(http2)+ 备用线路(pinggy / localhost.run,走 SSH:443)同时在线,
# 全部地址写进 TUN_FILE,服务端经 /api/state 下发给页面和浏览器模拟器,主线路卡死时它们自动切换。
TUN_FILE = Path(os.environ.get("CLONE_TUNNEL_FILE", "/tmp/clone_tunnels.json"))
_PUB = {"primary": None, "urls": []}
_SEC = {"proc": None, "url": None, "name": None, "busy": False}
SSH_BASE = ["ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3", "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=15"]


def publish(primary=None, sec=None):
    if primary is not None:
        _PUB["primary"] = primary
    if sec is not None:
        _SEC["url"] = sec or None
    urls = [u for u in (_PUB["primary"], _SEC["url"]) if u]
    _PUB["urls"] = urls
    try:
        TUN_FILE.write_text(json.dumps({"primary": _PUB["primary"], "urls": urls, "t": time.time()}), "utf-8")
    except Exception:  # noqa: BLE001
        pass


def _ssh_tunnel(name, cmd, rx, wait=40):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    found = {}

    def rd():
        for line in iter(p.stdout.readline, ""):
            m = re.search(rx, line)
            if m and "url" not in found:
                found["url"] = m.group(0)

    threading.Thread(target=rd, daemon=True).start()
    for _ in range(wait):
        if "url" in found:
            return p, found["url"]
        if p.poll() is not None:
            break
        time.sleep(1)
    try:
        p.kill()
    except Exception:  # noqa: BLE001
        pass
    return None, None


def start_secondary():
    """备用线路(后台线程运行,不拖慢主线路出网址)。成功返回网址,并写入 TUN_FILE。"""
    if _SEC["busy"]:
        return None
    _SEC["busy"] = True
    try:
        if not shutil.which("ssh"):
            sh("(apt-get install -y -qq openssh-client > /dev/null 2>&1) || true", check=False)
        if not shutil.which("ssh"):
            log("备用线路需要 ssh,当前环境没有,跳过(主线路不受影响)")
            return None
        old = _SEC.get("proc")
        if old and old.poll() is None:
            old.kill()
        for name, cmd, rx in (
            # pinggy 实测输出: https://xxx.free.pinggy.net(正则不能匹配到 dashboard.pinggy.io)
            ("pinggy", SSH_BASE + ["-p", "443", "-R0:127.0.0.1:%d" % PORT, "free.pinggy.io"], r"https://[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:free\.pinggy\.(?:net|link)|pinggy-free\.link)"),
            ("localhost.run", SSH_BASE + ["-R", "80:127.0.0.1:%d" % PORT, "nokey@localhost.run"], r"https://[a-z0-9]+\.lhr\.life"),
        ):
            try:
                proc, url = _ssh_tunnel(name, cmd, rx)
            except Exception as e:  # noqa: BLE001
                log("备用线路 %s 失败: %s" % (name, e))
                continue
            if url and url != _PUB["primary"]:
                _SEC.update(proc=proc, name=name)
                publish(sec=url)
                log("✓ 备用线路就绪 (%s): %s   (主线路卡死时页面会自动切换,无需操作)" % (name, url))
                return url
        log("备用线路暂时不可用,稍后看门狗会自动重试(主线路不受影响)")
        return None
    finally:
        _SEC["busy"] = False


def banner(url):
    line = "=" * 74
    print("\n" + line)
    print("  ✅ 克隆语音系统已就绪,请打开下面的网址(粘贴到「浏览器模拟器」地址栏即可):")
    print("\n      " + url + "\n")
    print("  🛟 备用线路会在后台自动建立并写入页面(主线路卡死时自动切换);浏览器模拟器会同步感知")
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


def _probe_ms(url, timeout=10):
    t0 = time.time()
    ok = tunnel_ok(url, timeout)
    return ok, (time.time() - t0) * 1000


def main():
    log("工作目录:", ROOT)
    load_secrets()
    check_gpus()
    if os.environ.get("CLONE_MOCK") != "1":
        kill_stale()
        install_deps()
        predownload()
    try:
        TUN_FILE.unlink()
    except Exception:  # noqa: BLE001
        pass
    server = start_server()
    url = open_tunnel()
    if url:
        publish(primary=url)
        banner(url)
        threading.Thread(target=start_secondary, daemon=True).start()
    else:
        log("⚠ 未能建立公网隧道。可在 Kaggle Secrets 添加 NGROK_TOKEN 后重试。")
    # ---- 看门狗(v10:10 秒一次):服务进程 / 主线路 / 备用线路任何一个挂了都自动拉起 ----
    bad, sec_bad, n, lat, fail_total, t_log = 0, 0, 0, [], 0, time.time()
    try:
        while True:
            time.sleep(10)
            if server.poll() is not None:
                log("⚠ 服务进程退出(code=%s),正在重启 …" % server.returncode)
                server = start_server()
                bad = 0
                continue
            if url:
                ok, ms = _probe_ms(url)
                n += 1
                if ok:
                    bad = 0
                    lat = (lat + [ms])[-60:]
                else:
                    bad += 1
                    fail_total += 1
                    log("⚠ 主线路探测失败 (%d/3)" % bad)
                    if bad >= 3:
                        log("重建主线路 …(网址会变化;备用线路/页面会自动获知新网址)")
                        new = open_tunnel()
                        if new:
                            url = new
                            publish(primary=url)
                            banner(url)
                        bad = 0
            # 备用线路:进程退出或探测连续 3 次失败 → 后台重建
            su = _SEC.get("url")
            if su and not _SEC["busy"]:
                sp = _SEC.get("proc")
                dead = (sp is not None and sp.poll() is not None)
                if not dead and n % 3 == 0:
                    dead = not tunnel_ok(su, 10)
                    sec_bad = sec_bad + 1 if dead else 0
                    dead = dead and sec_bad >= 3
                if dead:
                    sec_bad = 0
                    publish(sec="")
                    threading.Thread(target=start_secondary, daemon=True).start()
            elif url and not su and not _SEC["busy"] and n % 6 == 0:
                threading.Thread(target=start_secondary, daemon=True).start()
            if time.time() - t_log > 600 and lat:   # 每 10 分钟一行真实统计(服务端到公网隧道的往返)
                sl = sorted(lat)
                log("运行正常 · 主线路探测 %d 次,失败 %d 次,延迟均值 %.0fms / P95 %.0fms · 备用线路: %s" % (
                    n, fail_total, sum(lat) / len(lat), sl[int(len(sl) * 0.95) - 1 if len(sl) > 1 else 0], _SEC.get("url") or "无"))
                t_log = time.time()
    except KeyboardInterrupt:
        server.terminate()


if __name__ == "__main__":
    main()
