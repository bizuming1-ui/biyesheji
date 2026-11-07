## biyesheji · 克隆语音系统 (Voice Clone Studio) v5

在 **Kaggle GPU T4 x2** 上部署的实时语音克隆 + 语音/文字对话系统。

## 架构(双卡并行,模型常驻显存)

| 进程 | GPU | 作用 |
| ---- | --- | ---- |
| tts0 | GPU0 | Qwen3-TTS-12Hz-1.7B-Base 声音克隆(faster-qwen3-tts: CUDA Graph + 流式输出) |
| tts1 | GPU1 | 同上(第二份) |
| llm  | GPU0+GPU1 | Qwen3-4B-Instruct-2507 按层均分到两张卡,流式长文本对话(长文模式 4096 token) |
| aux  | GPU1 | Whisper(语音识别) + WavLM(说话人相似度) |

每个进程用 `CUDA_VISIBLE_DEVICES` 独占一张卡,互不抢 GIL,双卡真并行。

## v3 改动(针对「连接不稳定 / 训练太慢 / 沙沙声」)

### 1. 连接 100% 稳定:对话改为「与连接无关」的 HTTP 长轮询
旧版用 WebSocket,Cloudflare 隧道/代理会**悄悄丢掉**它,表现为「发了没反应」「服务端 12 秒没有响应」。
现在:
* `POST /api/chat` 创建一轮对话 → 服务端**独立生成**(文本 + 音频事件写入事件日志,与任何连接无关);
* `GET /api/events?after=N` 长轮询取增量(12 秒一次);任何请求失败都用**同一个游标**自动重试,数据一个不丢不重;
* 前端 5 秒心跳显示连接状态;刷新页面训练进度自动恢复;
* 服务端 worker 进程被系统杀掉 → 立刻通知界面并可一键重启(不再无限等待);
* `kaggle/run.py` 带**看门狗**:uvicorn 退出自动重启;公网隧道连续 3 次探测失败自动重建(cloudflared 用 http2 而不是易被封的 QUIC;再不行依次回退 ngrok / localhost.run);
* 关闭 uvicorn 访问日志,Kaggle 输出框不再被 `/api/status` 刷屏。

### 2. 训练提速:双卡从第 1 秒起同时干活
```
清洗参考音频
 ├─ GPU0: x-vector 声纹 → 合成校验句 → 相似度
 ├─ aux : Whisper 识别文本(短音频不分块、带语言提示、已预热)
 └─ GPU1: (等文本) ICL 上下文 → 合成校验句 → 相似度
择优 → 把胜出的模式补到另一张卡 → 完成
语气词缓存放到后台(不再占用训练时间)
```
旧版是 Whisper → 两张卡准备 → 合成校验 → 语气词,全部串行。

### 3. 音质:去沙沙 / 无爆音
* 软限幅改为「带膝点」:|x|<0.8 完全线性(旧版 tanh 对所有采样都有轻微失真);
* 流式「底噪门」(下行扩展器):词间/句尾底噪被平滑压低最多 ~16 dB,语音不动,10ms 帧 + 线性插值,无台阶;
* TTS 采样温度 0.75 / top_k 40(可用环境变量 `TTS_TEMPERATURE` 等调整:更低 = 更稳更像,更高 = 情绪起伏更大);
* 跨块连续高通、整句固定增益、句末才淡出(保留)。

## v4 改动(训练「点了没反应」/ 检测日志)

* **训练立即起步**:旧版点击「开始训练」后会等「两张 TTS + Whisper + 相似度」全部加载完(首次运行下载模型时可能十几分钟,界面像没反应)。
  现在只要 1 张卡的 TTS 就绪就开始;Whisper 在后台等待;另一张卡就绪后自动补上当前声音,双卡一起合成。
  等待期间界面会实时显示每个模型的加载状态(`tts0=loading, tts1=ready …`)。点击后立刻显示「已上传…」。
* **检测日志**(浏览器模拟器右上角):
  * `GET /api/diag`:版本 / GPU / 进程 / 全部配置 / 训练时间线 / 声音 / 内存磁盘 / 服务端日志(环形缓冲,不影响 Kaggle 输出框);
  * `POST /api/clientlog`:网页探针(`web/index.html` 内)上报每个 /api 请求的耗时、失败、按钮点击、JS 错误、音频状态;
  * 模拟器侧 3 秒心跳(延迟 / P95 / 丢包)、浏览器环境、GitHub 代码同步状态;可复制 / 下载 / 一键上传到 `logs/`。
* 自检:`CLONE_MOCK=1 python tests/e2e_diag.py`

## v5 改动(「点开始训练就失败」的根因修复 + 训练提速)

### 根因

Qwen3-TTS 声音克隆模型在 **T4(Turing 架构,没有原生 bf16)上用 float16 推理**时,logits 会出现 inf/NaN,采样处 `torch.multinomial`
触发 CUDA 设备端断言:`probability tensor contains either inf, nan or element < 0` / `device-side assert triggered`
(QwenLM/Qwen3-TTS issue #43)。旧版默认 `TTS_DTYPE=float16`,而这个错误恰好发生在训练的「提取声纹 / 合成校验句」里;
旧版的回退逻辑只检查「输出音频是不是 NaN」,**抓不到这个异常**,而且 CUDA 断言之后整个 GPU 进程的上下文已经报废,只能重启进程。

### 修复

* 默认精度改为 **bfloat16**(faster-qwen3-tts 官方在 Tesla T4 上的基准 RTF≈0.93 用的就是 bf16)。
* 新增 `server/errors.py`:识别数值异常 + 精度回退链 `float16 → bfloat16 → float32`。
* 引擎(`engine.py`)遇到数值异常:**自动重启 TTS 进程 → 换更稳的精度 → 训练自动重试**(用户什么都不用做);
  对话阶段遇到同样的异常也会后台自动降级。重启时旧的加载任务会作废,不会往新进程里塞第二次加载。
* 训练任务持有强引用(防止被垃圾回收)、卡死的旧训练会被新训练取代(不再出现「训练正在进行中」永远点不了)。

### 提速

* 训练时只提取并缓存声纹,不再多合成一句「嗯。」预热(`prepare(warm=False)`);校验句缩短、流式块加大(`TRAIN_CHUNK=16`)。
* 合成完最多再等 Whisper/相似度模型 30 秒(`TRAIN_AUX_WAIT`),识别文本最多等 90 秒(`TRAIN_ASR_WAIT`),不再被它们拖住。
* 失败时界面直接显示真实原因和最近日志(不再只是一闪而过的提示)。
* 自检:`CLONE_MOCK=1 python tests/e2e_numeric.py`(模拟 NaN 崩溃 → 自动降级 → 训练成功 → 对话可用)

## 文件

```
server/        后端 (FastAPI + 多进程 GPU worker)
web/index.html 前端(单屏,无需滚动)
kaggle/run.py  Kaggle 启动器(装依赖/下载模型/起服务/建公网隧道/看门狗)
tests/         Mock 端到端自检 (CLONE_MOCK=1, 不需要 GPU):e2e_mock / e2e_fail / e2e_diag / e2e_numeric
server/errors.py  数值异常识别 + 精度回退链(v5)
kaggle_all_in_one.py  粘贴到 Kaggle 单元格即可运行的完整脚本(内置以上全部文件)
```

## 在 Kaggle 运行

1. Notebook 右侧 Settings:Accelerator = **GPU T4 x2**,Internet = **On**。
2. 把「复制代码」页面里的 `kaggle_all_in_one.py` 整个粘贴到一个代码单元格运行。
3. 输出框会打印 `https://xxxx.trycloudflare.com`,粘贴到浏览器模拟器地址栏即可。
4. 首次运行需要下载约 15GB 模型并安装依赖(约 8~15 分钟);之后同一会话内再次运行很快。

## 本地自检(无 GPU)

```
CLONE_MOCK=1 python tests/e2e_mock.py   # 主流程 + 断线续传 + DSP 音频检查
CLONE_MOCK=1 python tests/e2e_fail.py   # 一张卡加载失败时的降级
CLONE_MOCK=1 python tests/e2e_numeric.py # 训练根因:NaN 崩溃 → 自动降级精度 → 训练成功
```

## 关于指标的说明(请读)

* **99.5% 相似度**:界面显示的是 WavLM 说话人嵌入余弦相似度的**真实测量值**(你的实测 96.35%)。这个度量下 99.5% 不是任何零样本克隆模型能保证的数值;系统通过「ICL / x-vector 双模式择优」+ 参考音频清洗尽量提高它。参考音频越干净(安静环境、8~20 秒、情绪自然、带自然呼吸)越像。想再提高:录 15~20 秒、一口气自然朗读、不要有背景音乐。
* **首声**:克隆声音的「嗯,」「好的,」等语气词在训练后台预先缓存,每轮对话开口即播,正文同时在双卡合成。
* **不会断流**:T4 单卡合成速度约实时的 0.9 倍;系统用「双卡交替合成 + 自适应缓冲」保证播放不断流。
* Kaggle 会话有时长上限(约 12 小时/长时间无操作会断开),会话结束后模型随之卸载,重新运行单元格即可。
