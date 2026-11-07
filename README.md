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

## v9 改动(对话大模型直接部署在 Kaggle,彻底移除 DeepSeek / Key)

* **对话模型 = Qwen3-4B-Instruct-2507**,直接加载在 Kaggle 的两张 T4 上(fp16,按层分到两卡,每卡约 3.8GiB),不需要任何外部 API、Key、Internet 之外的账号。原生 262K 上下文,适合长文。
* **长文本输出(`server/workers.py: RealLLM.chat`)**:一轮 `generate` 撞到 `LLM_ROUND_TOKENS`(默认 2048)而模型没有自然结束时,把已生成的文字接在提示词后面**自动续写**,直到模型自己说完或用完总预算(普通 `LLM_DEFAULT_TOKENS=6144`,「长文」模式 `LLM_LONG_TOKENS=20000`)。文字流不中断,前端只收到一个 `cont` 事件(toast 提示);用户取消时立刻停止、不再续写。
* **不和语音抢 GPU(令牌桶限速 `Pacer`)**:语音只消耗约 3~6 token/s,LLM 在两张卡上全速生成只会拖慢 TTS(TTS 的 RTF≈0.93 已经很紧)。每轮对话前 `LLM_PACE_BURST=48` 个 token 不限速(首句、首声最快),之后限制在 `LLM_PACE_TPS=12` token/s(仍是语音消耗速度的 2~4 倍,TTS 永远不会断粮)。设 `LLM_PACE_TPS=0` 关闭。
* **每一句都发 `plan` 事件**(旧的本地路径只在结尾发一次):播放器据此实时更新「预计总音频时长」,弹性变速/预缓冲才能做对决定 → 首声快且播放中不停顿。
* **对话模型还在加载时点发送**:最多等 `LLM_LOAD_WAIT=900` 秒并提示「对话模型正在加载…」,加载完自动开始回答(不再直接报错)。
* 界面「对话」指示灯直接反映本地 LLM worker 状态;`/api/ping` 返回 `llm` 状态。
* 自检:`CLONE_MOCK=1 python tests/e2e_localllm.py`(DeepSeek 已移除 / plan 事件 / 续写 / 故障回显 / 限速);`python tests/real_llm_cpu.py`(用真实小模型在 CPU 上验证续写、取消、限速逻辑,需 torch+transformers)。

## v7 改动(零停顿播放 + 复制代码界面;其中「DeepSeek API」已在 v9 移除)

* **零停顿播放**(`web/player.js`,自适应预缓冲):服务端为每轮对话发送 `plan` 事件(预计总音频时长),浏览器实测合成速度 r,按 `B ≥ E_rem·(1/r − 1)` 计算开播前需要攒多少缓冲才能一次不断;开播后块与块在采样级精确衔接。离线仿真:`node tests/player_sim.js`。
* **相似度**只显示真实测量值(WavLM 声纹余弦);未达 95% 时界面会如实提示,不会伪造数字。
* 语气词先行默认关闭(本地模型首字很快,语气词反而会造成空档)。

## v8 改动(首声最快 + 零停顿 + 歌曲参考音频 + 代码同步检测)

### 实测日志里的真问题(logs/run-log-20261107-143559.txt)
* 页面「首声 19.20s」≠ 服务端首声(1.4s)。差距来自:① 播放器为了「绝不断流」把合成速率≈实时的场景预缓冲到 8~14 秒(`node tests/player_sim.js` 在 v7 复现:1.0x → 首声 16.4s);② `POST /api/chat` 平均 4.6s、最大 8.7s(隧道 RTT≈400ms,POST→GET 两个往返 + 12 秒长轮询的批量到达);③ 音频走 JSON+base64(+33% 体积)。
* 仍然「断流 1 次」:预缓冲只能赌估计,网络批量到达的抖动一大就断。
* 参考音频是歌曲:Whisper 把歌词识别成乱码,ICL 相似度只有 55%,x-vector 96.6%、信噪比 18dB —— 伴奏被当成说话人特征克隆了。

### 修复
1. **弹性变速播放(`web/wsola.js` + `web/player.js`)**:第一性原理——不断流只要求「消耗速度 ≤ 生产速度」。缓冲不足时把播放速度 q 轻微降到 [0.87, 1](WSOLA,变速不变调,音色/谱包络不变);缓冲充足立刻回到 1.0(此时是逐采样精确重建,没有任何处理痕迹)。所需预缓冲由 `E_rem·(1/r−1)` 降为 `E_rem·(q_min/r−1)`,r≥0.87 时第一块到达即开播。仿真:1.0x 场景首声 16.4s → 2.0s;0.95x/180s 长文 15.9s → 5.9s;0 断流。`node tests/wsola_test.js` 验证:q=1 精确重建(误差 3e-8)、音高不变、无不连续、分块无关。
2. **流式通道 `POST /api/chat_stream`**:一个请求「创建对话 + 实时推送」,二进制帧(无 base64,−25% 体积)、无轮询间隙、少 1~2 个往返;4 秒内收不到首帧 → 判定隧道缓冲了流式响应,自动退回长轮询;流中途断开 → 用 tid + 已收事件数无缝续传(不丢不重)。真实 Chromium 自检:`node tests/browser_stream.js`(正常 / 隧道缓冲 / 中途掐断三种场景)。
3. **歌曲/带伴奏参考音频(`server/vocals.py` + `server/vocal_runner.py`)**:勾选「🎵 参考音频是歌曲」或信噪比 < 30dB 时,训练前先用 demucs(htdemucs)在 GPU1 子进程里抽出干净人声,再清洗、识别、克隆。失败自动沿用原音频,不影响训练。`REF_SEPARATE=auto|1|0`。
4. 首声提示的 tooltip 现在拆开显示:点击→听到声音 / 首块音频到达 / 起播等待 / 通道类型,方便定位瓶颈。

### 关于「98% 相似度」的说明(请读)
界面上的相似度是 **WavLM 说话人嵌入余弦的真实测量值**,不会伪造。零样本克隆能否稳定到 98% 取决于参考音频质量与模型本身,没有任何代码能在没有 GPU 实测的情况下保证这个数字。v8 做的是**去掉已知的拉低因素**(伴奏/噪声/歌词乱码文本),训练后以界面真实数值为准;若低于目标,界面会如实提示。

## v6 改动(「点开始训练卡在『正在上传参考音频』」+ 运行日志)

### 根因
截图里的现象:`已点击开始训练 · 正在上传参考音频(4683 KB)…`,进度条不动,顶部「连接 重连中」。旧版把整段录音(这里 4.6MB、1 分 10 秒)用**一个**请求经公网隧道(trycloudflare)上传,
隧道对大请求很不稳定:慢、被中途掐断,而旧版超时 60 秒 × 重试 4 次,每次都从头重传,界面看起来就是「训练不了」。而且服务端最终只用前 20 秒,其余 50 秒白传、降噪还对整段 70 秒做。

### 修复
* 浏览器端先压缩:解码 → 单声道 → 24kHz → 最多 30 秒 → 16bit WAV(4.6MB → 约 1.4MB;解码失败时原文件照样上传,服务端用 ffmpeg)。
* **分块上传**(`server/uploads.py`):192KB 一块、3 块并行、每块独立超时/重试(指数退避);已到达的块不重传(断点续传,再点一次「开始训练」会接着传);
  全部到齐后发一个很小的 `POST /api/voice/train_uploaded`(幂等:响应丢了重发不会重复训练;缺块返回 422 + 缺失列表,前端自动补传)。界面实时显示 `上传 63% · 480 KB/s · 已自动重试 N 次`。
* 旧接口 `/api/voice/train` 保留;新前端遇到旧服务端会自动退回一次性上传。
* 降噪前先截断到 32 秒(旧版对整段降噪),训练 CPU 阶段更快。

### 运行日志(顶部 📋 按钮,`server/perf.py`)
* 服务端剖析每个环节:上传分块接收/拼装、解码、清洗、每张卡的声纹提取/合成校验、Whisper、相似度、训练各阶段、对话首声/LLM 首字 …… 次数 / 平均 / P95 / 最大。
* **瓶颈分析**规则引擎:最慢训练阶段 + 建议、上传速度/重试、接口失败率与延迟、显存、首声、模型加载、磁盘。
* 面板可复制 / 下载 .txt / .json / 「复制并打开 GitHub」(在 `logs/` 下新建文件提交即可)。接口:`GET /api/diag`(含 `perf` `analysis`)、`GET /api/perf`。
* 自检:`CLONE_MOCK=1 python tests/e2e_upload.py`;真实浏览器(含模拟隧道丢包)`node tests/browser_e2e.js`。
* 改过任何文件后运行 `python tools/build_all_in_one.py` 重新生成 `kaggle_all_in_one.py`。

## 文件

```
server/        后端 (FastAPI + 多进程 GPU worker)
web/index.html 前端(单屏,无需滚动)
kaggle/run.py  Kaggle 启动器(装依赖/下载模型/起服务/建公网隧道/看门狗)
tests/         Mock 端到端自检 (CLONE_MOCK=1, 不需要 GPU):e2e_mock / e2e_fail / e2e_diag / e2e_numeric / e2e_stream;JS: player_sim / wsola_test / browser_stream
server/errors.py  数值异常识别 + 精度回退链(v5)
server/perf.py + uploads.py  性能剖析/瓶颈分析 + 分块上传(v6)
tools/build_all_in_one.py  重新生成 kaggle_all_in_one.py
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
