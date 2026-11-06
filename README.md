# biyesheji · 克隆语音系统 (Voice Clone Studio)

在 **Kaggle GPU T4 x2** 上部署的实时语音克隆 + 语音/文字对话系统(已移除摄像头/物体检测)。

## 架构(双卡并行,模型常驻显存)

| 进程 | GPU | 作用 |
| --- | --- | --- |
| `tts0` | GPU0 | Qwen3-TTS-12Hz-1.7B-Base 声音克隆(faster-qwen3-tts: CUDA Graph + 流式输出) |
| `tts1` | GPU1 | Qwen3-TTS-12Hz-1.7B-Base 声音克隆(同上) |
| `llm`  | GPU0+GPU1 | Qwen3-4B-Instruct-2507 按层均分到两张卡(每卡约 3.8GiB),流式长文本对话(长文模式 4096 token) |
| `aux`  | GPU1 | Whisper(语音识别) + WavLM(说话人相似度) |

每个进程用 `CUDA_VISIBLE_DEVICES` 独占一张卡,互不抢 GIL,双卡真并行。服务进程不退出,模型就一直常驻显存。

## 数据流

- **训练**(点击「开始训练」):参考音频清洗(高通/降噪/切静音/归一化)→ Whisper 自动转写 → **双卡同时**构建声纹并预热 CUDA Graph → ICL / x-vector 两种克隆模式**分别在两张卡上同时**合成校验句 → WavLM 计算说话人相似度 → 择优 → 双卡并行缓存「语气词」。
- **对话**:LLM 流式输出 → 首句 3~14 字即切句 → 每句分配给空闲的卡 → **每 0.33~0.67 秒音频就流式推送一块** → 带状态后处理(跨块连续高通滤波、整句固定增益、句末才淡出、软限幅)→ WebSocket → 浏览器 WebAudio 无缝排队播放(缺料时自适应加大缓冲)。
- **首声**:克隆声音的「嗯,」「好的,」等语气词在训练时预先缓存,每轮对话开口即播(≈0 延迟),正文同时在双卡合成。
- **不会「发了没反应」**:LLM 生成线程里的任何异常都会被捕获并回显到界面;fp16 溢出自动改用 bf16 重试;服务端 60 秒无输出会报错;前端有 12 秒/50 秒看门狗。

## 文件

```
server/        后端 (FastAPI + 多进程 GPU worker)
web/index.html 前端(单屏,无需滚动)
kaggle/run.py  Kaggle 启动器(装依赖/下载模型/起服务/建公网隧道)
tests/         Mock 端到端自检 (CLONE_MOCK=1, 不需要 GPU)
kaggle_all_in_one.py  粘贴到 Kaggle 单元格即可运行的完整脚本(内置以上全部文件)
```

## 在 Kaggle 运行

1. Notebook 右侧 Settings:Accelerator = **GPU T4 x2**,Internet = **On**。
2. 把「复制代码」页面里的 `kaggle_all_in_one.py` 整个粘贴到一个代码单元格运行。
3. 输出框会打印 `https://xxxx.trycloudflare.com`,粘贴到浏览器模拟器地址栏即可。
4. 首次运行需要下载约 15GB 模型并安装依赖(约 8~15 分钟);之后同一会话内再次运行很快。

## 本地自检(无 GPU)

```
CLONE_MOCK=1 python tests/e2e_mock.py
```

## 关于指标的说明(请读)

- “首声 < 3 秒”:T4 上 1.7B 模型的流式首块约 1 秒,加上语气词先行,通常能在 3 秒内听到声音;界面实时显示真实测得的首声时间。
- T4 单卡合成速度约为实时的 0.9 倍;系统用「双卡交替合成 + 自适应缓冲」保证播放不断流,但**不是**数学意义上的绝对保证。
- “相似度”显示的是 WavLM 说话人嵌入余弦相似度,是真实测量值;99.5% 在该度量下不是一个可保证的数值,系统通过「择优模式 + 高质量参考音频」尽量提高它。参考音频越干净(安静环境、8~20 秒、情绪自然)越像。
- Kaggle 会话有时长上限(约 12 小时/长时间无操作会断开),会话结束后模型随之卸载,重新运行单元格即可。

## 故障排查(v2 修复记录)

- `property 'upsample' of 'Qwen3TTSTokenizerV2Encoder' object has no setter` / `MimiEncoderConfig has no attribute rope_theta`
  → 官方 `qwen-tts` 与 transformers 5.x 不兼容。正确组合:`faster-qwen3-tts` + `qwen-tts-hf` + `transformers==5.15.1`(不要同时装 `qwen-tts`)。`kaggle/run.py` 会自动卸载冲突包并做一次真实的编码器构造自检。
- `CUDA out of memory`(GPU0)→ 以前 LLM(7.6GiB)+TTS 同在 GPU0 且失败回退又重复加载一次。现在 LLM 均分两卡、加载失败先释放显存再回退,并自动重试 2 次。
- 「发了没反应」→ 前端消息先上屏;WebSocket 8 秒心跳 + 30 秒无 pong 自动重连;断线期间发送的消息连上后自动补发;状态接口不再被 nvidia-smi 卡住事件循环。
- 界面顶部出现红条 = 某个模型加载失败,会显示真实原因,点「重新加载模型」即可重试,无需重启单元格。
