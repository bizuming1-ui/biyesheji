# biyesheji · 克隆语音系统 (Voice Clone Studio)

在 **Kaggle GPU T4 x2** 上部署的实时语音克隆 + 对话 + 万物检测系统。

## 架构(双卡并行)

| 进程 | GPU | 作用 |
| --- | --- | --- |
| `tts0` | GPU0 | Qwen3-TTS-12Hz-1.7B-Base 声音克隆合成 |
| `tts1` | GPU1 | Qwen3-TTS-12Hz-1.7B-Base 声音克隆合成 |
| `llm`  | GPU0 | Qwen3-4B-Instruct-2507 流式长文本对话 |
| `aux`  | GPU1 | Whisper(语音识别) + WavLM(说话人相似度) + OWL-ViT(开放词汇物体检测) |

- **训练**(点击「开始训练」):参考音频清洗(降噪/切静音/归一化)→ Whisper 自动转写 → **双卡同时**提取声学特征 → **双卡同时**合成校验句 → WavLM 计算说话人相似度 → 在 ICL / x-vector 两种克隆模式中择优。
- **对话**:LLM 流式输出 → 首句 4~14 字即切句 → 句子在两张卡上交替/并行合成 → 按顺序经 WebSocket 推送 → 浏览器 WebAudio 无缝排队播放(缺料时自适应加大缓冲)。
- **自然度**:60Hz 高通去直流/低频噪声、响度对齐参考、8ms 淡入淡出、软限幅;句间加入极弱的带通呼吸声(可调,默认很轻)。
- **万物检测**:摄像头画面 → OWL-ViT 开放词汇检测,可自定义词,可把检测结果交给 AI 对话。

## 文件

```
server/        后端 (FastAPI + 多进程 GPU worker)
web/index.html 前端(单屏,无需滚动)
kaggle/run.py  Kaggle 启动器(装依赖/下载模型/起服务/建公网隧道)
tests/         Mock 端到端自检 (CLONE_MOCK=1, 不需要 GPU)
```

## 在 Kaggle 运行

1. Notebook 右侧 Settings:Accelerator = **GPU T4 x2**,Internet = **On**。
2. 把「复制代码」页面里的 `kaggle_all_in_one.py` 整个粘贴到一个代码单元格运行。
3. 输出框会打印 `https://xxxx.trycloudflare.com`,粘贴到浏览器模拟器地址栏即可。

## 本地自检(无 GPU)

```
CLONE_MOCK=1 python tests/e2e_mock.py
```

## 关于指标的说明(请读)

- “首声 < 3 秒”取决于 T4 上 1.7B 模型的实时率。系统用最短首句 + 双卡并行 + 预热来压低首声,界面实时显示真实测得的首声时间。
- “相似度”显示的是 WavLM 说话人嵌入余弦相似度,是真实测量值,不是固定宣传值;99.5% 在该度量下并不是一个可保证的数值,系统通过「择优模式 + 高质量参考音频」尽量提高它。参考音频越干净(安静环境、8~20 秒、情绪自然)越像。
