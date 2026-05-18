# Magic Box

当前版本：V1.5.6

Magic Box 是 Apple Silicon 上的本地播客 TTS 工具。当前稳定入口只有两个：

- `streamlit_app.py`：正式 Web 生产台，支持 URL 正文抽取、长文切分、断点继续、多人源声音 Profile。
- `podcast_generator.py`：命令行长文生成入口，适合批处理和固定脚本。

旧 Gradio UI、打包 app、个人声音素材和历史生成音频已经清理；公开仓库只保留通用程序，用户需要自行录制或上传自己的参考音频。

## 最近修复

- 补齐并强校验 Qwen 模型目录里的 `speech_tokenizer/config.json` 和 `speech_tokenizer/model.safetensors`；缺少 speech tokenizer 权重时会直接报错，不再继续生成噪音音频。
- `download_model.py` 下载完成后会做模型完整性校验；Web/CLI 加载模型前也会校验主权重、配置和 speech tokenizer 权重是否齐全。
- 默认输出改为 MP3，CLI 与 Web 都支持 `mp3`、`wav`、`both`；MP3 转换依赖本机 `ffmpeg`。
- 修复响度标准化后的硬 clipping：现在使用 peak limiter 按比例降峰，并在写入前后输出音频健康警告。
- 生成链路保留后端返回的真实 sample rate，避免不同后端或实验模型被强行写成固定采样率。
- 依赖升级到稳定 PyPI 版本：`mlx==0.31.2`、`mlx-audio==0.4.3`、`mlx-lm==0.31.3`、`mlx-metal==0.31.2`、`transformers==5.8.1`、`huggingface_hub==1.15.0`。

## 快速启动

```bash
cd /Users/liuchang/Apprun/chenxi/qwen3-tts-apple-silicon
source .venv/bin/activate
streamlit run streamlit_app.py
```

浏览器打开 `http://127.0.0.1:8501/`。

## Web 生产台

Streamlit 页面用于正式长文播客生产：

- 粘贴、上传 TXT，或从 URL 抽取文章正文，并清除广告、订阅、分享、相关阅读等网页杂质后整理成播客稿。
- 选择 Qwen 模型，默认使用本地 `1.7B-Base`。
- 选择已保存的“克隆声音 Profile”，或临时上传自己的参考音频。
- 使用“表达预设”和高级微调控制语速、随机性和单段字符上限。
- 输出 WAV、MP3，或同时输出 WAV + MP3。
- 开启“从断点继续”后，中断任务可按同一输出文件名续跑。
- 生成后在页面底部输出库试听和下载所选格式。

### 声音 Profile

公开版不会内置任何个人声音。用户可以在页面里临时上传参考音频，也可以在“管理声音 Profile”里保存自己的样本。

新保存的 profile 会写入：

```text
voices/profiles/<profile_id>/
  metadata.json
  reference.<ext>
  transcript.txt
```

这些个人声音数据默认不提交 Git。

`audio_samples/` 仅作为本机可选素材目录保留占位，目录内音频默认被 Git 忽略。不要把真实人声素材提交到公开仓库。

## CLI 生成

```bash
.venv/bin/python podcast_generator.py \
  --file 稿件.txt \
  --ref-audio 我的声音.m4a \
  --ref-text "参考音频里具体说的文字内容" \
  --output outputs/my_episode.mp3 \
  --format both \
  --no-normalise
```

关键参数：

- `--file` / `--text`：输入稿件。
- `--ref-audio`：用于克隆的源声音。
- `--ref-text`：源声音里实际说的文字，必须尽量逐字一致。
- `--output`：输出路径；默认自动生成 `.mp3` 文件名。长文建议固定文件名，方便断点继续。
- `--format`：输出格式，支持 `wav`、`mp3`、`both`，默认 `mp3`。
- `--resume`：从已有 `.ckpt` 断点继续。
- `--speed` / `--temperature`：控制语速和表达随机性。

断点文件会记录稿件切分和声音 profile 元数据；如果换了源声音，不会复用旧断点，避免混入不同人的声音片段。

## 模型与依赖

首次运行或模型异常时，先重新下载主模型：

```bash
.venv/bin/python download_model.py 1
```

完整的 Qwen voice clone 模型目录必须包含：

```text
config.json
model.safetensors
speech_tokenizer/config.json
speech_tokenizer/model.safetensors
```

如果缺少 `speech_tokenizer/model.safetensors`，Qwen 可能输出近似噪音的音频；当前版本会在加载阶段拦截这个问题。

## 性能参考

在 Apple Silicon 本地运行时，Qwen 生成速度会明显受内存压力、分段长度、模型大小和后台应用影响。历史实测大致范围：

| 输出音频时长 | 生成耗时 | 说明 |
|---|---:|---|
| 30 秒 | 约 5 分钟 | 短片段试听 |
| 1-1.5 分钟 | 约 5-23 分钟 | 波动较大，取决于分段和内存状态 |
| 10 分钟 | 约 2.5 小时 | 建议固定输出文件名并开启断点续跑 |

长文建议接电运行，减少其它重型应用，并优先使用 `--resume` 保护进度。

## 项目结构

```text
streamlit_app.py        # 正式 Web app
podcast_generator.py    # CLI 长文生成
voice_profiles.py       # 多人源声音 profile 管理
voice_controls.py       # 表达预设和节奏优化
article_extractor.py    # URL 正文抽取
tts_backends.py         # Qwen/VoxCPM 后端封装
audio_samples/          # 本机可选参考音频，默认忽略，只提交 .gitkeep
voices/profiles/        # 用户保存的声音 profile，默认忽略
outputs/                # 生成结果，默认忽略
models/                 # 本地模型，默认忽略
tests/                  # 单元测试
```

## 清理边界

可以清理：

- `outputs/` 下除 `.gitkeep` 之外的生成结果。
- `audio_samples/` 下除 `.gitkeep` 之外的个人参考音频。
- `voices/profiles/` 下除 `.gitkeep` 之外的个人声音 profile。
- `runtime/`、缓存、日志、`.DS_Store`、打包产物。

不要清理：

- `models/`：本机运行依赖。
- `.venv/`：本机 Python 环境。
- `audio_samples/.gitkeep`：保留目录结构。
- `voices/.gitkeep` 和 `voices/profiles/.gitkeep`：保留目录结构。

## 验证

```bash
.venv/bin/python -m py_compile streamlit_app.py voice_profiles.py voice_controls.py article_extractor.py podcast_generator.py
.venv/bin/python -m unittest discover -s tests
```
