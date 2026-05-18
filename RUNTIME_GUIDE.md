# Runtime Guide

## Stable Entrypoints

- Web app: `streamlit run streamlit_app.py`
- CLI: `.venv/bin/python podcast_generator.py --help`
- Version: `V1.5.6`

旧 Gradio UI 和 PyInstaller 打包产物已下线。当前正式工作流只维护 Streamlit Web app 和 CLI。

## Recent Runtime Fixes

- Qwen 模型加载前会校验 `config.json`、`model.safetensors`、`speech_tokenizer/config.json`、`speech_tokenizer/model.safetensors`。缺少 speech tokenizer 权重会导致输出噪音，现在会直接报错并提示重新运行 `download_model.py`。
- `download_model.py` 只保留当前稳定 voice clone 模型下载入口，并在下载后做同一套完整性校验。
- 默认输出格式改为 MP3；`--format wav` 保留 WAV，`--format both` 同时保留 WAV 和 MP3。MP3 需要 `ffmpeg`。
- 响度标准化从硬截断改为 peak limiter，目标 -16 LUFS 后按比例降峰到安全 ceiling，降低 clipping 风险。
- 写入前后会做音频健康检查，提示空音频、低 RMS、低 peak、NaN/Inf、接近 clipping 等风险。
- 后端输出会保留真实 sample rate，拼接时校验片段采样率一致，避免错误采样率写盘。
- 依赖升级：`mlx==0.31.2`、`mlx-audio==0.4.3`、`mlx-lm==0.31.3`、`mlx-metal==0.31.2`、`transformers==5.8.1`、`huggingface_hub==1.15.0`。

## Models

- 主线模型：`models/Qwen3-TTS-12Hz-1.7B-Base-8bit`
- 轻量 fallback：`models/Qwen3-TTS-12Hz-0.6B-Base-8bit`
- 实验后端：`voxcpm`，需要单独安装 `requirements-voxcpm.txt`

不要把 `CustomVoice` 或 `VoiceDesign` 当作当前稳定主线；多人的声音切换通过 voice profile 的源音频完成。

完整 Qwen 模型目录必须包含：

```text
config.json
model.safetensors
speech_tokenizer/config.json
speech_tokenizer/model.safetensors
```

模型缺失或历史下载不完整时运行：

```bash
.venv/bin/python download_model.py 1
```

## Voice Profiles

公开版不内置任何个人声音素材。用户可以临时上传参考音频，也可以新建 profile 写入 `voices/profiles/`。
`audio_samples/` 和 `voices/profiles/` 里的真实人声素材默认被 Git 忽略。

生成时 checkpoint 会记录声音 profile 元数据；如果切换了源声音，不复用旧断点。

## URL Article Cleanup

Web app 的 URL 抽取会先寻找正文容器，再清理广告、订阅、分享、相关阅读、评论、cookie 提示等网页杂质。页面写入稿件区的是 `podcast_text`，不是原始网页全文。

## Output Formats

默认输出 MP3；需要无损中间文件或后期混音时，可以在 Web app 选择 WAV / WAV + MP3，或在 CLI 使用：

```bash
.venv/bin/python podcast_generator.py --file 稿件.txt --format mp3
.venv/bin/python podcast_generator.py --file 稿件.txt --format wav
.venv/bin/python podcast_generator.py --file 稿件.txt --format both
```

MP3 转换依赖 `ffmpeg`。

## Performance Notes

本机长文生成不是实时任务。Apple Silicon 上的耗时主要受模型大小、内存压力、文本分段和后台应用影响，历史参考如下：

| output duration | wall time | note |
|---|---:|---|
| ~30s | ~5 min | short preview |
| ~1-1.5 min | ~5-23 min | highly variable |
| ~10 min | ~2.5 hr | use fixed output name and resume |

长文生产建议接电运行、关闭重型后台任务，并固定 `--output` 以便 checkpoint 对齐。

## Generated Files

`outputs/`、`runtime/`、日志、cache、build/dist 都是可再生数据或本机状态。
清理项目时可以删除它们，但保留：

- `outputs/.gitkeep`
- `voices/.gitkeep`
- `voices/profiles/.gitkeep`
- `audio_samples/.gitkeep`
- `models/`
- `.venv/`

## Verification

```bash
.venv/bin/python -m py_compile streamlit_app.py voice_profiles.py voice_controls.py article_extractor.py podcast_generator.py
.venv/bin/python -m unittest discover -s tests
```
