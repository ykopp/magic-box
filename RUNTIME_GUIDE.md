# Runtime Guide

## Current Decision

- 主线后端: `qwen`
- 主模型: `models/Qwen3-TTS-12Hz-1.7B-Base-8bit`
- 速度 fallback: `models/Qwen3-TTS-12Hz-0.6B-Base-8bit`
- 实验后端: `voxcpm`
- 当前不作为主线: `0.6B-CustomVoice`、`podcast-tts/Kokoro`

## Recommended Presets

按当前用户场景，推荐把模型分成三类用途，而不是频繁替换主线：

| scene | preferred model | goal | suggested starting params |
|---|---|---|---|
| 正文播客 | `Qwen3-TTS-12Hz-1.7B-Base-8bit` | 像本人、稳定、长文可用 | `temperature=1.0`, `speed=1.0`, `--no-normalise` |
| 开场 / 结尾 / 预告 | `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | 更强起伏和播客感 | `temperature=1.1`, `speed=1.05`, `--no-normalise` |
| 风格化实验 | `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | 片头包装、角色化旁白 | `temperature=1.1`, `speed=1.0`, `--no-normalise` |

这些是工作流预设，不是新的 benchmark 结论。真正切换前，仍应以试听结果为准。

## Switching Rules

- 如果正文已经够像本人，不要为了“更有情绪”直接把整期切到 `CustomVoice`
- 如果问题只是偏平，先升级参考音频情绪，再微调 `temperature`
- 如果开场、结尾、预告需要更抓人，再优先试 `1.7B-CustomVoice`
- 如果目标已经不是“像你本人”，而是明显风格化表达，再试 `VoiceDesign`
- `VoiceDesign` 默认不回灌到正文主线

可执行顺序：

1. 主体内容先用 `1.7B-Base`
2. 短片段情绪不足时，先试更有表现力的参考音频
3. 再把 `temperature` 从 `1.0` 提到 `1.1`
4. 只有短片段仍然不够抓人时，才切 `1.7B-CustomVoice`
5. 只有做风格化包装时，才切 `1.7B-VoiceDesign`

## Why

`qwen3-tts-apple-silicon` 是面向 Apple Silicon 本地工作流优化的主链路，已经覆盖：

- 长文自动切段
- checkpoint / resume
- 参考音频转换
- 可选响度标准化
- Web UI
- 模型路由

`VoxCPM2` 的定位是能力更高的实验后端。它不替换当前默认链路，只作为独立 backend 进入 A/B 试听和能力验证。

## Backend Matrix

| backend | default model | task support | routing | intended usage |
|---|---|---|---|---|
| `qwen` | `Qwen3-TTS-12Hz-1.7B-Base-8bit` | `clone`, `custom`, `design` | yes | 日常本机生产 |
| `voxcpm` | `openbmb/VoxCPM2` | `clone`, `design` | no | 隔离实验 / 对比试听 |

## CLI

### Qwen mainline

```bash
python3 podcast_generator.py \
  --backend qwen \
  --file 稿件.txt \
  --ref-audio 我的声音.m4a \
  --ref-text "参考音频对应文本"
```

### Qwen custom intro / outro

```bash
python3 podcast_generator.py \
  --backend qwen \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
  --file 开场文案.txt \
  --ref-audio 我的激情开场.m4a \
  --ref-text "参考音频对应文本" \
  --temperature 1.1 \
  --speed 1.05 \
  --no-normalise
```

### Qwen voice design experiment

```bash
python3 podcast_generator.py \
  --backend qwen \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit \
  --text "今晚，我们来聊一首适合深夜独处时反复循环的歌。" \
  --temperature 1.1 \
  --speed 1.0 \
  --no-normalise
```

### VoxCPM experiment

VoxCPM 依赖与 Qwen/MLX 主线隔离。只有明确要跑 `backend=voxcpm` 时，才安装：

```bash
python3 -m venv .venv-voxcpm
source .venv-voxcpm/bin/activate
pip install -r requirements-voxcpm.txt
```

```bash
python3 podcast_generator.py \
  --backend voxcpm \
  --model openbmb/VoxCPM2 \
  --file 稿件.txt \
  --ref-audio 我的声音.m4a \
  --ref-text "参考音频对应文本"
```

## Web UI

启动后可在 `BACKEND` 下拉框中切换：

```bash
python3 web_interface.py
```

规则：

- `Qwen Mainline` 保持现有工作流和模型路由
- `VoxCPM Experimental` 不参与 Qwen 路由，且不支持 `Custom Voice`

## Benchmark

统一 benchmark 入口：

```bash
python3 benchmark_tts_backends.py \
  --ref-audio audio_samples/龙湖安置小区\ 2.m4a \
  --ref-text "大家好，今天是个好日子。很高兴能和大家分享这些内容，希望对你们有所帮助。让我们开始吧。大家好，今天是个好日子。很高兴能和大家分享这些内容，希望对你们有所帮助。让我们开始吧。"
```

输出目录：

- `outputs/benchmarks/benchmark_results.json`
- `outputs/benchmarks/benchmark_results.md`
- `outputs/benchmarks/*.wav`

## Acceptance Rules

- 如果 `VoxCPM` 在本机无法稳定跑完整流程，则只保留为远端候选
- 如果 `VoxCPM` 音质更强但平台成本更高，则只保留为实验候选
- 只有当其本机稳定性和工作流贴合度不低于 `qwen` 主线时，才允许进入替换评估
