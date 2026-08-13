# Magic Box

> 最后更新：2026-06-22

当前版本：V2.3.2

Magic Box 是 Apple Silicon 上的本地播客 TTS 工具，支持 **Qwen3-TTS** 和 **VoxCPM2** 两个 MLX 原生后端。

- `streamlit_app.py`：Web 生产台，支持 URL 正文抽取、长文切分、断点续跑、多人源声音 Profile、模型管理、生成实时进度。
- `podcast_generator.py`：CLI 长文生成，适合批处理和固定脚本。
- `skills/magic-box-tts/`：Codex/Hermes 友好的 CLI Skill，适合把长音频生成投递到另一台 Apple Silicon Mac。
- `scripts/bootstrap_mac.sh` / `scripts/sync_update.sh`：新 Mac 部署和后续 GitHub 同步更新入口。

> 本目录是唯一工作目录（当前 v2.3.2）。旧副本 `/Users/liuchang/Apprun/chenxi/Magic Box/`（v1.5.x）已于 2026-08-13 归档为 `/Users/liuchang/Apprun/chenxi/Magic Box.bak-20260813/`（内含旧 `champ` 声线与旧 8-bit 模型，如需可从此恢复）。

## 最近更新

### v2.3.2（2026-06-22 齿音压制 + 更慢默认语速）

- **Qwen 默认语速再降一档**：Web 端 Qwen 显示语速按 `0.82` 折算后传给生成器；显示 `1.00` 实际传入 `0.82`，显示 `0.95` 实际传入 `0.779`。Skill / Hermes wrapper 默认同样使用 `0.82`。
- **最终音频加保守 de-esser**：响度标准化后会对 4.5-9.5 kHz 的异常高频齿音做轻量压制，缓解 “si / 思思 / 丝丝” 一类破音，同时保留正常人声亮度。
- **CLI 默认更稳**：直接使用 `podcast_generator.py` 时默认 `--speed` 从 `1.00` 调整为 `0.90`。

测试覆盖增至 **122**，新增齿音压制回归测试。

### v2.3.1（2026-06-22 GitHub 部署 + Hermes 同步）

- **GitHub 作为统一发布源**：Web、CLI、Skill、部署脚本和文档都保留在仓库内。两台 Mac 都从同一分支拉取更新，私有模型、声音 Profile 和生成输出继续只留在本机。
- **Home Mac 一键部署/更新脚本**：新增 `scripts/bootstrap_mac.sh` 和 `scripts/sync_update.sh`，用于首次 clone、创建 `.venv`、安装依赖、安装 Skill symlink，以及后续 `git pull --ff-only` 同步更新。
- **Hermes 运行路径标准化**：`skills/magic-box-tts/scripts/magic_box_tts.py` 是远程/家里 Mac 的固定 CLI 入口，支持 `--print-command` 生成可投递给 Hermes 的命令。
- **进程生命周期修复**：Web 和 Skill 调用生成器时默认带 `--force-exit-after-run`，避免 MP3 已产出但底层 MLX/Python 子进程迟迟不退出。
- **全局防并发生成**：Web 端会检测任何仍在运行的 `podcast_generator.py`，不再只检查同一输出文件，避免两个长文生成任务抢 CPU/内存。

### v2.3.0（2026-06-22 子进程生成 + 常驻进度监控）

- **模型推理隔离到独立 Python 子进程**（`streamlit_app.py`）：Web 端不再在 Streamlit 主进程内直接加载 Qwen / VoxCPM 模型。底层 MLX 崩溃或被中断时，页面服务会继续保留，断点和质量报告仍可查看。
- **常驻生成监控**（`streamlit_app.py`）：切分预览上方会显示当前后台生成进程 PID、运行时间、CPU、已完成段数、segment 文件数、估算进度和预计剩余时间；页面会定时刷新监控，不再只依赖按钮点击后的临时状态块。
- **防重复启动**（`streamlit_app.py`）：有任何 `podcast_generator.py` 正在生成时，“生成播客音频”按钮会禁用，避免多个 Qwen / VoxCPM 子进程同时抢 CPU、内存或写 checkpoint。
- **默认断点续跑开启**：Web 端“从断点继续”默认打开。长文生成中断后，保持同一个输出文件名即可续跑已完成片段。
- **Qwen 默认语速更慢**：Web 端 Qwen 显示语速会按 `0.82` 折算后传给生成器；显示 `1.00` 实际传入 `0.82`，显示 `0.95` 实际传入 `0.779`。生成器仍会在 CLI 内按参考音频语速做二次校准，并把最终 `model_speed` 写入 checkpoint / quality report。
- **CLI 与 Web 参数统一**（`podcast_generator.py`）：CLI 增加 `--chunk-max-chars` 和 `--checkpoint-metadata-file`，Web 子进程调用可以完整传递切分上限、Profile 指纹、断点元数据和输出格式。

相关 UI / 子进程测试覆盖增至 **120**，包含子进程异常尾日志、进度快照、ETA 估算、进程时间解析、Qwen 显示语速折算和后台进程扫描。

### v2.2.0（2026-06-21 参考语速校准 + 项目清理）

- **参考语速校准**（`podcast_generator.py`）：Qwen clone 下 UI / CLI 的 `speed=1.00` 现在表示“尽量贴近参考音频语速”，生成前会根据 `reference_clean.wav` 的字/秒折算成实际模型 speed，并写入 `quality_report.json` 的 `generation.model_speed` 和 `generation.rate_calibration`。
- **默认预设回归自然速度**（`voice_controls.py`）：默认“自然播客”保持 `speed=1.00`、`temperature=0.85`、`chunk_max_chars=260`，避免默认参数把参考音频较慢的 Profile 播快。
- **运行产物清理**：`outputs/`、`runtime/`、`audio_samples/` 默认只保留 `.gitkeep`；缓存、`.DS_Store`、旧调试音频和临时上传文件均可再生，不应提交。
- **当前个人 Profile 保留在本机**：`voices/profiles/刘畅/` 是当前使用的本机声音 Profile，仍被 Git 忽略，不会推送到 GitHub。

测试增至 **113**，覆盖参考音频清洗、质量报告、质量门禁、Profile 保存清洗产物、UI 报告解析、默认参数范围、参考语速校准和模型路由。

### v2.1.0（2026-06-18 参考音频清洗 + 质量门禁）

针对“bf16 仍会生成断续 burst / 低能量停顿 / sample jump 爆音”的问题，生成链路从“生成后只提示 warning”升级为“生成前清洗参考音频、生成中保留证据、生成后写质量报告并阻断坏音频”：

- **参考音频自动清洗**（`utils.py:prepare_reference_audio_clip`）：源声音会先转成 24 kHz mono WAV，再自动裁出 3-8 秒连续、低停顿、RMS/peak 正常的人声片段，写入 `reference_clean.wav` 和 `reference_quality.json`。找不到合格片段时会失败并写明原因。
- **Profile 保存清洗产物**（`voice_profiles.py`）：保存声音 Profile 时保留 `reference_original.<ext>`，实际克隆使用 `reference_clean.wav`，并保存 `reference_quality.json` 便于复查。
- **生成前参考音频门禁**（`tts_backends.py:prepare_reference_audio`）：Qwen / VoxCPM clone 只接收清洗后的 WAV；参考音频太短、过静、停顿太长或 clipping 时会提前失败。
- **每段原始 chunk 保留证据**（`podcast_generator.py`）：默认把 `_seg_*.wav` 写到 `outputs/.<output_stem>_segments/`，方便定位是哪一段模型输出坏。
- **质量报告**（`quality_report.json`）：记录每段和最终音频的 `duration`、`rms`、`peak`、`low_energy_ratio`、`longest_low_energy_run`、`jump_count_gt_0_3`、`jump_count_gt_0_5`、`max_jump`。
- **质量门禁**（`audio_quality_issues`）：遇到明显爆音跳变、长低能量段、全静默、NaN/Inf、near clipping 等问题时标记 failed，不再把坏音频当成成功结果。
- **Web UI 展示质量报告**（`streamlit_app.py`）：生成后显示质量摘要、失败段编号、报告路径和 segments 文件夹；失败但已有证据时也会展示质量报告。
- **默认参数收敛**（`voice_controls.py`）：默认“自然播客”改为更稳的 `speed=1.00`、`temperature=0.85`、`chunk_max_chars=260`；Qwen clone 会把 `1.00` 按参考音频语速折算成模型 speed，高级参数旁增加高风险提示。

测试增至 **112+**，覆盖参考音频清洗、质量报告、质量门禁、Profile 保存清洗产物、UI 报告解析和默认参数范围。

### v2.0.1（2026-06-17 音频质量二审）

针对音频生成链路做了二次深度审计与修复，覆盖清晰度、音量平衡、无杂音、格式兼容性、播放流畅度、内容完整性六个维度：

- **VoxCPM2 采样率探测改为显式失败**（`tts_backends.py:_voxcpm_sample_rate`）：原代码在无法识别模型采样率时静默 fallback 到 24 kHz,会导致 48 kHz 的 VoxCPM2 输出被错标为 24 kHz,播放速度减半。现改为 `raise ValueError`,并给出"补充探测路径 / 显式指定"的可执行修复指引。
- **VoxCPM2 空迭代器防护**（`generate_voxcpm_chunk`）：`next(result)` 在空生成器上抛 `StopIteration`,现与 Qwen 路径一致包 `try/except` 并抛出包含 chunk 文本的 `RuntimeError`。
- **VoxCPM2 模型文件校验**（`_validate_voxcpm_model_files`）：镜像 Qwen 的 `_validate_qwen_model_files`,在 `load_model` 之前先确认本地 snapshot 含 `config.json`,否则给出可读错误（避免 mlx_audio 内部报晦涩异常）。
- **生产链路采样率一致性**（`podcast_generator.py:generate_podcast`）：在主循环中跟踪 `expected_sample_rate`,跨段采样率不一致时立即 `raise ValueError`,与 `benchmark_generate_case` 行为对齐。原来的"最后再读盘检查"会让坏段先污染 checkpoint。
- **健康检查阈值与限幅对齐**（`utils.py:check_audio_health`）：`high_peak_threshold` 由 0.98 下调到 0.95,与 `limit_audio_peak(ceiling=0.95)` 对齐。原阈值 0.98 让限幅器失效的样本（peak 介于 0.95–0.98）能蒙混过关,只在真正爆音时才告警,已经太晚。
- **短段 crossfade 智能收缩**（`utils.py:crossfade_concat`）：当段长度 < 请求的 fade 窗口时,改为 `fade = min(requested, len(out), len(s))`,仅在两边都 < `min_fade_samples`(默认 8) 时才退化到硬拼接。原来的"任一不足即硬切"会在 trim_silence 后段过短时产生 click,与之前已修复的咔嗒声属于同一类问题。
- **单 chunk 音频健康检查**（`podcast_generator.py`）：每段生成后立即调 `check_audio_health`,NaN/Inf / 全静默 / peak clipping / 低 RMS 会在出问题时立刻打 `[audio warning]`,而不是等到最终 mix 才发现无法定位。
- **`trim_silence` 全静默输入保护**（`utils.py:trim_silence`）：当 chunk 长度 ≤ head+tail 强制裁剪量时,原来返回空数组(导致静默失败),现改为打 warning 并保留原音频,避免整段被吞掉。
- **VoxCPM2 ref_text 隐式约定清理**（`generate_voxcpm_chunk`）：`ref_text.strip() != "."` 这种"句点表示无 ref"隐式约定容易遗漏,简化为 `if ref_text and ref_text.strip():`。
- **ffmpeg 缺失行为一致化**（`utils.py:convert_audio_if_needed`）：原来静默返回 `None` 让上游笼统报"参考音频转换失败",现与 `convert_wav_to_mp3` 一样 `raise RuntimeError("ffmpeg not found...")`,让用户立刻知道要 `brew install ffmpeg`。
- **`as_tts_result` 采样率校验**（`tts_backends.py:as_tts_result`）：对 sample_rate 做 `int(...) > 0` 校验,无效时 `raise ValueError` 而不是写入错误元数据。

测试从 55 增至 **61**（新增 1 个 VoxCPM2 错误行为测试 + 跨模块回归覆盖）。

### v2.0.0（2026-06-17 合并）

- **VoxCPM2 后端全面迁移到 MLX 原生**：不再需要 PyTorch / torchaudio / einops。
  默认模型 `mlx-community/VoxCPM2-8bit`，另可选 bf16 / 4-bit 量化版。
  VoxCPM2 支持 48 kHz 高保真输出、30 语言 + 9 方言、Voice Design + Clone + Ultimate Clone。
- **Qwen3-TTS bf16 模型**：补充了 5 个 bf16 变体，质量分数 82-100。
- **模型注册表扩展到 13 个模型**：5 个 Qwen 8-bit + 5 个 Qwen bf16 + 3 个 VoxCPM2 MLX。
- **`_safe_remove` 改为 `safe_remove`**，返回 `bool` 表示是否实际删除了文件。
- **`save_checkpoint` 返回 `bool`**，让 CLI 和 Web 端能感知磁盘满或权限不足等失败。
- **`model_manager.get_models_dir` 回退时打印 warning 日志**。
- **`VoiceProfile.fingerprint` 使用 ref_audio 内容 SHA-256**，不再嵌入绝对路径，跨机器可复用断点。
- **上传文件大小限制 50 MB**，runtime 目录总量限制 500 MB，超出时自动清理最旧文件。
- **CSS 抽出到 `app_styles.css`**，Streamlit 启动时按需注入。
- **`logging` 替代部分 `print`**，CLI 输出到 stderr，Streamlit 以 WARNING 级别输出。
- **`pyproject.toml` + `Makefile`** 标准化项目管理。
- **Web 端生成链路加进度回调**（`progress_callback`）：每段完成会刷新到 `st.progress` 进度条，方便观察长文生成状态。
- **V1.5.7 → V2.0.0 合并**：移除 V1.5.7 严格的 `validate_generated_audio`（chunk 1 冷启动被误杀），V2.0.0 用更轻的 `check_audio_health` 只警告不阻断。
- **Chatterbox 后端下线**：原 V1.5.7 的 Chatterbox 实验后端和 `requirements-chatterbox.txt` 已删除（V2.0.0 改用 VoxCPM2）。

### 历史修复

- 补齐并强校验 Qwen 模型目录里的 `speech_tokenizer/config.json` 和 `speech_tokenizer/model.safetensors`。
- 默认输出改为 MP3，CLI 与 Web 都支持 `mp3`、`wav`、`both`。
- 响度标准化后使用 peak limiter 按比例降峰。
- 生成链路保留后端返回的真实 sample rate。

## 快速启动

```bash
cd /Users/liuchang/Workspaces/Magic\ Box
./.venv/bin/python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8507
```

浏览器打开 `http://127.0.0.1:8507/`。

不要直接运行：

```bash
python3 streamlit_app.py
```

也不要使用系统 Python 的 `python3 -m streamlit`。Qwen / VoxCPM 后端必须使用项目 `.venv` 中的 `mlx-audio==0.4.3`；旧版 `mlx-audio` 会让 Qwen speech tokenizer encoder 不可用，生成音频出现固定空洞和断续。

或使用 Makefile：

```bash
cd /Users/liuchang/Workspaces/Magic\ Box
make run-streamlit
```

## GitHub 部署与同步

这个仓库同时服务两种运行方式：

- 当前 Mac 跑 Web 生产台：用 Streamlit 交互、管理声音 Profile、试听和下载输出。
- 家里 M2 / Hermes Mac 跑长任务：用 Skill CLI 接收文本并在那台机器本地生成 MP3，当前电脑不用一直开着。

GitHub 应只保存可复用能力：源码、测试、Skill、部署脚本和文档。以下内容默认不提交：`models/`、`.venv/`、`voices/profiles/`、`outputs/`、`runtime/`、个人参考音频和生成结果。

### 新 Mac 首次部署

在目标 Mac 上执行：

```bash
curl -fsSL https://raw.githubusercontent.com/ykopp/magic-box/codex/interactive-web-app/scripts/bootstrap_mac.sh | bash
```

如需自定义目录或分支：

```bash
MAGIC_BOX_DIR="$HOME/Workspaces/Magic Box" \
MAGIC_BOX_BRANCH="codex/interactive-web-app" \
curl -fsSL https://raw.githubusercontent.com/ykopp/magic-box/codex/interactive-web-app/scripts/bootstrap_mac.sh | bash
```

部署脚本会 clone 仓库、切到指定分支、创建 `.venv`、安装 Python 依赖、安装 `ffmpeg`（如果本机有 Homebrew），并把 `skills/magic-box-tts` 链接到 `~/.codex/skills/`。模型文件和个人声音 Profile 仍需手动复制，因为它们体积大且是私有数据。

### 后续同步更新

每次这边更新并推送 GitHub 后，另一台 Mac 执行：

```bash
cd "$HOME/Workspaces/Magic Box"
make sync-update
```

或直接：

```bash
bash scripts/sync_update.sh
```

同步脚本会 `git pull --ff-only`、刷新 `.venv` 依赖，并重新安装 Skill symlink。它不会删除模型、声音 Profile 或输出文件；如果代码目录有未提交改动，会先停止，避免覆盖本地修改。

## Web 生产台

Streamlit 页面用于正式长文播客生产：

- 粘贴、上传 TXT（UTF-8 / GB18030，上限 50 MB），或从 URL 抽取文章正文，清除广告、订阅、分享等网页杂质后整理成播客稿。
- 在 Qwen 和 VoxCPM2 两个后端间切换。Qwen 支持 speed / temperature / 模型路由；VoxCPM2 输出 48 kHz 高保真音频。
- 选择已保存的"克隆声音 Profile"，或临时上传参考音频。
- 使用"表达预设"（稳定清晰 / 自然播客 / 热情开场 / 沉稳叙事 / 快速草稿）和高级微调控制语速、随机性和单段字符上限；Web 端 Qwen 显示语速会先按 `0.82` 折算，再交给生成器按参考音频语速校准。
- 输出 WAV、MP3，或同时输出 WAV + MP3。
- "从断点继续"默认开启，中断任务可按同一输出文件名续跑。断点文件 fingerprint 已改为内容哈希，跨机器可复用。
- 生成过程中页面会显示常驻进度监控：子进程 PID、运行时间、CPU、当前段、已完成段、segment 文件数、估算进度和预计剩余时间。
- 生成后会读取 `quality_report.json`，显示质量摘要、失败段编号和 `_seg_*.wav` 证据目录。
- 生成后在页面底部输出库试听和下载所选格式。

### 后端选择

| 后端 | 模型 | 特点 |
|---|---|---|
| **qwen** | Qwen3-TTS 0.6B/1.7B (8-bit / bf16) | speed / temperature 控制，10 语言，3 秒 Clone，Voice Design |
| **voxcpm** | VoxCPM2 (8-bit / bf16 / 4-bit) | 48 kHz，30 语言 + 9 方言，Voice Design + Clone |

### 声音 Profile

公开版不会内置任何个人声音。用户可以临时上传参考音频，或在"管理声音 Profile"里保存自己的样本。

新保存的 profile 写入：

```text
voices/profiles/<profile_id>/
  metadata.json
  reference_original.<ext>
  reference_clean.wav
  reference_quality.json
  transcript.txt
```

这些个人声音数据默认不提交 Git。

## CLI 生成

```bash
python3 podcast_generator.py \
  --file 稿件.txt \
  --ref-audio 我的声音.m4a \
  --ref-text "参考音频里具体说的文字内容" \
  --output outputs/my_episode.mp3 \
  --format both \
  --backend qwen
```

关键参数：

- `--file` / `--text`：输入稿件。
- `--ref-audio`：用于克隆的源声音。
- `--ref-text`：源声音里实际说的文字，必须尽量逐字一致。
- `--output`：输出路径；默认自动生成 `.mp3` 文件名。
- `--format`：输出格式，支持 `wav`、`mp3`、`both`，默认 `mp3`。
- `--resume`：从已有 `.ckpt` 断点继续。
- `--chunk-max-chars`：每段最大字符数；Web 端会把滑块值原样传入 CLI。
- `--backend`：`qwen`（默认）或 `voxcpm`。
- `--speed` / `--temperature`：控制语速和表达随机性（仅 Qwen 后端）；Qwen clone 会把 `--speed 1.0` 按参考音频语速自动校准。
- `--no-normalise`：跳过响度标准化。

生成默认会在输出目录写入 `quality_report.json`，并保留每段原始 chunk 到 `outputs/.<output_stem>_segments/`。如果质量门禁失败，输出文件不会被当作成功结果返回，但报告和片段证据会保留。

## Codex Skill / Hermes 投递

仓库内置 `skills/magic-box-tts/`，用于把 Magic Box 作为 CLI-first 的音频生成 Skill 使用。它适合在另一台已部署 Magic Box 和 Hermes 的 Mac 上接收文本、调用本机模型生成 MP3，避免当前电脑长时间占用。

本机直接生成：

```bash
python3 skills/magic-box-tts/scripts/magic_box_tts.py \
  --file story.txt \
  --profile 刘畅 \
  --output outputs/story.mp3
```

生成一条适合投递给 Hermes / 远程 Mac 执行的命令：

```bash
python3 skills/magic-box-tts/scripts/magic_box_tts.py \
  --text "要朗读的正文" \
  --profile 刘畅 \
  --output outputs/story.mp3 \
  --print-command
```

目标 Mac 需要提前准备好 repo、`.venv`、模型文件、`ffmpeg` 和对应声音 Profile。详细说明见 `skills/magic-box-tts/references/home-mac-setup.md`。

## 模型与依赖

### 模型管理

使用 Model Manager 查看和下载模型：

```bash
# CLI 路由工具
python3 model_route_cli.py --task clone --objective quality
python3 model_route_cli.py --auto-download --task clone

# Makefile
make route ARGS="--task clone"
```

### 可用模型（13 个）

**Qwen3-TTS 8-bit**（推荐日常使用）：

| 模型 | 任务 | 大小 | Q/S 分数 |
|---|---|---|---|
| Qwen3-TTS-12Hz-1.7B-Base-8bit | Clone | 2.9 GB | Q98/S62 |
| Qwen3-TTS-12Hz-0.6B-Base-8bit | Clone | 1.9 GB | Q78/S95 |
| Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit | Custom Voice | 3.1 GB | Q100/S61 |
| Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit | Custom Voice | 2.0 GB | Q82/S94 |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit | Voice Design | 3.1 GB | Q99/S61 |

**Qwen3-TTS bf16**（最高质量，适合录音级场景）：

| 模型 | 任务 | 大小 | Q/S 分数 |
|---|---|---|---|
| Qwen3-TTS-12Hz-1.7B-Base-bf16 | Clone | 5.8 GB | Q100/S50 |
| Qwen3-TTS-12Hz-0.6B-Base-bf16 | Clone | 3.8 GB | Q82/S80 |
| Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 | Custom Voice | 5.8 GB | Q100/S50 |
| Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16 | Custom Voice | 3.8 GB | Q85/S80 |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 | Voice Design | 5.8 GB | Q100/S50 |

**VoxCPM2 MLX**（48 kHz 高保真）：

| 模型 | 大小 | Q/S 分数 | 说明 |
|---|---|---|---|
| VoxCPM2-8bit | 2.2 GB | Q92/S70 | 推荐，质量/速度平衡 |
| VoxCPM2-bf16 | 3.8 GB | Q95/S55 | 最高音质 |
| VoxCPM2-4bit | 1.88 GB | Q85/S90 | 最快，62% 更小 |

### 下载模型

```bash
# 通过 Model Manager
python3 model_route_cli.py --auto-download --task clone

# 或手动下载
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base-8bit

huggingface-cli download mlx-community/VoxCPM2-8bit \
  --local-dir models/VoxCPM2-8bit
```

## 性能参考

在 Apple Silicon 本地运行时，Qwen 生成速度会明显受内存压力、分段长度、模型大小和后台应用影响。历史实测大致范围：

| 输出音频时长 | 生成耗时 | 说明 |
|---|---:|---|
| 30 秒 | 约 5 分钟 | 短片段试听 |
| 1-1.5 分钟 | 约 5-23 分钟 | 波动较大，取决于分段和内存状态 |
| 10 分钟 | 约 2.5 小时 | 建议固定输出文件名并开启断点续跑 |

VoxCPM2 MLX 8-bit 在 M2 Pro 上实测约 1.2x RTF（48 kHz），4-bit 约 0.8x RTF。长文建议接电运行，减少其它重型应用，优先使用 `--resume` 保护进度。

## 音频质量保护

v2.1.0 起，生成链路在以下节点强制校验，避免无声失败、断续 burst、尖刺爆音或坏音频静默通过：

| 阶段 | 保护机制 | 触发条件 | 行为 |
|---|---|---|---|
| 参考音频清洗 | `prepare_reference_audio_clip` | 参考音频 < 3s / 过静 / 长停顿 / clipping / 找不到连续人声 | 写 `reference_quality.json` 并提前失败 |
| Profile 保存 | `save_profile` | 上传源声音 | 保存 `reference_original.<ext>`，实际使用 `reference_clean.wav` |
| 模型加载 | `_validate_voxcpm_model_files` / `_validate_qwen_model_files` | 本地 snapshot 缺关键文件 | `FileNotFoundError` 含可执行指引 |
| 采样率探测 | `_voxcpm_sample_rate` / `as_tts_result` | 无法识别 sample rate 或 ≤ 0 | `ValueError` 立即停止,绝不静默 fallback |
| 单段生成 | `check_audio_health(per chunk)` | NaN/Inf / 全静默 / peak ≥ 0.95 / 低 RMS | `[audio warning]` 立即打印,定位到具体段号 |
| 单段证据 | `_seg_*.wav` | 每段 TTS 生成完成 | 默认保留到 `outputs/.<output_stem>_segments/` |
| 质量报告 | `quality_report.json` | 每次生成 | 记录 chunk/final 的 RMS、peak、低能量、sample jumps 等指标 |
| 质量门禁 | `audio_quality_issues` | 爆音跳变、长低能量段、全静默、NaN/Inf、near clipping | 标记 failed，保留报告和 segment 证据 |
| 跨段一致性 | `expected_sample_rate` 跟踪 | 后端中途切换采样率 | `ValueError` 立即停止,避免播放变速 |
| 静默裁剪 | `trim_silence` head/tail 守卫 | chunk 短于 head+tail 请求量 | warning + 保留原音频,避免整段被吞 |
| 交叉淡化 | `crossfade_concat` 动态 fade 收缩 | 段长度 < fade 窗口 | 自动收缩到 `min(requested, len(out), len(s))`,仅 sub-ms 退化到硬切 |
| 响度限幅 | `limit_audio_peak(ceiling=0.95)` + `check_audio_health(threshold=0.95)` | peak ≥ 0.95 | tanh 软限,失败时立刻告警 |
| 齿音压制 | `reduce_sibilance` | 4.5-9.5 kHz 高频齿音占比异常 | 轻量压制 “si/思思/丝丝” 一类尖锐破音 |
| ffmpeg 依赖 | `convert_audio_if_needed` / `convert_wav_to_mp3` | 缺少 ffmpeg | 两者一致 `raise RuntimeError("brew install ffmpeg")` |

历史音频问题已经收敛进当前生成链路：优先使用 Qwen bf16/Base 克隆模型、保留真实 sample rate、拼接前后做健康检查、以 -23 LUFS 做保守响度标准化，并在质量报告中记录参考语速校准、sample jumps、低能量段和最终交付文件指标。

## 项目结构

```text
streamlit_app.py        # Web 生产台
podcast_generator.py    # CLI 长文生成
tts_backends.py         # Qwen / VoxCPM2 后端封装
model_manager.py        # 模型下载、更新、路由
article_extractor.py    # URL 正文抽取
voice_profiles.py       # 多人源声音 Profile 管理
voice_controls.py       # 表达预设和节奏优化
utils.py                # 音频转换、checkpoint、切分等工具
app_styles.css          # Streamlit 暗色主题 CSS
Makefile                # 日常命令 (install/test/lint/clean)
pyproject.toml          # 项目元数据和工具配置
scripts/                # Mac 部署、同步更新脚本
skills/magic-box-tts/   # Codex/Hermes CLI Skill
model_route_cli.py      # 模型路由 CLI 工具
benchmark_tts_backends.py # TTS 后端基准对比
download_model.py       # 手动模型下载脚本
audio_samples/          # 本机可选参考音频
voices/profiles/        # 用户保存的声音 Profile
outputs/                # 生成结果、quality_report.json、.<output>_segments/
models/                 # 本地模型
runtime/                # 临时上传文件
tests/                  # 单元测试 (122 个)
使用指南.md             # 中文使用指南（保留自 V1.5.7）
项目交接文档.md         # 项目交接文档（保留自 V1.5.7）
```

## 清理边界

可以清理：

- `outputs/` 下除 `.gitkeep` 之外的生成结果。
- `audio_samples/` 下除 `.gitkeep` 之外的个人参考音频。
- `voices/profiles/` 下不用的个人声音 Profile；正在使用的 Profile 应保留在本机，且默认不会提交到 Git。
- `runtime/`、缓存、日志、`.DS_Store`、打包产物。

不要清理：

- `models/`：本机运行依赖。
- `.venv/`：本机 Python 环境。
- `audio_samples/.gitkeep`、`voices/.gitkeep`、`voices/profiles/.gitkeep`：保留目录结构。

## 验证

```bash
# 语法检查
make lint
# 或者
python3 -m py_compile streamlit_app.py voice_profiles.py voice_controls.py article_extractor.py podcast_generator.py tts_backends.py model_manager.py utils.py

# 运行测试
make test
# 或者
python3 -m pytest tests/
```

---

## Star History

[![Star History Chart](https://api.star-history.com/chart?repos=ykopp/magic-box&type=date&legend=top-left)](https://www.star-history.com/?repos=ykopp%2Fmagic-box&type=date&legend=top-left)
