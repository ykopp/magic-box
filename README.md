# Qwen3-TTS 个人克隆播客生成器 (Apple Silicon 优化版)

> **最近更新于：2026-03-23 20:25**

本项目是基于 MLX 框架深度优化的 Qwen3-TTS 工具，专为 Apple Silicon (M系列芯片) 打造。

根据您的工作流需求（**单人文字转克隆声音 -> 输出独立的高质量 WAV -> 导入 Audition 自己做后期混音**），本项目已去除冗余的多人对话功能，修复了所有相关 Bug（临时文件残留、并发锁问题等），并大幅提升了长音频生成的稳定性和效率。

---

## 🚀 核心工作流：一键生成克隆音频

这是您最常用的脚本，专为**长篇文稿**设计，具有**断点续生成**和**自动清理**的能力。

**环境激活：**
开机或新开终端后，请务必先激活运行环境：

```bash
cd /Users/liuchang/Apprun/chenxi/qwen3-tts-apple-silicon
source .venv/bin/activate
```

如果你不想依赖 shell 激活，也可以直接使用项目解释器：

```bash
.venv/bin/python podcast_generator.py --help
```

### 标准生成命令
将您写好的稿件（如 `稿件.txt`）和一段您的声音样本（如 `我的声音.m4a`）放在电脑任意位置，然后运行：

```bash
python3 podcast_generator.py \
  --file 稿件.txt \
  --ref-audio 我的声音.m4a \
  --ref-text "参考音频里具体说的文字内容" \
  --output outputs/my_episode.wav \
  --no-normalise
```

**参数说明：**
* `--file`: 您的长篇文稿文件路径（支持自动按句号、感叹号安全切分段落）。
* `--ref-audio`: 您本人的声音样本（3-10秒即可，周围不要有明显噪音）。
* `--ref-text`: **必须**与参考音频中说到的话完全一致，这能大幅提高克隆相似度和稳定性。
* `--no-normalise`: **强烈建议加上此参数**。因为您后续会在 Audition 中自己做混音和音量均衡，加上此参数会输出原始响度，避免自动标准化带来的额外增益。

生成的最终结果将存放在 `outputs/` 文件夹中。长文建议固定 `--output`，这样断点文件会稳定对应到同一个 `.ckpt`。

---

## 🛠 进阶技巧：断点续传（防奔溃）

如果您的文稿非常长（例如 10,000 字），生成到一半时您不小心按了 `Ctrl+C` 退出了，或者遭遇了罕见错误，**不用从头再来**！

程序会在同目录下自动维护一个 `.ckpt` 断点文件。您只需要在上一次的命令后加上 `--resume` 即可从上次断掉的段落继续生成：

```bash
python3 podcast_generator.py \
  --file 稿件.txt \
  --ref-audio 我的声音.m4a \
  --ref-text "参考音频里具体说的文字内容" \
  --output outputs/my_episode.wav \
  --no-normalise \
  --resume
```

---

## 🎧 音乐播客进阶：发音准确度与“人味”控制

作为分享中英/小语种音乐的播客，您非常需要**准确的发音**和**饱满的情绪（人味）**。在 Qwen3-TTS 架构下，决定最终情绪的最核心因素是**参考音频的质量**，其次是**解码参数**。

请按照以下 4 点优化您的输入，可以达到绝非机器人的播客效果：

1. **录制“戏精”级的参考音频 (最重要！)**
   模型的情绪和语调是**零样本克隆**自您的 `ref-audio`。如果您录参考音频时像在平淡地读课文，克隆出来的声音就会有“机器感”。
   * **建议：** 录制一段 5~10 秒的极富激情的播客开场白（例如：“哈喽大家好！欢迎收听本期播客，今天我要给大家分享一首绝赞的独立音乐！”）。模型会完美继承这种“人味”和起伏。

2. **利用标点符号控制呼吸和节奏**
   排版对节奏影响极大。不要用大段连续的文字，多用逗号 `,`、省略号 `...` 和波浪号 `~` 控制停顿，用感叹号 `!` 增加冲击力。
   * *示例：* `这首歌... 哇，第一次听的时候，真的~ 绝了！`

3. **提高 `--temperature` (温度) 增加灵动性**
   默认温度是 1.0。如果您觉得太平铺直叙，可以加上 `--temperature 1.1` 或 `1.2`。这会让发音的音调有更多随机起伏，大大增加“人味”（但若过高可能会出现吃字或念错，1.1 是个好起点）。

   ```bash
   python3 podcast_generator.py --file 稿件.txt --ref-audio 我的激情开场.m4a --ref-text "哈喽大家好..." --no-normalise --temperature 1.1 --speed 1.05
   ```

4. **双语与小语种的发音准确度**
   Qwen3-TTS 原生支持中英混排，发音普遍非常准确。但遇到冷门乐队名或小语种（如法语、西班牙语歌名）时，如果出现发音奇怪的现象，**请在文稿中直接使用空耳/谐音注音**，或者将其翻译为最接近的英语发音拼写。这是目前保证专有名词零失误的最有效制作手段。

---

## 🎛 Streamlit 长文播客生成台

v1.5 的 Streamlit 版本面向正式长文播客生产：提供专业生产控制台、URL 文章提取、声音预设与“人味”控制、节奏优化预览、断点续生成，并在生成后直接试听/下载 WAV。它也支持粘贴/上传 `.txt` 稿件、上传参考音频、使用默认参考音频、选择后端和模型、预览切分段落。

```bash
cd /Users/liuchang/Apprun/chenxi/qwen3-tts-apple-silicon
source .venv/bin/activate
streamlit run streamlit_app.py
```

生成结果仍然保存到 `outputs/`，页面底部会列出最近的 WAV 文件，方便回听和下载。这个界面面向正式长文播客生产；下面的 Gradio Web UI 更适合短句试听、模型路由调试和参数试验。

**断点续生成说明：** 长文生成会先切成多个 chunk；每个完成的 chunk 会保存为临时音频片段，并记录到对应的 `.ckpt` 文件。若任务中断，请保持同一个输出文件名/文件 stem，并打开 `Resume from checkpoint`，程序会跳过已完成片段继续生成，而不是从头开始。任务成功完成后，checkpoint 和临时片段会自动清理。

## 🌐 可视化界面 (Web UI)

如果您更喜欢通过浏览器进行可视化操作，项目也提供了一个功能完整的网页版界面，支持模型自动路由和参数实时调节。

**启动 Web UI：**

```bash
python3 web_interface.py
```

启动后，在浏览器访问：`http://127.0.0.1:7860`

**Web 版特点：**
* **可视化调节：** 直观拖动语速、温度等参数。
* **模型自动分发：** 能够根据你选择的任务（克隆、设计、预设）自动加载最合适的模型（1.7B 或 0.6B）。
* **实时预览：** 适合用来反复调试一段话的最完美读法，调试好后再去跑 `podcast_generator.py` 生成长音频。

## 🔀 Backend Strategy

当前默认后端仍然是 `qwen`。  
如果你要做隔离实验，可以显式切换到 `voxcpm`，但它不会替换主线。
VoxCPM 的 PyTorch 依赖不放进默认 Qwen/MLX 环境；需要实验时用 `requirements-voxcpm.txt` 单独安装。

## 🎙 推荐工作流：按场景选模型

面向当前这套工作流，建议不要追求“一支模型包打天下”，而是按内容段落分工：

- 正文播客：优先用 `Qwen3-TTS-12Hz-1.7B-Base-8bit`
- 开场 / 结尾 / 预告：优先试 `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`
- 风格化实验：用 `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`

这套建议只定义使用分工，不代表这里对三支模型做了新的性能结论。核心原则仍然是：

- 先保证“像你本人”和长文稳定性
- 再单独给高情绪片段更强的风格控制
- 风格化实验不要反向污染正文主线

### 1. 正文播客

正文、长文、整期主内容，默认从 `1.7B-Base` 开始。

推荐起点：

```bash
python3 podcast_generator.py \
  --backend qwen \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --file 稿件.txt \
  --ref-audio 我的声音.m4a \
  --ref-text "参考音频对应文本" \
  --temperature 1.0 \
  --speed 1.0 \
  --no-normalise
```

适合切回 `1.7B-Base` 的情况：

- 你最在意“像你本人”
- 文稿较长，需要稳定连续输出
- 开场已经单独生成，正文只需要自然、清楚、稳定

### 2. 开场 / 结尾 / 预告

这些片段更看重起伏、能量和播客感，优先试 `1.7B-CustomVoice`。

推荐起点：

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

什么时候优先试 `CustomVoice`：

- 你觉得 `Base` 太平、太像念稿
- 这段时长短，允许多试几次
- 这段承担“抓人”的功能，比如片头、预告、收尾 CTA

什么时候不要强行继续用 `CustomVoice`：

- 你开始明显感觉“不像你本人”
- 文稿变长，稳定性和一致性更重要
- 只是想把正文稍微提一点情绪，这时先调参考音频和参数，不要先换整段模型

### 3. VoiceDesign 风格化实验

`VoiceDesign` 适合做风格化、角色化、片头包装或特殊栏目实验，不建议默认混入正文主线。

推荐起点：

```bash
python3 podcast_generator.py \
  --backend qwen \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit \
  --text "今晚，我们来聊一首适合深夜独处时反复循环的歌。" \
  --temperature 1.1 \
  --speed 1.0 \
  --no-normalise
```

适合切到 `VoiceDesign` 的情况：

- 你想做明显不同于本人日常口播的包装段
- 你在做预告、栏目片头、特殊旁白
- 你接受这是实验用途，而不是“最像本人”的路径

### 4. 切换原则

建议按这个顺序决策：

1. 先用 `1.7B-Base` 跑正文
2. 如果问题只是“太平”，先换更有情绪的参考音频，再把 `temperature` 提到 `1.1`
3. 只有在短片段仍然不够抓人时，再切到 `1.7B-CustomVoice`
4. 只有当目标变成“风格化包装”而不是“像你本人”，才切到 `VoiceDesign`

一个实用判断：

- 想要“像我本人”：先守在 `Base`
- 想要“更有播客感”：优先试 `CustomVoice`
- 想要“像一个设计出来的声音角色”：再试 `VoiceDesign`

### CLI

```bash
# 主线
python3 podcast_generator.py --backend qwen --file 稿件.txt --ref-audio 我的声音.m4a --ref-text "参考文本"

# 实验后端
python3 podcast_generator.py --backend voxcpm --model openbmb/VoxCPM2 --file 稿件.txt --ref-audio 我的声音.m4a --ref-text "参考文本"
```

### Web UI

Web UI 新增了 `BACKEND` 选择：

- `Qwen Mainline`：默认生产链路
- `VoxCPM Experimental`：隔离实验链路，不参与 Qwen 模型路由

### Benchmark

统一 benchmark 入口：

```bash
python3 benchmark_tts_backends.py \
  --ref-audio audio_samples/龙湖安置小区\ 2.m4a \
  --ref-text "大家好，今天是个好日子。很高兴能和大家分享这些内容，希望对你们有所帮助。让我们开始吧。大家好，今天是个好日子。很高兴能和大家分享这些内容，希望对你们有所帮助。让我们开始吧。"
```

更完整的运行时说明见 [RUNTIME_GUIDE.md](./RUNTIME_GUIDE.md)。

---

## 🖥 交互式选单界面 (CLI)

如果您只是想临时测一段短话，或者不想敲这么长的命令，可以进入交互式菜单：

```bash
python3 main.py
```

在出现的数字菜单中输入 **`3`** (Voice Cloning)，然后将参考音频拖入终端，再输入想要测试的短句即可。临时测试生成的音频同样保存在 `outputs/Clones/` 下。

*(注：长篇播客稿件依然建议使用上方的 `podcast_generator.py` 自动化脚本）。*

---

## 🗑 关于已清理的文件

依据优化您的单一工作流的目的，项目中原有的一些无效、打包报错的残留文件已在此次更新中被安全删除，包括：
* ❌ 之前报错的 Mac App 独立打包脚本（`build_app.sh`, `app_main.py` 等）。
* ❌ 各类在开发早期遗留的单一测试文件（`test_*.py`）。
* ❌ 各种残留的临时错误音频 (`temp_convert_*.wav`)。
项目中目前剩下的都是保证核心跑通的最精简文件，请放心使用。
