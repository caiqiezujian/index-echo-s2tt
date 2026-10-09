# Index-Echo S2TT

英语语音→中文文本的单卡流式翻译原型。音频只以已接收的不可变快照进入模型；草稿可修订，提交冲突通过明确 Correction 事件处理。

**软件首版面向 AutoDL RTX 5090 32GB。真实模型的英译中质量、显存、速度和长流能力尚待服务器验收。CPU/mock 测试成功不等于模型或同传质量通过。**

## 下载哪个模型

| 模型 | 用途 | 当前项目 |
|---|---|---|
| [Index-Echo-S2TT-2B](https://huggingface.co/IndexTeam/Index-Echo-S2TT-2B) | 音频→翻译文本，包含音频塔、connector、解码器 | 首选完整包 |
| [Index-Echo-S2TT-9B](https://huggingface.co/IndexTeam/Index-Echo-S2TT-9B) | 更大 S2TT 包 | 后续质量对照，单独运行 |
| [Index-Translate-2B](https://huggingface.co/IndexTeam/Index-Translate-2B) | 文本翻译，Transformers 权重 | 仅独立文本/级联实验需要 |
| [Index-Translate-2B-GGUF](https://huggingface.co/IndexTeam/Index-Translate-2B-GGUF) | 文本翻译，llama.cpp 等运行时 | 不能直接替换当前 Echo 音频后端 |

完整 2B 包约 5.90GB，9B 约 19.26GB。模型存放仓库之外。不要额外下载完整基础 Qwen3-Omni 或另一个文本解码器。

## 本地代码检查与模拟演示

Python 3.10+ 可运行纯逻辑；服务器真实模型环境采用 Python 3.12。

```bash
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows PowerShell 改用 .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
ruff check src tests
pytest
echo-s2tt serve --backend mock
```

另一个终端运行：

```bash
echo-s2tt web
```

访问 `http://127.0.0.1:8080`，点击开始收音。模拟模式醒目标识，不进行真实翻译。结束会等待音频尾包与模型任务；取消不会把残缺内容标记为成功。

## AutoDL 5090 安装

先选择 5090 32GB、较新的 Ubuntu 镜像，建立独立 Python 3.12 环境并安装系统 FFmpeg。建议测试 9B 时主机内存 64GB、存储约 100GB；这是准备预算，非实测最低配置。详细步骤见 [环境准备指南](docx/06_AutoDL_5090环境准备.md)。

```bash
export INDEX_ROOT=/root/autodl-tmp/index-echo
mkdir -p "$INDEX_ROOT/code"
cd "$INDEX_ROOT/code"
git clone https://github.com/caiqiezujian/index-echo-s2tt.git
cd index-echo-s2tt
# 此时先激活你创建的 Python 3.12 venv/conda 环境
bash deploy/install_autodl.sh
python scripts/download_echo.py --size 2B --root "$INDEX_ROOT/models" --report-dir "$INDEX_ROOT/reports"
```

候选模型运行环境：torch/torchaudio 2.11.0 cu128、Transformers 5.6.0、BF16。安装器不更换驱动/CUDA Toolkit；实际依赖版本写入 `cuda5090.freeze.txt`。首次模型加载再次核对全部包文件 SHA256，不跳过校验，也不下载隐式基础模型。

## 先运行官方方向，再验证英译中

准备一段清晰的中文音频 `zh_smoke.wav`：

```bash
python "$INDEX_ROOT/models/Index-Echo-S2TT-2B/infer.py" zh_smoke.wav \
  --target-lang en --out "$INDEX_ROOT/reports/official_zh_en" --max-win 20 --ctx-k 0
```

准备 PCM16 WAV 英语音频 `en_smoke.wav`，测试完整输入和真实截断前缀：

```bash
echo-s2tt probe en_smoke.wav \
  --model-dir "$INDEX_ROOT/models/Index-Echo-S2TT-2B" \
  --config configs/5090_2b.json --prefix-seconds 2,4,8,full \
  --out "$INDEX_ROOT/reports/en_zh_probes.jsonl"
```

官方 CLI 没有 `--target-lang zh` 入口；本后端沿用三行模板增加中文目标指令，用内存数组提取特征，字段统一为 source/target。请人工检查英语源转写、中文译文、否定、数字和专名。生成非空中文不构成语义质量验收。

## 因果回放与实时服务

```bash
echo-s2tt replay en_smoke.wav \
  --model-dir "$INDEX_ROOT/models/Index-Echo-S2TT-2B" --config configs/5090_2b.json \
  --mode causal_fast --out "$INDEX_ROOT/reports/causal_fast.jsonl"

echo-s2tt replay en_smoke.wav \
  --model-dir "$INDEX_ROOT/models/Index-Echo-S2TT-2B" --config configs/5090_2b.json \
  --mode wallclock_1x --out "$INDEX_ROOT/reports/wallclock.jsonl"

echo-s2tt summarize "$INDEX_ROOT/reports/wallclock.jsonl"

echo-s2tt serve \
  --model-dir "$INDEX_ROOT/models/Index-Echo-S2TT-2B" --config configs/5090_2b.json \
  --report-dir "$INDEX_ROOT/reports/live"
```

`causal_fast` 等待当前可见前缀的推理，用于算法对照；`wallclock_1x` 按录音时间交付小包，推理与接收独立。完整文件读取仅在离线 probe/回放读取器，后端只接收数组。日志保存原始假设、输入 hash、三个音频光标、版本、Correction 和唯一音频口径 RTF；无语义标注时不编造语义延迟。

服务和网页默认监听服务器 loopback。在本地用控制台 SSH 地址和端口建立两个转发：

```bash
ssh -p <SSH端口> -L 8080:127.0.0.1:8080 -L 8765:127.0.0.1:8765 root@<AutoDL主机>
```

在服务器另一个终端执行 `echo-s2tt web`；本地打开 `http://127.0.0.1:8080`。麦克风权限通过浏览器申请。公网部署自行配置 HTTPS、WSS 与反向代理；非 loopback WebSocket 绑定要求设置 `S2TT_AUTH_TOKEN`，并通过 `--origin` 显式允许网页来源。

## 策略、边界与已知限制

- `--policy boundary`：仅在观察到的静音边界/End 提交，作为保守 B1 对照。
- `--policy la2`：同一 epoch、新增音频、冻结历史下的两轮共同前缀，默认保留 6 个 Unicode 码点；这只是经验稳定证据。
- 当前使用因果 20ms RMS 声学边界。只有最终解析完整、生成未截断、预测 cue 覆盖观察到的发声区间后才裁剪。它不是语义对齐真值；错误和漏译仍需人工/质量评估。
- 一个模型副本、一个活动会话、一个运行快照。请求合并不丢音频；断连明确取消，首版不恢复会话。超过音频硬上限而无可处理边界时明确失败，保留日志，避免静默截断。
- 没有跨更新 cache、量化、注意力门控、gRPC 或 Ascend 真实执行。9B、长流与并发能力须独立验收。
- 可选 `s2tt.transports.simulstream.IndexEchoSpeechProcessor` 对接审查版本的 SpeechProcessor。安装基础 simulstream，避免 `[hf]` 旧依赖。上游同步接口不继承本服务接收/推理解耦保证。

可选适配器的配置见 `configs/simulstream_processor.json`；作为上游 `speech_processor` 配置使用，`type` 指向完整类路径。审查的上游 commit 为 `b9952dd95a4672ba2806d1f0cd778b7652514697`，模型和上游运行依赖需在目标环境核对。

独立文本评分环境安装 `.[eval]` 后，可用 `echo-s2tt score --hypotheses hypotheses.txt --references references.txt` 计算成对文本的 chrF。参考数据不进入模型提示或在线路径。

## 测试与目录

```text
src/s2tt/  audio, core, backends, devices, parsing, policies, transports, evaluation, web
tests/     单元/性质/合同/真实模型入口
configs/   5090 配置、完整模型 revision/校验清单、上游 commit
envs/      CUDA 候选依赖、独立 evaluation、Ascend 待实现说明
deploy/    项目环境安装脚本
docx/      原始设计、审查、环境准备
```

真实模型合同测试：

```bash
export ECHO_MODEL_DIR="$INDEX_ROOT/models/Index-Echo-S2TT-2B"
export ECHO_MODEL_SIZE=2B
export ECHO_SMOKE_WAV=/absolute/path/en_smoke.wav
export ECHO_REAL_REPORT_DIR="$INDEX_ROOT/reports/real_model"
pytest -m real_model -rA
```

硬件/权重缺失会 SKIP 并写出 BLOCKED 原因；设置真实测试路径后加载错误应失败。基础 CI 只验 CPU 合同，不代表 CUDA 已通过。无用户录音、权重、密钥、请求记录或本地 `reports/exports` 数据进入 Git。

参考与限制见 [模型审查](docx/05_IndexEcho模型审查与同传可行性.md)、[协议](docs/protocol.md) 与 [验收状态](docs/implementation_status.md)。
