# Index-Echo 模型审查与英中同传可行性

核查日期：2026-10-08。范围为 Index-Echo-S2TT 2B、9B 的官方说明、发布源码、组件元数据，以及流式接入和提交策略。当前结论来自网络资料与静态代码审查；真实模型加载、英中前缀质量、显存、延迟和 NPU 兼容性均未实测。

**现有主线可作为首版研发路线：simulstream + 独立 SessionCore + Echo 音频前缀重译 + 草稿与稳定提交。系统接入具有明确实现依据；句中稳定英中翻译、目标延迟和长流可持续性必须通过真实权重实验决定。** 保留现有四份方案文件，将本报告作为实施前的核查补充。

## 模型能力和发布接口

官方 S2TT 模型由 Qwen3-Omni AuT 音频编码器、connector 和 Index-Translate 解码器组成。2B、9B 指解码器规模，完整模型还包含音频组件。[2B 模型卡](https://huggingface.co/IndexTeam/Index-Echo-S2TT-2B)、[9B 模型卡](https://huggingface.co/IndexTeam/Index-Echo-S2TT-9B)。

官方 S2TT 评测包含英语→中文，但发布入口限定中文→英语、日语、西班牙语。评测共 140 个 50–60 秒窗口，每方向 20 个；这是完整窗口质量证据，不能证明短前缀、句中提交或在线延迟已经达标。表中的跨方向平均分也不能直接决定英中应选 2B 还是 9B。[技术报告第 4.4 节](https://arxiv.org/html/2609.40181#S4.SS4)。

| 审查项 | 2B 发布包 | 9B 发布包 | 实施要求 |
|---|---|---|---|
| 中文目标指令 | `INSTR` 没有 `zh` | `INSTR` 有 `zh` | 2B 添加同格式中文指令；9B 使用已有指令。两者分别做英中探针。 |
| CLI 目标选项 | `en/ja/es` | `en/ja/es` | 两者均需修正路由；放开选项不等于模型质量通过。 |
| connector 文件头 | `w1/w2` 为 `[2048,2048]`，含 `log_alpha/beta` 参数 | `proj.weight` 为 `[4096,2048]` | 按实际权重键构造并严格加载，不能只替换目录。 |
| 解码器配置 | hidden 2048，24 层，其中 18 层线性、6 层完整注意力 | hidden 4096，32 层，其中 24 层线性、8 层完整注意力 | 两者都有混合状态，缓存与 NPU 回归均须覆盖。 |
| 官方输入输出 | 文件路径；时间戳、源转写、目标译文 | 相同主流程 | 新增内存音频接口，保留原三行模板和原始输出。 |

上述代码与配置取自官方 IndexTeam 的 ModelScope 发布包；源码、配置的 SHA256 与发布元数据一致。connector 只读取文件头，未下载权重张量。对应本地证据见 `reports/index_echo_audit_2026-10-08/source_manifest.json`、`connector_headers.json` 和 `sources/`。[官方 2B 发布包](https://www.modelscope.cn/models/IndexTeam/Index-Echo-S2TT-2B)、[官方 9B 发布包](https://www.modelscope.cn/models/IndexTeam/Index-Echo-S2TT-9B)。

## 官方 stream_translate 的边界

两份 `infer.py` 都先通过 ffmpeg 转换完整输入，再用 `vad_windows()` 对完整波形执行 VAD，最后逐窗推理并 yield 结果。代码自身将其描述为串行伪流式。该流程可以逐窗展示离线字幕，但不能直接用于只可见已到达音频的同传服务。[官方 S2TT 使用说明](https://github.com/bilibili/Index-Translate/blob/f940606e8faf18fca4f4ccfd36185e07415f848f/inference/echo-s2tt/README.md)。

后端应接受不可变的 16 kHz 单声道 float32 音频快照，由 SessionCore 决定可见终点。先验证同一规范化音频的文件路径与数组路径一致，再接入流式输入。`parse_hyp()` 也不能直接承担提交验收：它未检查时间戳是否反向、超出快照或是否适合提交，且可能保留缺失译文的 cue。

当前 `translate_window()` 每轮重新提取特征、运行音频塔、注入 embedding 并生成。它没有公开追加音频状态接口。保留一次生成内部的模型缓存；跨音频更新的缓存继续作为后续研究。

这项限制有具体前处理依据：Echo 使用的 `WhisperFeatureExtractor` 按当前窗口的 log-mel 最大值截断特征。新增更强音频可能改变旧位置的特征数值，因此旧音频 embedding 并非天然不变。[Transformers 5.6.0 特征提取源码](https://github.com/huggingface/transformers/blob/v5.6.0/src/transformers/models/whisper/feature_extraction_whisper.py)。

## 流式框架选择

继续采用 **hlt-mt/simulstream** 作为回放、浏览器演示与评测底座。它支持自定义处理器、WebSocket 和新增/删除输出；需要由独立核心补充稳定提交与更正语义。[simulstream 维护者仓库](https://github.com/hlt-mt/simulstream)。

三个接入细节必须写入实现检查：

1. `SpeechProcessor` 实际定义在 `simulstream.server.speech_processors` 包入口，`BaseSpeechProcessor` 是部分实现。适配器实现模型加载、`process_chunk`、语言设置、`end_of_stream`、清理及文本拼接合同。
2. 上游 `MessageProcessor` 在各处理块上调用独立 `librosa.resample`。首版可先协商输入 16 kHz，避免进入这条重采样分支；随后在项目音频层实现有状态连续重采样并做分包等价验证。
3. WebSocket 已通过执行器运行推理，避免直接阻塞事件循环；但同一连接的取消息协程仍等待本轮处理返回。要落实持续接收、一个运行任务和一个最新待处理快照，应显式分离接收、推理调度与结果发送，并记录真实接收和使用时刻。

对应源码已经锁定在 simulstream commit `b9952dd95a4672ba2806d1f0cd778b7652514697`；见本地 `sources/simulstream/`。

**ufal/SimulStreaming 是另一个项目。** 它的现成路径是 Whisper 直接转英语或 Whisper + EuroLLM 级联，不提供现成 Echo 英中插件。可借鉴其调度与翻译策略，不作为 Echo 已经流式跑通的证明。[SimulStreaming 作者仓库](https://github.com/ufal/SimulStreaming)。

## 稳定提交和计算成本

前缀重译配合 LocalAgreement 有公开实践依据：多个包含新增音频的连续假设确认公共前缀。但一致只反映稳定性，不能证明语义正确；中文重述和英中语序调整还可能阻碍公共前缀增长。[Whisper-Streaming 作者说明](https://github.com/ufal/whisper_streaming#background)。

首版采用三组对照：B0 完整音频参考；B1 在线边界分段；B2 固定当前段起点、增加音频终点的重译加 LA2。B2 比较时冻结上下文、术语与提示版本，去除时间戳对显示文本比较的干扰，保留数字、否定和专名。先测自由假设，不靠强制复写已提交前缀制造一致性。

音频裁剪与文字提交必须分别决定。长流需要保留足够源音频依据，同时限制重算成本。StreamAtt 的研究也将音频历史选择作为独立问题。[StreamAtt 作者论文](https://arxiv.org/abs/2406.06097)。

以每新增音频时长为计算分母。若每 1 秒更新一次、每轮推理需要 2 秒，即使处理的是 10 秒窗口，单 worker 仍无法逐轮跟上；这是计算示例，不是 Echo 实测。快照合并只能减少推理次数，不能丢失输入，也不保证提交延迟达标。

## 环境和硬件

Echo 的 `requirements.txt` 声明验证环境为 Python 3.12、CUDA 12.9、A100，并固定 Torch 2.11.0 与 Transformers 5.6.0。现有方案的 Python 3.11 仍是候选环境，需要验证，不能写成官方已验证组合。

simulstream 基础依赖没有固定 Transformers，但其可选 `hf` extra 固定 `transformers==4.48.1`。Echo 接入使用基础框架加独立后端依赖，避免安装该 extra；评测环境继续分离。2B 约 10 GB 显存是官方使用说明中的估计，9B 不沿用脚本复制的显存/速度注释作为容量证据。

先完成 NVIDIA 单卡单会话。Ascend 官方矩阵提供 Torch 2.11.0、TorchNPU 2.11.0.post2、CANN 9.1.X 的候选组合，但整包 Echo 仍需逐组件验证。vLLM-Ascend 的 Qwen3.5 支持不能代替音频塔、connector、自定义 embedding 与混合状态的整包测试。[TorchNPU 兼容矩阵](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)、[vLLM-Ascend 0.23.0 支持表](https://docs.vllm.ai/projects/ascend/en/v0.23.0/user_guide/support_matrix/supported_models.html)。

## 现有方案的调整

| 方案内容 | 审查判断 | 下一步 |
|---|---|---|
| 独立核心、因果音频台账、草稿和提交分离 | 保留 | 通过不变量测试落实。 |
| 2B 首版、9B 对照 | 合理的工程起点 | 两者先做完整英中和前缀探针，再按质量与成本选择。 |
| 增长前缀重译 + LA2 | 有依据的实验路线 | 先验证尾部幻觉、稳定错误和句中提交比例。 |
| 内存音频接口、严格组件加载 | 必需 | 同时检查 2B 嵌套配置与 9B 文本配置的实际加载结果。 |
| simulstream 接入 | 可实现，需具体适配 | 补充连续重采样和接收/推理/发送解耦。 |
| 两模型的缓存和设备迁移 | 需补充说明 | 明确 9B 也有线性注意力状态。 |
| p50 2–3 秒、p95 约 5 秒 | 继续作为候选目标 | 在真实计算感知回放上决定是否接受，不能提前承诺。 |
| 注意力门控、因果塔、量化、NPU 引擎 | 后续实验 | 保留正确参考路径，每次改变一个因素。 |

AlignAtt4LLM 的新增模型指南需要适配运行时注意力结构并按模型、语言方向标定注意力头，不能直接复制到 Echo。[适配指南](https://github.com/QuentinFuxa/Alignatt4LLM/blob/main/docs/adding_a_model.md)。Qwen3-ASR-causal 也使用专门微调的音频塔，公开结果显示计算和质量之间有取舍；它提供研究参考，不能直接替换 Echo 音频塔。[作者仓库](https://github.com/QuentinFuxa/Qwen3-ASR-causal)。

## 最小验证顺序和退出条件

1. **官方方向健康检查**：2B、9B 分别严格加载，确认组件、特殊 token 和音频注入，保留原始日志。
2. **完整英中与前缀探针**：选取 20–30 条人工可核查英语片段，包含否定、数字、专名、长句、口音和静音；同一片段分别输入完整音频及 1、2、4、8 秒等合法前缀，检查源转写、中文译文、越界时间戳、补全幻觉与推理耗时。该数量和前缀档位是初始实验设计。
3. **B0/B1/B2 因果回放**：先比较策略，再按 1× 真实时钟回放；记录每轮已接收/已使用采样点、排队、草稿、提交与更正。
4. **最小实时字幕**：通过单卡单会话后接麦克风；测试 End、Cancel、长静音、连续说话和至少 30 分钟长流。
5. **按证据决定优化**：英中完整输入失败先查方向和加载；完整输入成功但前缀失败，研究策略或前缀训练；质量成功但成本过高，先 profiling；核心通过后再迁移 NPU。

质量、稳定性与延迟分别报告。simulstream 的重译质量采用最终输出，延迟采用最终 token 的最后更新时间，因此还需要项目自己的提交错误率、更正率、首个有意义提交和语义证据时刻，避免最终译文掩盖在线错误。[simulstream 评测说明](https://arxiv.org/html/2512.17648v2#S2.SS3)。

## 证据版本与限制

Index-Translate 仓库 commit 为 `f940606e8faf18fca4f4ccfd36185e07415f848f`。ModelScope 的 2B 推理源码 revision 为 `1c71a7441fdba791c62f141c82bda614355a320e`，9B 为 `e3cae11e118944a4381dbc568c35e34d8abc0580`；README 和根配置的不同文件 revision 记录在元数据中。这些是 ModelScope revision，未冒充 Hugging Face revision。

Hugging Face 模型卡可读，但本次源码/API 连接失败；原始失败记录与后续取得的 ModelScope 元数据均保留。源码和配置已校验，connector 文件头已检查；完整权重未下载，模型代码未执行。当前可确认的是架构适配路径和需要修改的接口，不能给出服务器上的显存、吞吐、英中质量或同传延迟实测结论。
