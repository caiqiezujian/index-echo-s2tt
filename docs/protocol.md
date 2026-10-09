# WebSocket protocol v1

连接默认 `ws://127.0.0.1:8765`。单卡单会话，服务端固定模型目录；客户端不能传模型路径。

## Start

第一条消息是 JSON：

```json
{"type":"Start","protocol_version":1,"source_language":"en","target_language":"zh","sample_rate":48000,"channels":1,"glossary":["Echo:回声"]}
```

如设置服务鉴权，额外带 `token`。允许 PCM16LE，16k/44.1k/48k，1 或 2 通道；双通道取均值。非16k 输入经过有状态 FIR 和插值，31 个输入样本滤波延迟单独记录。先等 Ready，再发音频。

## Audio / End / Cancel / Ack

音频二进制：12字节 little-endian 头，`uint32 frame_seq`、`uint64 first_sample`，后接 interleaved PCM16LE。`first_sample` 单位是输入采样率下的 sample frame，立体声每帧含两个样本。序号从0开始，offset从0开始；20–40ms为推荐包长，单包上限2秒。

重复帧需内容和offset一致，最近256帧可幂等识别；过旧重复、缺帧、乱序、半样本、空包拒绝，不补静音。

```json
{"type":"End","last_frame_seq":123}
{"type":"Cancel"}
{"type":"Ack","event_seq":456}
```

End 检查最后接收序号，排空重采样尾部及所有待处理 epoch。空会话使用 `last_frame_seq:-1`。Cancel/断连不产生成功 final。Ack 只校验事件序号，首版不提供恢复承诺；重连必须创建新会话。

## Server events

所有会话事件包含 `session_id/event_seq/output_version/epoch_id`、三个16k绝对样本光标 `received_end_sample/used_audio_end_sample/evict_before_sample`、单调时钟 `server_emit_time`、`model_kind/model_revision`。

Ready、AudioAck、EndAccepted、Hypothesis、DraftSnapshot、CommitAppend、Correction、Boundary、Warning、StreamEnd、Error。

输出事件原子携带 `committed_text/draft_text` 完整快照。客户端去重 event_seq，按 output_version 应用；不要以UTF-8字节长度裁剪。CommitAppend含commit_id及追加text；Correction引用commit_id并带replacement_text。时间戳为模型预测，不能用于计算语义延迟真值。

StreamEnd `complete:true` 表示该输入已排空并满足机械解析/声学覆盖合同，不表示译文语义已人工验收。Error/CANCELLED为未完成。连接期间的状态记录在JSONL日志。

接收与同步模型调用独立；一个线程执行器串行生成。慢推理期间保留所有收到音频，只合并快照请求。缓冲/输出队列达到限制后可见失败。断连后仍运行的设备调用须返回，才释放单会话准入。
