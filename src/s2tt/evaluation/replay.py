from __future__ import annotations

import asyncio
import time
import wave

from s2tt.core.runner import SessionRunner
from s2tt.core.session import SessionConfig, SessionCore
from s2tt.evaluation.trace import TraceWriter, summarize_trace
from s2tt.types import ProtocolError


async def replay_wav(path, backend, config: SessionConfig, trace_path, *, mode="causal_fast", packet_ms=40):
    """Read recording packets in the orchestrator; the backend receives only snapshots."""
    if not 10 <= packet_ms <= 200:
        raise ValueError("Replay packet size must be 10–200ms")
    with wave.open(str(path), "rb") as recording:
        if recording.getsampwidth() != 2 or recording.getcomptype() != "NONE":
            raise ProtocolError("Replay input must be uncompressed PCM16 WAV")
        from dataclasses import replace
        config = replace(config, sample_rate=recording.getframerate(), channels=recording.getnchannels())
        trace = TraceWriter(trace_path)
        core = SessionCore(config, model_kind=backend.model_kind, model_revision=backend.revision, on_event=trace.write)
        runner = SessionRunner(core, backend).start()
        try:
            frames_per_packet = round(config.sample_rate * packet_ms / 1000)
            seq, source_offset, origin = 0, 0, time.monotonic()
            while core.state not in core.terminal_states:
                pcm = recording.readframes(frames_per_packet)
                if not pcm:
                    break
                count = len(pcm) // (2 * config.channels)
                if mode == "wallclock_1x":
                    deadline = origin + (source_offset + count) / config.sample_rate
                    await asyncio.sleep(max(0, deadline - time.monotonic()))
                runner.receive(seq, source_offset, pcm)
                seq, source_offset = seq + 1, source_offset + count
                if mode == "causal_fast":
                    await runner.drain_ready()
                else:
                    await asyncio.sleep(0)
            if core.state not in core.terminal_states:
                runner.end(seq - 1)
            await runner.wait()
        except BaseException:
            runner.cancel("replay_interrupted")
            await runner.wait()
            raise
        finally:
            trace.close()
    result = summarize_trace(trace_path)
    result["input_mode"] = mode
    return result
