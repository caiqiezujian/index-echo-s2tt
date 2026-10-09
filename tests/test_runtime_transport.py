import asyncio
import json
import threading

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from s2tt.backends.mock import MockBackend
from s2tt.core.runner import SessionRunner
from s2tt.core.session import SessionConfig, SessionCore
from s2tt.transports.protocol import pack_audio
from s2tt.transports.websocket import WebSocketService


async def until(connection, kind):
    while True:
        event = json.loads(await asyncio.wait_for(connection.recv(), 5))
        if event["type"] == kind:
            return event


def test_slow_worker_keeps_receiving_and_coalesces_only_requests(speech_pcm):
    class SlowMock(MockBackend):
        def __init__(self):
            self.started, self.release = threading.Event(), threading.Event()
            self.tasks = []

        def infer(self, task):
            self.tasks.append(task)
            self.started.set()
            assert self.release.wait(5)
            return super().infer(task)

    async def run():
        backend = SlowMock()
        core = SessionCore(SessionConfig(update_seconds=0.1), model_kind="mock", model_revision=backend.revision)
        runner = SessionRunner(core, backend).start()
        runner.receive(0, 0, speech_pcm(0.2))
        assert await asyncio.to_thread(backend.started.wait, 5)
        for seq in (1, 2, 3):
            runner.receive(seq, seq * 3200, speech_pcm(0.2))
        assert core.audio.received_end == 12800 and core.audio.used_end == 0
        runner.end(3)
        backend.release.set()
        await runner.wait(5)
        assert core.state == "CLOSED"
        assert len(backend.tasks) == 2
        assert backend.tasks[-1].snapshot.end_sample == 12800
    asyncio.run(run())


@pytest.mark.integration
def test_websocket_start_audio_end_and_single_session_admission(tmp_path, speech_pcm):
    async def run():
        service = WebSocketService(MockBackend(), SessionConfig(update_seconds=0.1), report_dir=tmp_path)
        try:
            async with serve(service.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                uri = f"ws://127.0.0.1:{port}"
                async with connect(uri) as first:
                    await first.send(json.dumps({"type": "Start", "protocol_version": 1, "sample_rate": 16000}))
                    assert (await until(first, "Ready"))["model_kind"] == "mock"
                    async with connect(uri) as second:
                        busy = json.loads(await second.recv())
                        assert busy["code"] == "capacity_full"
                    await first.send(pack_audio(0, 0, speech_pcm(0.2)))
                    await first.send(json.dumps({"type": "End", "last_frame_seq": 0}))
                    final = await until(first, "StreamEnd")
                    assert final["complete"] and final["model_kind"] == "mock"
                    assert final["used_audio_end_sample"] == final["received_end_sample"] == 3200
        finally:
            service._executor.shutdown(wait=True)
        assert len(list(tmp_path.glob("*.jsonl"))) == 1
    asyncio.run(run())


@pytest.mark.integration
def test_websocket_uses_local_language_and_glossary_defaults(tmp_path, speech_pcm):
    class TrackingMock(MockBackend):
        def infer(self, task):
            assert (task.source_language, task.target_language) == ("zh", "ja")
            assert task.glossary == ("Echo → エコー",)
            return super().infer(task)

    async def run():
        service = WebSocketService(TrackingMock(), SessionConfig(source_language="zh", target_language="ja",
                                   glossary=("Echo:エコー",), update_seconds=0.1), report_dir=tmp_path)
        try:
            async with serve(service.handle, "127.0.0.1", 0) as server:
                uri = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                async with connect(uri) as connection:
                    await connection.send(json.dumps({"type": "Start", "protocol_version": 1}))
                    ready = await until(connection, "Ready")
                    assert (ready["source_language"], ready["target_language"]) == ("zh", "ja")
                    await connection.send(pack_audio(0, 0, speech_pcm(0.2)))
                    await connection.send(json.dumps({"type": "End", "last_frame_seq": 0}))
                    assert (await until(connection, "StreamEnd"))["complete"]
        finally:
            service._executor.shutdown(wait=True)
    asyncio.run(run())


@pytest.mark.integration
def test_websocket_gap_becomes_visible_error(tmp_path, speech_pcm):
    async def run():
        service = WebSocketService(MockBackend(), SessionConfig(), report_dir=tmp_path)
        try:
            async with serve(service.handle, "127.0.0.1", 0) as server:
                uri = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
                async with connect(uri) as connection:
                    await connection.send(json.dumps({"type": "Start", "protocol_version": 1}))
                    await until(connection, "Ready")
                    await connection.send(pack_audio(1, 0, speech_pcm(0.2)))
                    error = await until(connection, "Error")
                    assert not error["complete"]
                    assert "out-of-order" in error["detail"]
        finally:
            service._executor.shutdown(wait=True)
    asyncio.run(run())
