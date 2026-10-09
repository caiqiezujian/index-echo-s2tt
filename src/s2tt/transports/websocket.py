from __future__ import annotations

import asyncio
import hmac
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from s2tt.core.runner import SessionRunner
from s2tt.core.session import SessionCore
from s2tt.evaluation.trace import TraceWriter
from s2tt.transports.protocol import control_message, unpack_audio
from s2tt.types import ProtocolError


class WebSocketService:
    """Single active session / one model replica. Recovery is explicitly unsupported."""

    def __init__(self, backend, config, *, report_dir="reports/live", auth_token=None, idle_timeout=60):
        self.backend, self.config = backend, config
        self.report_dir, self.auth_token = Path(report_dir), auth_token
        self.idle_timeout = idle_timeout
        self._admission = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="echo-gpu")

    async def handle(self, connection):
        import json

        if self._admission.locked():
            await connection.send(json.dumps({"type": "Error", "code": "capacity_full", "complete": False}))
            await connection.close(code=1013, reason="Single-session capacity is occupied")
            return
        async with self._admission:
            core = runner = sender = trace = None
            queue = asyncio.Queue(maxsize=256)
            overloaded = asyncio.Event()

            async def send_events():
                while True:
                    event = await queue.get()
                    try:
                        await asyncio.wait_for(connection.send(json.dumps(event, ensure_ascii=False)), 10)
                    finally:
                        queue.task_done()

            def emit(event):
                trace.write(event)
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    overloaded.set()

            try:
                first = await asyncio.wait_for(connection.recv(), 10)
                start = control_message(first)
                if start["type"] != "Start" or type(start.get("protocol_version")) is not int or start["protocol_version"] != 1:
                    raise ProtocolError("First message must be Start with protocol_version=1")
                if start.get("resume_token"):
                    raise ProtocolError("Session recovery is unsupported; start a fresh session")
                if self.auth_token and not hmac.compare_digest(str(start.get("token", "")).encode(), self.auth_token.encode()):
                    raise ProtocolError("Authentication failed")
                glossary = start.get("glossary", [])
                if not isinstance(glossary, list) or not all(isinstance(term, str) for term in glossary):
                    raise ProtocolError("Glossary must be a list of strings")
                config = replace(self.config, sample_rate=start.get("sample_rate", 16000),
                                 channels=start.get("channels", 1), glossary=tuple(glossary),
                                 source_language=start.get("source_language", "en"),
                                 target_language=start.get("target_language", "zh"))
                trace = TraceWriter(self.report_dir / f"{uuid.uuid4().hex}.jsonl")
                core = SessionCore(config, model_kind=self.backend.model_kind, model_revision=self.backend.revision,
                                   on_event=emit)
                sender = asyncio.create_task(send_events())
                runner = SessionRunner(core, self.backend, self._executor).start()
                while core.state not in core.terminal_states:
                    # Reception does not await model inference. Event completion also wakes this wait.
                    incoming = asyncio.create_task(connection.recv())
                    done, _ = await asyncio.wait([incoming, runner._task], timeout=self.idle_timeout,
                                                 return_when=asyncio.FIRST_COMPLETED)
                    if incoming not in done:
                        incoming.cancel()
                        await asyncio.gather(incoming, return_exceptions=True)
                        if runner._task in done:
                            break
                        raise ProtocolError("Input idle timeout")
                    message = incoming.result()
                    if isinstance(message, bytes):
                        runner.receive(*unpack_audio(message))
                    else:
                        control = control_message(message)
                        if control["type"] == "End":
                            runner.end(control.get("last_frame_seq"))
                        elif control["type"] == "Cancel":
                            runner.cancel("client_cancelled")
                        elif control["type"] == "Ack":
                            if type(control.get("event_seq")) is not int or not 0 <= control["event_seq"] <= core.event_seq:
                                raise ProtocolError("Invalid event acknowledgement")
                        else:
                            raise ProtocolError("Unsupported control message")
                    if overloaded.is_set() or sender.done():
                        core.fail("slow_consumer", "Event output limit exceeded; reconnect starts a new session")
                        await connection.close(code=1008, reason="Slow event consumer")
                        break
                if runner:
                    await runner.wait()
                if sender and not sender.done():
                    await asyncio.wait_for(queue.join(), 10)
            except ConnectionClosed:
                if runner:
                    runner.cancel("disconnected_no_recovery")
            except (ProtocolError, ValueError, TypeError, asyncio.TimeoutError) as error:
                if core:
                    core.fail("protocol_or_timeout", str(error))
                    if runner:
                        runner._wake.set()
                    if sender and not sender.done():
                        await asyncio.wait_for(queue.join(), 10)
                else:
                    await connection.send(json.dumps({"type": "Error", "code": "invalid_start", "detail": str(error)}))
            finally:
                if runner:
                    if core.state not in core.terminal_states:
                        runner.cancel("connection_closed")
                    # Keep admission occupied until an in-flight GPU call actually returns.
                    await runner._task
                if sender:
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
                if trace:
                    trace.close()

    async def serve(self, host="127.0.0.1", port=8765, origins=None):
        if host not in ("127.0.0.1", "localhost", "::1") and not self.auth_token:
            raise ValueError("Non-loopback binding requires S2TT_AUTH_TOKEN")
        try:
            async with serve(self.handle, host, port, origins=origins, max_size=512 * 1024,
                             max_queue=16, compression=None) as server:
                await server.serve_forever()
        finally:
            self._executor.shutdown(wait=False)
