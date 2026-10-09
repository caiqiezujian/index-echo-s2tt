from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


class SessionRunner:
    """One in-flight immutable snapshot; pending input is coalesced in the core ledger."""

    def __init__(self, core, backend, executor=None):
        self.core, self.backend = core, backend
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="echo-model")
        self._owns_executor = executor is None
        self._wake = asyncio.Event()
        self._task = None

    def start(self):
        self._task = asyncio.create_task(self._run())
        return self

    def receive(self, seq, offset, payload):
        result = self.core.receive(seq, offset, payload)
        self._wake.set()
        return result

    def end(self, last_frame_seq):
        self.core.end(last_frame_seq)
        self._wake.set()

    def cancel(self, reason="cancelled"):
        self.core.cancel(reason)
        self._wake.set()

    async def _run(self):
        try:
            while self.core.state not in self.core.terminal_states:
                await self._wake.wait()
                self._wake.clear()
                while self.core.state not in self.core.terminal_states:
                    task = self.core.next_task()
                    if task is None:
                        break
                    started = time.monotonic()
                    try:
                        result = await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.infer, task)
                    except Exception as error:
                        self.core.fail("backend_error", f"{type(error).__name__}: {error}")
                        break
                    self.core.apply(task, result, time.monotonic() - started)
        finally:
            if self._owns_executor:
                self.executor.shutdown(wait=False)

    async def drain_ready(self):
        """Algorithm replay only: wait for the current visible prefix, not future input."""
        self._wake.set()
        while self.core.state not in self.core.terminal_states:
            await asyncio.sleep(0.001)
            if self.core._active_generation is None and not self._wake.is_set():
                break

    async def wait(self, timeout=120):
        await asyncio.wait_for(asyncio.shield(self._task), timeout)
