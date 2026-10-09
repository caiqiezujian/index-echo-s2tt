from __future__ import annotations

from collections import deque

import numpy as np

from s2tt.types import AudioSnapshot, ProtocolError


class AudioLedger:
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.received_end = self.used_end = self.evict_before = 0
        self._blocks = deque()

    def append(self, samples):
        if self.received_end - self.evict_before + len(samples) > self.max_samples:
            raise ProtocolError("audio_buffer_limit: no safe boundary; session must stop explicitly")
        if len(samples):
            immutable = np.frombuffer(np.asarray(samples, dtype="<f4").tobytes(), dtype="<f4")
            self._blocks.append((self.received_end, immutable))
            self.received_end += len(immutable)

    def snapshot(self, start, end):
        if not self.evict_before <= start <= end <= self.received_end:
            raise ValueError("Snapshot exceeds visible audio or references evicted samples")
        pieces = [block[max(0, start - offset):min(len(block), end - offset)]
                  for offset, block in self._blocks if offset < end and offset + len(block) > start]
        samples = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
        return AudioSnapshot(start, end, samples)

    def mark_used(self, end):
        if not self.evict_before <= end <= self.received_end:
            raise ValueError("Invalid model visibility watermark")
        self.used_end = max(self.used_end, end)

    def evict(self, before, *, authorized_through):
        if not self.evict_before <= before <= min(self.used_end, authorized_through):
            raise ValueError("Eviction requires a processed source interval")
        while self._blocks and self._blocks[0][0] + len(self._blocks[0][1]) <= before:
            self._blocks.popleft()
        if self._blocks and self._blocks[0][0] < before:
            offset, block = self._blocks.popleft()
            self._blocks.appendleft((before, block[before - offset:]))
        self.evict_before = before
