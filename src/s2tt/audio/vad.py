from __future__ import annotations

from collections import deque

import numpy as np

from s2tt.types import SAMPLE_RATE


class EnergyVAD:
    """Observed 20ms RMS boundaries, not semantic sentence boundaries or gold alignment."""

    def __init__(self, threshold=0.008, silence_seconds=0.8):
        self.threshold = threshold
        self.silence_samples = round(silence_seconds * SAMPLE_RATE)
        self.frame_size = 320
        self.processed_end = 0
        self.pending = np.empty(0, dtype=np.float32)
        self.voiced = []
        self.boundaries = deque()
        self._has_voice = False
        self._silence_start = 0
        self._last_silent_boundary = 0

    def push(self, samples, final=False):
        self.pending = np.concatenate((self.pending, samples))
        while len(self.pending) >= self.frame_size or (final and len(self.pending)):
            count = min(len(self.pending), self.frame_size)
            frame, self.pending = self.pending[:count], self.pending[count:]
            start, end = self.processed_end, self.processed_end + count
            speech = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) >= self.threshold
            self.processed_end = end
            if speech:
                if self.voiced and self.voiced[-1][1] == start:
                    self.voiced[-1] = (self.voiced[-1][0], end)
                else:
                    self.voiced.append((start, end))
                self._has_voice = True
                self._silence_start = end
            elif self._has_voice and end - self._silence_start >= self.silence_samples:
                self.boundaries.append(end)
                self._has_voice = False
                self._last_silent_boundary = end
            elif not self._has_voice and end - self._last_silent_boundary >= 8 * SAMPLE_RATE:
                self.boundaries.append(end)
                self._last_silent_boundary = end

    def intervals(self, start, end):
        return tuple((max(a, start), min(b, end)) for a, b in self.voiced if a < end and b > start)

    def evict(self, before):
        self.voiced = [(max(a, before), b) for a, b in self.voiced if b > before]
        while self.boundaries and self.boundaries[0] <= before:
            self.boundaries.popleft()
