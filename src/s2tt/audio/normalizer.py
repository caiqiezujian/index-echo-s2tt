from __future__ import annotations

import numpy as np

from s2tt.types import SAMPLE_RATE, ProtocolError


class AudioNormalizer:
    """Stateful causal FIR + interpolation resampler; never uses unseen samples.

    Non-16k inputs have an uncompensated 31-input-sample FIR group delay.
    Mean mixdown is explicit; opposite-phase stereo may cancel.
    """

    def __init__(self, input_rate: int, channels: int = 1):
        if type(input_rate) is not int or input_rate not in (16000, 44100, 48000):
            raise ProtocolError("Supported input rates: 16000, 44100, 48000")
        if type(channels) is not int or channels not in (1, 2):
            raise ProtocolError("Supported channel counts: 1, 2 (mean mixdown)")
        self.input_rate, self.channels = input_rate, channels
        self.input_end = self.output_end = self._base = 0
        self._buffer = np.empty(0, dtype=np.float32)
        self._closed = False
        cutoff = 0.45 * SAMPLE_RATE / input_rate
        n = np.arange(63) - 31
        taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(63)
        self._taps = taps / taps.sum()
        self._history = np.zeros(62, dtype=np.float32)

    @property
    def filter_delay_seconds(self):
        return 0.0 if self.input_rate == SAMPLE_RATE else 31 / self.input_rate

    def push_pcm16(self, payload: bytes):
        if self._closed:
            raise ProtocolError("Audio received after normalizer flush")
        if not payload or len(payload) % (2 * self.channels):
            raise ProtocolError("PCM packet must contain complete, nonempty sample frames")
        frames = len(payload) // (2 * self.channels)
        if frames > self.input_rate * 2:
            raise ProtocolError("An audio packet may contain at most two seconds")
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32).reshape(-1, self.channels)
        mono = samples.mean(axis=1) / 32768.0
        self.input_end += len(mono)
        if self.input_rate == SAMPLE_RATE:
            self.output_end += len(mono)
            return mono
        extended = np.concatenate((self._history, mono))
        filtered = np.convolve(extended, self._taps, mode="valid").astype(np.float32)
        self._history = extended[-62:].copy()
        self._buffer = np.concatenate((self._buffer, filtered))
        # j * input_rate / output_rate must be strictly before the last input sample.
        interpolation_stop = ((self.input_end - 1) * SAMPLE_RATE + self.input_rate - 1) // self.input_rate
        stop = min(interpolation_stop, self.input_end * SAMPLE_RATE // self.input_rate)
        return self._emit(stop, final=False)

    def _emit(self, stop, final):
        if stop <= self.output_end:
            return np.empty(0, dtype=np.float32)
        numerators = np.arange(self.output_end, stop, dtype=np.int64) * self.input_rate
        left = numerators // SAMPLE_RATE - self._base
        fraction = (numerators % SAMPLE_RATE).astype(np.float64) / SAMPLE_RATE
        right = np.minimum(left + 1, len(self._buffer) - 1) if final else left + 1
        result = ((1 - fraction) * self._buffer[left] + fraction * self._buffer[right]).astype(np.float32)
        self.output_end = stop
        keep_from = min(self.output_end * self.input_rate // SAMPLE_RATE, self.input_end - 1)
        remove = max(0, keep_from - self._base)
        self._buffer = self._buffer[remove:].copy()
        self._base += remove
        return result

    def flush(self):
        if self._closed:
            return np.empty(0, dtype=np.float32)
        self._closed = True
        if self.input_rate == SAMPLE_RATE or self.input_end == 0:
            return np.empty(0, dtype=np.float32)
        # End authorizes only endpoint hold, never a future-sample lookahead.
        result = self._emit(self.input_end * SAMPLE_RATE // self.input_rate, final=True)
        self._buffer = np.empty(0, dtype=np.float32)
        self._history = np.empty(0, dtype=np.float32)
        return result
