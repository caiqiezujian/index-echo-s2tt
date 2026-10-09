"""Optional adapter for the audited hlt-mt/simulstream SpeechProcessor contract.

Its upstream synchronous callback cannot provide this project's continuous
receiver guarantees; use our WebSocket service for the primary live path.
"""
from __future__ import annotations

import numpy as np
import threading
from simulstream.server.speech_processors import SpeechProcessor
from simulstream.server.speech_processors.incremental_output import IncrementalOutput

from s2tt.backends.index_echo import IndexEchoBackend
from s2tt.core.session import SessionConfig, SessionCore


class IndexEchoSpeechProcessor(SpeechProcessor):
    _backend = None
    _model_lock = threading.Lock()

    @classmethod
    def load_model(cls, config):
        cls._backend = IndexEchoBackend(config.model_dir, getattr(config, "model_size", "2B"),
                                       getattr(config, "device", "cuda:0"))

    def __init__(self, config):
        super().__init__(config)
        if self._backend is None:
            raise RuntimeError("Call load_model before constructing the processor")
        self.source_language = getattr(config, "source_language", "en")
        self.target_language = getattr(config, "target_language", "zh")
        self.clear()

    def clear(self):
        self.core = SessionCore(SessionConfig(source_language=self.source_language, target_language=self.target_language),
                                model_kind=self._backend.model_kind, model_revision=self._backend.revision)
        self._visible = ""

    def _decode(self):
        while task := self.core.next_task():
            with self._model_lock:
                result = self._backend.infer(task)
            self.core.apply(task, result)
        if self.core.state == "ERROR":
            raise RuntimeError(self.core.events[-1])
        visible = "\n".join(item["text"] for item in self.core.commits)
        if self.core.draft:
            visible += self.core.draft
        common = 0
        for left, right in zip(self._visible, visible):
            if left != right:
                break
            common += 1
        deleted, added = self._visible[common:], visible[common:]
        self._visible = visible
        return IncrementalOutput(list(added), added, list(deleted), deleted)

    def process_chunk(self, waveform):
        if waveform.ndim != 1 or not np.isfinite(waveform).all() or np.max(np.abs(waveform), initial=0) > 1:
            raise ValueError("Expected finite mono waveform in [-1, 1]")
        pcm = np.clip(np.rint(waveform * 32768), -32768, 32767).astype("<i2").tobytes()
        if pcm:
            self.core.receive(self.core.next_frame_seq, self.core.normalizer.input_end, pcm)
        return self._decode()

    def set_source_language(self, language):
        if self.core.next_frame_seq:
            raise ValueError("Language cannot change during an active stream")
        self.source_language = language
        self.clear()

    def set_target_language(self, language):
        if self.core.next_frame_seq:
            raise ValueError("Language cannot change during an active stream")
        self.target_language = language
        self.clear()

    def end_of_stream(self):
        self.core.end(self.core.next_frame_seq - 1)
        return self._decode()

    def tokens_to_string(self, tokens):
        return "".join(tokens)
