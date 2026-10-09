import os

import numpy as np
import pytest


def test_adapter_uses_actual_audited_upstream_abstract_interface(monkeypatch):
    source = os.environ.get("SIMULSTREAM_SOURCE_ROOT")
    if source:
        monkeypatch.syspath_prepend(source)
    pytest.importorskip("simulstream", reason="Optional simulstream interface is not installed")
    from types import SimpleNamespace
    from simulstream.server.speech_processors import SpeechProcessor
    from s2tt.backends.mock import MockBackend
    from s2tt.transports.simulstream import IndexEchoSpeechProcessor
    monkeypatch.setattr(IndexEchoSpeechProcessor, "_backend", MockBackend())
    assert issubclass(IndexEchoSpeechProcessor, SpeechProcessor)
    processor = IndexEchoSpeechProcessor(SimpleNamespace(speech_chunk_size=1.0))
    waveform = np.full(16000, 0.1, dtype=np.float32)
    provisional = processor.process_chunk(waveform)
    final = processor.end_of_stream()
    assert "模拟" in provisional.new_string
    assert final.deleted_string == "" and final.new_string == ""
    assert "模拟" in processor.core.commits[0]["text"]
    assert processor.core.state == "CLOSED"
    processor.clear()
    assert processor.core.audio.received_end == 0
