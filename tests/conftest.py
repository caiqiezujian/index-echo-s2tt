import numpy as np
import pytest

from s2tt.types import Cue, Hypothesis


@pytest.fixture
def speech_pcm():
    def make(seconds=1.0, rate=16000):
        time = np.arange(round(seconds * rate)) / rate
        return (0.2 * np.sin(2 * np.pi * 220 * time) * 32767).astype("<i2").tobytes()
    return make


@pytest.fixture
def hypothesis():
    def make(task, text="你好。", *, status="complete", stop="eos", model_kind="mock"):
        cues = (Cue(task.snapshot.start_sample, task.snapshot.end_sample, "Synthetic source.", text),) if text else ()
        return Hypothesis(text, cues, status, stop_reason=stop, model_kind=model_kind)
    return make
