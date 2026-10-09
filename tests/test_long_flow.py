import numpy as np

from s2tt.backends.mock import MockBackend
from s2tt.core.session import SessionConfig, SessionCore


def test_simulated_thirty_minute_flow_bounds_audio_and_history():
    """Accelerated synthetic/mock logic soak, not a wall-clock GPU benchmark."""
    backend = MockBackend()
    core = SessionCore(SessionConfig(), model_kind="mock", model_revision=backend.revision)
    speech = (np.sin(np.arange(16000) * 2 * np.pi * 220 / 16000) * 6000).astype("<i2").tobytes()
    silence = b"\x00\x00" * 16000
    for seq in range(1800):
        core.receive(seq, seq * 16000, speech if seq % 8 < 6 else silence)
        while task := core.next_task():
            core.apply(task, backend.infer(task))
        assert core.state not in core.terminal_states
        assert core.audio.received_end - core.audio.evict_before <= 8 * 16000
        assert len(core.events) <= 256 and len(core.receipts) <= 256
        assert len(core.context) <= 3
    core.end(1799)
    while task := core.next_task():
        core.apply(task, backend.infer(task))
    assert core.state == "CLOSED"
    assert core.audio.received_end == 1800 * 16000
    assert core.audio.evict_before == core.audio.received_end
