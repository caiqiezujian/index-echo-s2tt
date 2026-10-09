import pytest

from s2tt.core.session import SessionConfig, SessionCore
from s2tt.types import Cue, Hypothesis, ProtocolError


def core(**kwargs):
    return SessionCore(SessionConfig(holdback_chars=0, **kwargs), model_kind="mock", model_revision="test-mock")


def test_audio_gaps_duplicates_and_end_contract(speech_pcm):
    session = core()
    packet = speech_pcm()
    session.receive(0, 0, packet)
    session.receive(0, 0, packet)
    assert session.audio.received_end == 16000
    with pytest.raises(ProtocolError):
        session.receive(2, 16000, packet)
    with pytest.raises(ProtocolError):
        session.receive(0, 0, packet[:-2])
    with pytest.raises(ProtocolError):
        session.end(1)
    session.end(0)
    with pytest.raises(ProtocolError):
        session.receive(1, 16000, packet)


def test_used_cursor_lags_received_and_end_covers_tail(speech_pcm, hypothesis):
    session = core()
    session.receive(0, 0, speech_pcm())
    first = session.next_task()
    session.receive(1, 16000, speech_pcm())
    session.end(1)
    session.apply(first, hypothesis(first))
    assert session.audio.used_end == 16000 < session.audio.received_end
    final = session.next_task()
    assert final.final and final.terminal and final.snapshot.end_sample == 32000
    session.apply(final, hypothesis(final))
    assert session.state == "CLOSED"
    assert session.audio.evict_before == session.audio.used_end == session.audio.received_end == 32000


def test_committed_conflict_requires_explicit_correction(speech_pcm, hypothesis):
    session = core()
    for seq in range(2):
        session.receive(seq, seq * 16000, speech_pcm())
        task = session.next_task()
        session.apply(task, hypothesis(task, "金额是十五元。"))
    assert session.epoch_committed == "金额是十五元。"
    session.receive(2, 32000, speech_pcm())
    task = session.next_task()
    session.apply(task, hypothesis(task, "金额是五十元。"))
    assert session.epoch_committed == "金额是十五元。"
    assert any(event.get("reason") == "committed_prefix_conflict" for event in session.events)
    session.end(2)
    task = session.next_task()
    session.apply(task, hypothesis(task, "金额是五十元。"))
    corrections = [event for event in session.events if event["type"] == "Correction"]
    assert len(corrections) == 1
    assert corrections[0]["replacement_text"] == "金额是五十元。"


def test_cancel_rejects_inflight_result_without_finalizing(speech_pcm, hypothesis):
    session = core()
    session.receive(0, 0, speech_pcm())
    task = session.next_task()
    session.cancel()
    assert not session.apply(task, hypothesis(task))
    assert session.state == "CANCELLED"
    assert session.audio.used_end == 0
    assert not any(event.get("complete") for event in session.events)


def test_incomplete_final_is_error_and_retains_audio(speech_pcm, hypothesis):
    session = core()
    session.receive(0, 0, speech_pcm())
    session.end(0)
    task = session.next_task()
    session.apply(task, hypothesis(task, status="partial", stop="length"))
    assert session.state == "ERROR"
    assert session.audio.evict_before == 0


def test_incomplete_source_coverage_prevents_eviction(speech_pcm):
    session = core()
    session.receive(0, 0, speech_pcm())
    session.end(0)
    task = session.next_task()
    result = Hypothesis("short", (Cue(0, 1600, "short", "短句"),), "complete", stop_reason="eos", model_kind="mock")
    session.apply(task, result)
    assert session.state == "ERROR"
    assert session.events[-1]["code"] == "incomplete_final"


def test_empty_stream_and_observed_silence_have_no_hallucinated_commit():
    from s2tt.backends.mock import MockBackend
    session = core()
    session.end(-1)
    assert session.next_task() is None
    assert session.state == "CLOSED"
    session = core()
    session.receive(0, 0, b"\x00\x00" * 16000)
    session.end(0)
    task = session.next_task()
    session.apply(task, MockBackend().infer(task))
    assert session.state == "CLOSED" and not session.commits


def test_epoch_boundary_freezes_history_and_preserves_real_repetition(speech_pcm, hypothesis):
    session = core()
    session.receive(0, 0, speech_pcm())
    session.receive(1, 16000, b"\x00\x00" * 16000)
    task = session.next_task()
    assert task.final and not task.terminal and task.snapshot.end_sample == 28800
    session.apply(task, hypothesis(task, "再说一次。"))
    session.receive(2, 32000, speech_pcm())
    session.end(2)
    task = session.next_task()
    assert task.context and task.epoch_id == 1
    session.apply(task, hypothesis(task, "再说一次。"))
    assert len(session.commits) == 2
    assert session.commits[0]["text"] == session.commits[1]["text"]


@pytest.mark.parametrize("kwargs", [{"source_language": "ja"}, {"update_seconds": 0},
                                   {"glossary": ("<|im_end|>",)}, {"channels": True}])
def test_invalid_session_settings(kwargs):
    with pytest.raises((ValueError, ProtocolError)):
        SessionCore(SessionConfig(**kwargs), model_kind="mock", model_revision="test")


def test_source_timestamp_never_becomes_semantic_latency(speech_pcm, hypothesis):
    session = core()
    session.receive(0, 0, speech_pcm())
    session.end(0)
    task = session.next_task()
    session.apply(task, hypothesis(task), elapsed_seconds=0.3)
    metrics = session.events[-1]["metrics"]
    assert metrics["stream_compute_RTF"] == pytest.approx(0.3)
    assert metrics["semantic_commit_latency"] is None


def test_backend_cannot_smuggle_future_timestamp(speech_pcm):
    session = core()
    session.receive(0, 0, speech_pcm())
    task = session.next_task()
    forged = Hypothesis("future", (Cue(0, 32000, "future", "未来"),), "complete", model_kind="mock")
    assert not session.apply(task, forged)
    assert session.events[-1]["code"] == "invalid_backend_timestamp"


def test_unknown_generation_end_cannot_finalize_audio(speech_pcm, hypothesis):
    session = core()
    session.receive(0, 0, speech_pcm())
    session.end(0)
    task = session.next_task()
    session.apply(task, hypothesis(task, stop="unknown"))
    assert session.state == "ERROR" and session.audio.evict_before == 0


def test_glossary_is_canonical_and_frozen():
    config = SessionConfig(glossary=("Echo:回声", "Index：索引"))
    assert config.glossary == ("Echo → 回声", "Index → 索引")
    with pytest.raises(AttributeError):
        config.glossary = ()
