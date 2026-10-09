import numpy as np
import pytest

from s2tt.parsing.echo import parse_echo
from s2tt.policies.agreement import LocalAgreement
from s2tt.types import AudioSnapshot


def snapshot():
    return AudioSnapshot(16000, 16000 * 5, np.zeros(16000 * 4, dtype=np.float32))


def test_parse_direction_neutral_absolute_times_and_repeated_speech():
    result = parse_echo("[00:00.00-00:01.00]\nAgain.\n再说一次。\n[00:01.00-00:02.00]\nAgain.\n再说一次。", snapshot())
    assert result.parse_status == "complete"
    assert len(result.cues) == 2
    assert result.cues[0].start_sample == 16000
    assert result.target_text == "再说一次。\n再说一次。"


@pytest.mark.parametrize("raw,issue", [
    ("[00:00.00-00:01.00]\nEnglish only", "incomplete_cue"),
    ("[00:02.00-00:01.00]\nSource\n译文", "invalid_timestamp"),
    ("[00:00.00-00:10.00]\nSource\n译文", "outside_snapshot"),
    ("[00:60.00-01:02.00]\nSource\n译文", "invalid_timestamp"),
    ("not a timestamp\nSource\n译文", "unexpected_line"),
])
def test_bad_structures_are_not_complete(raw, issue):
    result = parse_echo(raw, snapshot())
    assert result.parse_status != "complete"
    assert any(entry.startswith(issue) for entry in result.issues)
    assert result.raw_text == raw


def test_generation_length_does_not_become_complete_final():
    result = parse_echo("[00:00.00-00:01.00]\nSource\n译文", snapshot(), stop_reason="length")
    assert result.parse_status == "partial"


def test_partial_tail_keeps_earlier_cue_but_discloses_failure():
    raw = "[00:00.00-00:01.00]\nSource\n译文\n[00:01.00-00:02.00]\nunfinished"
    result = parse_echo(raw, snapshot())
    assert result.parse_status == "partial"
    assert len(result.cues) == 1


def test_repeated_identical_audio_is_not_new_agreement_evidence():
    policy = LocalAgreement(holdback_chars=0)
    assert policy.observe("中文🙂", 16000, "epoch") == ""
    assert policy.observe("中文🙂", 16000, "epoch") == ""
    assert policy.observe("中文🙂", 32000, "epoch") == "中文🙂"
    assert policy.observe("中文🙂", 48000, "new-prompt") == ""


def test_holdback_and_english_word_boundary():
    policy = LocalAgreement(holdback_chars=2)
    policy.observe("一二三四五", 1, "same")
    assert policy.observe("一二三四五六", 2, "same") == "一二三"
    english = LocalAgreement(holdback_chars=2, target_language="en")
    english.observe("one seventy five", 1, "same")
    assert english.observe("one seventy five", 2, "same") == "one seventy"
