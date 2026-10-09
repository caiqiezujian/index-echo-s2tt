import pytest

pytest.importorskip("sacrebleu", reason="Optional separate evaluation dependency is not installed")

from s2tt.evaluation.quality import score_text_files


def test_chrf_is_offline_paired_scoring(tmp_path):
    hypothesis, reference = tmp_path / "hyp.txt", tmp_path / "ref.txt"
    text = "独立合成句子。\n另一个例子。\n"
    hypothesis.write_text(text, encoding="utf-8")
    reference.write_text(text, encoding="utf-8")
    result = score_text_files(hypothesis, reference)
    assert result["score"] == 100 and result["paired_lines"] == 2
    assert result["semantic_latency_available"] is False


def test_unpaired_quality_inputs_fail(tmp_path):
    hypothesis, reference = tmp_path / "hyp.txt", tmp_path / "ref.txt"
    hypothesis.write_text("one\ntwo\n", encoding="utf-8")
    reference.write_text("one\n", encoding="utf-8")
    with pytest.raises(ValueError):
        score_text_files(hypothesis, reference)
