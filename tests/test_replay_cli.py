import asyncio
import json
import wave

import pytest

from s2tt.backends.mock import MockBackend
from s2tt.cli import main
from s2tt.core.session import SessionConfig
from s2tt.evaluation.replay import replay_wav


@pytest.mark.parametrize("mode", ["causal_fast", "wallclock_1x"])
def test_replay_records_model_identity_and_unique_audio_cost(tmp_path, speech_pcm, mode):
    audio = tmp_path / "synthetic.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(speech_pcm(0.3))
    report = asyncio.run(replay_wav(audio, MockBackend(), SessionConfig(update_seconds=0.1),
                                   tmp_path / "trace.jsonl", mode=mode))
    assert report["status"] == "COMPLETE"
    assert report["model_kinds"] == ["mock"]
    assert report["terminal"]["metrics"]["unique_audio_seconds"] == 0.3
    assert report["semantic_quality_verified"] is False


def test_missing_real_weights_return_blocked_without_mock_fallback(capsys):
    assert main(["probe", "missing.wav", "--model-dir", "missing-model", "--out", "unused.jsonl"]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "BLOCKED" and output["model_kind"] == "real"


def test_packaged_manifest_matches_download_manifest():
    from importlib.resources import files
    from pathlib import Path
    assert json.loads(files("s2tt").joinpath("resources/model_snapshots.json").read_text(encoding="utf-8")) == \
        json.loads(Path("configs/model_snapshots.json").read_text(encoding="utf-8"))
