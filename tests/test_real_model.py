import os
import re
import wave
import json
from pathlib import Path

import numpy as np
import pytest

from s2tt.audio.normalizer import AudioNormalizer
from s2tt.audio.vad import EnergyVAD
from s2tt.core.session import SessionConfig, SessionCore

pytestmark = pytest.mark.real_model


def record_result(backend, name, payload):
    directory = Path(os.environ.get("ECHO_REAL_REPORT_DIR", "reports/real_model"))
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"model_kind": "real", "model_revision": backend.revision, "model_size": backend.size,
               "semantic_quality_verified": False, **payload}
    (directory / f"{backend.size.lower()}_{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def real_inputs():
    directory, audio = os.environ.get("ECHO_MODEL_DIR"), os.environ.get("ECHO_SMOKE_WAV")
    if not directory or not audio:
        pytest.skip("BLOCKED: set ECHO_MODEL_DIR and ECHO_SMOKE_WAV on a CUDA server; no real model tested")
    from s2tt.backends.index_echo import IndexEchoBackend
    backend = IndexEchoBackend(directory, os.environ.get("ECHO_MODEL_SIZE", "2B"))
    with wave.open(str(Path(audio)), "rb") as recording:
        assert recording.getsampwidth() == 2
        normalizer = AudioNormalizer(recording.getframerate(), recording.getnchannels())
        pieces = []
        while payload := recording.readframes(recording.getframerate()):
            pieces.append(normalizer.push_pcm16(payload))
        pieces.append(normalizer.flush())
    return backend, np.concatenate(pieces)


def test_real_english_to_chinese_contract(real_inputs):
    backend, waveform = real_inputs
    core = SessionCore(SessionConfig(), model_kind="real", model_revision=backend.revision)
    assert 16000 <= len(waveform) <= 30 * 16000, "Use a 1–30s clearly spoken English smoke recording"
    for seq, start in enumerate(range(0, len(waveform), 16000)):
        pcm = np.clip(waveform[start:start + 16000] * 32768, -32768, 32767).astype("<i2").tobytes()
        core.receive(seq, start, pcm)
    core.end(core.next_frame_seq - 1)
    task = core.next_task()
    result = backend.infer(task)
    record_result(backend, "full", {"raw_text": result.raw_text, "source_text": result.source_text,
                                   "target_text": result.target_text, "parse_status": result.parse_status,
                                   "issues": result.issues, "stop_reason": result.stop_reason,
                                   "input_sha256": task.snapshot.digest, "metadata": result.metadata})
    assert result.model_kind == "real" and result.parse_status == "complete"
    assert re.search(r"[\u4e00-\u9fff]", result.target_text)
    assert re.search(r"[A-Za-z]", result.source_text)
    # Direction/format check only; semantic accuracy requires human evaluation.


def test_real_partial_input_is_bounded_and_keeps_raw_output(real_inputs):
    from dataclasses import replace
    from s2tt.types import AudioSnapshot
    backend, waveform = real_inputs
    core = SessionCore(SessionConfig(), model_kind="real", model_revision=backend.revision)
    clip = waveform[:min(len(waveform), 2 * 16000)]
    core.receive(0, 0, np.clip(clip * 32768, -32768, 32767).astype("<i2").tobytes())
    task = core.next_task()
    assert task is not None
    snapshot = AudioSnapshot(0, len(clip), clip)
    vad = EnergyVAD()
    vad.push(clip, final=True)
    result = backend.infer(replace(task, snapshot=snapshot, voiced_intervals=vad.intervals(0, len(clip))))
    record_result(backend, "prefix", {"raw_text": result.raw_text, "parse_status": result.parse_status,
                                     "issues": result.issues, "snapshot_end": snapshot.end_sample,
                                     "input_sha256": snapshot.digest, "metadata": result.metadata})
    assert result.model_kind == "real" and isinstance(result.raw_text, str)
    assert all(cue.end_sample <= snapshot.end_sample + 320 for cue in result.cues)


def test_array_encoder_matches_official_wav_encoder(real_inputs, tmp_path):
    """Use the same loaded components; do not allocate a second GPU model copy."""
    import importlib.util
    backend, waveform = real_inputs
    import torch
    specification = importlib.util.spec_from_file_location("verified_echo_reference", backend.root / "infer.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    reference = module.AudioTransModel.__new__(module.AudioTransModel)
    for name, value in {"device": backend.device, "dtype": backend.dtype, "tower": backend.tower,
                        "connector": backend.connector, "fe": backend.extractor}.items():
        setattr(reference, name, value)
    pcm = np.clip(waveform[:2 * 16000] * 32768, -32768, 32767).astype("<i2")
    path = tmp_path / "prefix.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(pcm.tobytes())
    from_array, _ = backend.encode_audio_array(pcm.astype(np.float32) / 32768)
    from_file = reference.encode_audio(str(path))
    record_result(backend, "encoder_comparison", {"shape": list(from_array.shape),
                  "max_absolute_error": float((from_array - from_file).abs().max())})
    torch.testing.assert_close(from_array, from_file, rtol=0.01, atol=0.01)
