import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from s2tt.audio.ledger import AudioLedger
from s2tt.audio.normalizer import AudioNormalizer
from s2tt.audio.vad import EnergyVAD
from s2tt.types import ProtocolError


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
def test_resampling_is_independent_of_packet_boundaries(rate, channels):
    rng = np.random.default_rng(11)
    frames = rate + 113
    pcm = rng.integers(-20000, 20000, size=(frames, channels), dtype=np.int16).astype("<i2")
    full = AudioNormalizer(rate, channels)
    expected = np.concatenate([full.push_pcm16(pcm.tobytes()), full.flush()])
    split = AudioNormalizer(rate, channels)
    outputs, start = [], 0
    while start < frames:
        end = min(frames, start + int(rng.integers(1, 1500)))
        outputs.append(split.push_pcm16(pcm[start:end].tobytes()))
        start = end
    outputs.append(split.flush())
    actual = np.concatenate(outputs)
    assert len(actual) == frames * 16000 // rate
    np.testing.assert_array_equal(actual, expected)
    assert len(split._buffer) <= 2


@given(st.lists(st.integers(-32768, 32767), min_size=1, max_size=1000), st.integers(1, 100))
@settings(max_examples=40)
def test_pcm_extremes_and_arbitrary_chunks(samples, packet_frames):
    pcm = np.array(samples, dtype="<i2")
    normalizer = AudioNormalizer(16000)
    outputs = [normalizer.push_pcm16(pcm[i:i + packet_frames].tobytes())
               for i in range(0, len(pcm), packet_frames)]
    np.testing.assert_array_equal(np.concatenate(outputs), pcm.astype(np.float32) / 32768)


@pytest.mark.parametrize("payload", [b"", b"\x00", b"\x00\x01\x02"])
def test_invalid_packets_are_rejected(payload):
    with pytest.raises(ProtocolError):
        AudioNormalizer(16000).push_pcm16(payload)


def test_stereo_mean_and_uncompensated_filter_delay():
    n = AudioNormalizer(16000, 2)
    np.testing.assert_array_equal(n.push_pcm16(np.array([[10000, -10000]], dtype="<i2").tobytes()), [0])
    assert n.filter_delay_seconds == 0
    assert AudioNormalizer(48000).filter_delay_seconds == 31 / 48000


def test_downsampling_attenuates_out_of_band_energy():
    times = np.arange(48000) / 48000
    pcm = (10000 * np.sin(2 * np.pi * 10000 * times)).astype("<i2")
    n = AudioNormalizer(48000)
    result = np.concatenate([n.push_pcm16(pcm.tobytes()), n.flush()])
    assert np.sqrt(np.mean(result[100:] ** 2)) < 0.005


def test_ledger_snapshots_are_immutable_and_causal():
    ledger = AudioLedger(100)
    ledger.append(np.array([0.1, 0.2], dtype=np.float32))
    snapshot = ledger.snapshot(0, 2)
    digest = snapshot.digest
    ledger.append(np.array([100000], dtype=np.float32))
    assert snapshot.digest == digest
    with pytest.raises(ValueError):
        snapshot.samples.setflags(write=True)
    with pytest.raises(ValueError):
        ledger.snapshot(0, 4)
    with pytest.raises(ValueError):
        ledger.evict(2, authorized_through=2)
    ledger.mark_used(2)
    ledger.evict(2, authorized_through=2)
    assert (ledger.evict_before, ledger.used_end, ledger.received_end) == (2, 2, 3)


def test_audio_limit_never_silently_discards_samples():
    ledger = AudioLedger(2)
    ledger.append(np.ones(2, dtype=np.float32))
    with pytest.raises(ProtocolError, match="audio_buffer_limit"):
        ledger.append(np.ones(1, dtype=np.float32))
    assert ledger.received_end == 2


def test_vad_packet_invariance_and_only_observed_boundaries():
    waveform = np.concatenate([np.full(16000, 0.1, dtype=np.float32), np.zeros(16000, dtype=np.float32)])
    full, split = EnergyVAD(), EnergyVAD()
    full.push(waveform)
    for chunk in np.array_split(waveform, 101):
        split.push(chunk)
    assert full.voiced == split.voiced == [(0, 16000)]
    assert list(full.boundaries) == list(split.boundaries) == [28800]


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_future_loud_audio_never_changes_previously_delivered_samples(rate):
    prefix = np.full(rate, 1000, dtype="<i2").tobytes()
    first, second = AudioNormalizer(rate), AudioNormalizer(rate)
    visible_a, visible_b = first.push_pcm16(prefix), second.push_pcm16(prefix)
    first.push_pcm16(np.full(rate, 32767, dtype="<i2").tobytes())
    second.push_pcm16(np.full(rate, -32768, dtype="<i2").tobytes())
    np.testing.assert_array_equal(visible_a, visible_b)
