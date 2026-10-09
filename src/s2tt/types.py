from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000


class ProtocolError(ValueError):
    """Invalid client input; audio must never be silently repaired."""


class ModelBlocked(RuntimeError):
    """Hardware or verified artifacts are missing."""


@dataclass(frozen=True)
class AudioSnapshot:
    start_sample: int
    end_sample: int
    samples: np.ndarray = field(repr=False, compare=False)
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self):
        if self.end_sample - self.start_sample != len(self.samples):
            raise ValueError("Snapshot length disagrees with its absolute interval")
        if self.samples.ndim != 1 or not np.isfinite(self.samples).all():
            raise ValueError("Snapshot must contain finite mono samples")
        # bytes backing prevents consumers from making the array writable again.
        immutable = np.frombuffer(self.samples.astype("<f4").tobytes(), dtype="<f4")
        object.__setattr__(self, "samples", immutable)

    @property
    def digest(self):
        return hashlib.sha256(self.samples.tobytes()).hexdigest()

    @property
    def duration(self):
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True)
class Cue:
    start_sample: int
    end_sample: int
    source_text: str
    target_text: str


@dataclass(frozen=True)
class Hypothesis:
    raw_text: str
    cues: tuple[Cue, ...]
    parse_status: str
    issues: tuple[str, ...] = ()
    stop_reason: str = "unknown"
    generated_tokens: int | None = None
    model_kind: str = "real"
    metadata: dict = field(default_factory=dict)

    @property
    def target_text(self):
        return "\n".join(cue.target_text for cue in self.cues)

    @property
    def source_text(self):
        return "\n".join(cue.source_text for cue in self.cues)


@dataclass(frozen=True)
class DecodeTask:
    session_id: str
    epoch_id: int
    generation_id: int
    snapshot: AudioSnapshot
    final: bool
    terminal: bool
    context: tuple[str, ...]
    glossary: tuple[str, ...]
    source_language: str
    target_language: str
    history_hash: str
    prompt_hash: str
    model_revision: str
    policy_version: str
    voiced_intervals: tuple[tuple[int, int], ...]
    created_at: float


class Backend(Protocol):
    model_kind: str
    revision: str

    def infer(self, task: DecodeTask) -> Hypothesis: ...
