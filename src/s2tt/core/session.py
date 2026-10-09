from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass

from s2tt.audio.ledger import AudioLedger
from s2tt.audio.normalizer import AudioNormalizer
from s2tt.audio.vad import EnergyVAD
from s2tt.parsing.echo import covers_observed_speech
from s2tt.policies.agreement import BoundaryOnly, LocalAgreement
from s2tt.types import SAMPLE_RATE, DecodeTask, ProtocolError


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class SessionConfig:
    source_language: str = "en"
    target_language: str = "zh"
    sample_rate: int = SAMPLE_RATE
    channels: int = 1
    update_seconds: float = 1.0
    max_buffer_seconds: float = 60.0
    silence_seconds: float = 0.8
    vad_threshold: float = 0.008
    policy: str = "la2"
    holdback_chars: int = 6
    context_windows: int = 3
    glossary: tuple[str, ...] = ()

    def __post_init__(self):
        if (self.source_language, self.target_language) not in (("en", "zh"), ("zh", "en"), ("zh", "ja"), ("zh", "es")):
            raise ProtocolError("Supported routes: en→zh (experimental), zh→en/ja/es")
        if not 0.1 <= self.update_seconds <= 10 or not 2 <= self.max_buffer_seconds <= 300:
            raise ProtocolError("Invalid update interval or bounded audio window")
        if not 0.2 <= self.silence_seconds <= 5 or not 0 < self.vad_threshold < 1:
            raise ProtocolError("Invalid acoustic boundary settings")
        if self.policy not in ("la2", "boundary") or not 0 <= self.holdback_chars <= 100:
            raise ProtocolError("Invalid commit policy")
        if any(not isinstance(term, str) or "<|" in term or "\n" in term or "[Context]" in term for term in self.glossary):
            raise ProtocolError("Glossary terms must not contain template control tokens")
        if not 0 <= self.context_windows <= 5 or sum(map(len, self.glossary)) > 2048:
            raise ProtocolError("Context/glossary limit exceeded")
        normalized = []
        for term in self.glossary:
            source, separator, target = term.replace("：", ":").partition(":")
            normalized.append(f"{source.strip()} → {target.strip()}" if separator and target.strip() else term.strip())
        object.__setattr__(self, "glossary", tuple(normalized))


class SessionCore:
    terminal_states = {"CLOSED", "CANCELLED", "ERROR"}

    def __init__(self, config: SessionConfig, *, model_kind, model_revision, on_event=None, clock=time.monotonic):
        self.config, self.model_kind, self.model_revision = config, model_kind, model_revision
        self.clock, self.on_event = clock, on_event
        self.session_id = uuid.uuid4().hex
        self.state = "READY"
        self.normalizer = AudioNormalizer(config.sample_rate, config.channels)
        self.audio = AudioLedger(round(config.max_buffer_seconds * SAMPLE_RATE))
        self.vad = EnergyVAD(config.vad_threshold, config.silence_seconds)
        self.policy = (LocalAgreement(holdback_chars=config.holdback_chars, target_language=config.target_language)
                       if config.policy == "la2" else BoundaryOnly())
        self.started_at = clock()
        self.epoch_id = self.epoch_start = self.generation = self.event_seq = self.output_version = 0
        self.next_frame_seq = 0
        self.receipts = OrderedDict()
        self.events = deque(maxlen=256)
        self.commits = []
        self.epoch_committed = self.draft = ""
        self.context = ()
        self._last_task_end = 0
        self._active_generation = None
        self.inference_seconds = 0.0
        self.first_draft_seconds = self.first_commit_seconds = None
        self.emit("Ready", reason="session_started", source_language=config.source_language,
                  target_language=config.target_language, direction_status="experimental",
                  resampler_delay_seconds=self.normalizer.filter_delay_seconds, reconnect_supported=False)

    def emit(self, kind, **payload):
        self.event_seq += 1
        event = {
            "type": kind, "session_id": self.session_id, "event_seq": self.event_seq,
            "output_version": self.output_version, "epoch_id": self.epoch_id,
            "used_audio_end_sample": self.audio.used_end, "received_end_sample": self.audio.received_end,
            "evict_before_sample": self.audio.evict_before, "server_emit_time": self.clock(),
            "model_kind": self.model_kind, "model_revision": self.model_revision, **payload,
        }
        self.events.append(event)
        if self.on_event:
            self.on_event(event)
        return event

    def receive(self, frame_seq, first_sample, payload):
        if self.state not in ("READY", "RECEIVING", "DECODING"):
            raise ProtocolError(f"Audio is not accepted in state {self.state}")
        if type(frame_seq) is not int or type(first_sample) is not int or min(frame_seq, first_sample) < 0:
            raise ProtocolError("Frame sequence and source sample offset must be nonnegative integers")
        signature = (first_sample, hashlib.sha256(payload).hexdigest())
        if frame_seq < self.next_frame_seq:
            if self.receipts.get(frame_seq) != signature:
                raise ProtocolError("Conflicting or expired duplicate audio frame")
            return self.emit("AudioAck", frame_seq=frame_seq, duplicate=True,
                             input_end_sample=self.normalizer.input_end)
        if frame_seq != self.next_frame_seq or first_sample != self.normalizer.input_end:
            raise ProtocolError("Audio gap or out-of-order frame; no implicit silence insertion")
        samples = self.normalizer.push_pcm16(payload)
        self.audio.append(samples)
        self.vad.push(samples)
        self.receipts[frame_seq] = signature
        if len(self.receipts) > 256:
            self.receipts.popitem(last=False)
        self.next_frame_seq += 1
        if self.state != "DECODING":
            self.state = "RECEIVING"
        return self.emit("AudioAck", frame_seq=frame_seq, duplicate=False,
                         input_end_sample=self.normalizer.input_end)

    def end(self, last_frame_seq):
        if self.state == "DRAINING" and last_frame_seq == self.next_frame_seq - 1:
            return
        if self.state in self.terminal_states:
            raise ProtocolError("Session is already terminal")
        if type(last_frame_seq) is not int or last_frame_seq != self.next_frame_seq - 1:
            raise ProtocolError("End does not acknowledge the last accepted frame")
        tail = self.normalizer.flush()
        self.audio.append(tail)
        self.vad.push(tail, final=True)
        self.state = "DRAINING"
        self.emit("EndAccepted", last_frame_seq=last_frame_seq, final_end_sample=self.audio.received_end)

    def cancel(self, reason="client_cancelled"):
        if self.state not in self.terminal_states:
            self.state = "CANCELLED"
            self.generation += 1
            self.emit("StreamEnd", status="CANCELLED", reason=reason, complete=False)

    def fail(self, code, detail):
        if self.state not in self.terminal_states:
            self.state = "ERROR"
            self.generation += 1
            self.emit("Error", code=code, detail=detail, complete=False)

    def next_task(self):
        if self.state in self.terminal_states or self._active_generation is not None:
            return None
        end = self.vad.boundaries[0] if self.vad.boundaries else self.audio.received_end
        final = bool(self.vad.boundaries) or self.state == "DRAINING"
        if not final and end - self._last_task_end < round(self.config.update_seconds * SAMPLE_RATE):
            return None
        if end == self.epoch_start:
            if self.state == "DRAINING":
                self._close()
            return None
        self.generation += 1
        self._active_generation = self.generation
        self._last_task_end = end
        snapshot = self.audio.snapshot(self.epoch_start, end)
        if self.state != "DRAINING":
            self.state = "DECODING"
        history_hash = fingerprint(self.context)
        prompt_hash = fingerprint({"glossary": self.config.glossary, "target": self.config.target_language,
                                   "template": "official-three-line-v1"})
        return DecodeTask(self.session_id, self.epoch_id, self.generation, snapshot, final,
                          self.state == "DRAINING" and end == self.audio.received_end,
                          self.context, self.config.glossary, self.config.source_language,
                          self.config.target_language, history_hash, prompt_hash, self.model_revision,
                          self.policy.version, self.vad.intervals(self.epoch_start, end), self.clock())

    def apply(self, task, hypothesis, elapsed_seconds=0.0):
        self.inference_seconds += elapsed_seconds
        if (self.state in self.terminal_states or task.session_id != self.session_id or
                task.epoch_id != self.epoch_id or task.generation_id != self._active_generation):
            self.emit("Warning", reason="stale_result_rejected", generation_id=task.generation_id)
            return False
        self._active_generation = None
        self.audio.mark_used(task.snapshot.end_sample)
        if hypothesis.model_kind != self.model_kind:
            self.fail("backend_identity_mismatch", "Backend result has an inconsistent model_kind")
            return False
        if any(cue.start_sample < task.snapshot.start_sample or
               cue.end_sample > task.snapshot.end_sample + 320 or cue.end_sample <= cue.start_sample
               for cue in hypothesis.cues):
            self.fail("invalid_backend_timestamp", "Backend cues exceed the supplied immutable snapshot")
            return False
        self.emit("Hypothesis", generation_id=task.generation_id, snapshot_start=task.snapshot.start_sample,
                  snapshot_end=task.snapshot.end_sample, audio_sha256=task.snapshot.digest,
                  history_hash=task.history_hash, prompt_hash=task.prompt_hash, policy_version=task.policy_version,
                  raw_text=hypothesis.raw_text, parse_status=hypothesis.parse_status,
                  issues=list(hypothesis.issues), stop_reason=hypothesis.stop_reason,
                  generated_tokens=hypothesis.generated_tokens, elapsed_seconds=elapsed_seconds,
                  cues=[vars(cue) for cue in hypothesis.cues], metadata=hypothesis.metadata)
        text = hypothesis.target_text
        if not task.voiced_intervals and text:
            self.fail("silence_hallucination", "Text was produced for an observed silent snapshot")
            return False
        if task.final:
            if (hypothesis.parse_status != "complete" or hypothesis.stop_reason != "eos" or
                    not covers_observed_speech(hypothesis.cues, task.voiced_intervals)):
                self.fail("incomplete_final", "Final output failed parsing/acoustic coverage; audio retained")
                return False
            self._set_output(text, final=True)
            if hypothesis.source_text or text:
                history = (*self.context, hypothesis.source_text + "\n" + text)
                self.context = history[-self.config.context_windows:] if self.config.context_windows else ()
                while sum(map(len, self.context)) > 6000:
                    self.context = self.context[1:]
            boundary = task.snapshot.end_sample
            self.audio.evict(boundary, authorized_through=boundary)
            self.vad.evict(boundary)
            self.emit("Boundary", reason="observed_pause_and_completed_output" if not task.terminal else "end_drained",
                      source_start_sample=self.epoch_start, source_end_sample=boundary,
                      alignment_status="model_prediction_with_acoustic_gate", overlap_samples=0)
            self.epoch_start, self.epoch_id = boundary, self.epoch_id + 1
            self.epoch_committed = self.draft = ""
            if task.terminal:
                self._close()
            elif self.state != "DRAINING":
                self.state = "RECEIVING"
        else:
            if hypothesis.parse_status in ("complete", "partial"):
                key = (self.epoch_id, task.history_hash, task.prompt_hash, task.model_revision, task.policy_version)
                candidate = self.policy.observe(text, task.snapshot.end_sample, key)
                if not text.startswith(self.epoch_committed):
                    self.emit("Warning", reason="committed_prefix_conflict", candidate_text=text)
                    self._set_output(self.epoch_committed, draft=text, conflict=True)
                else:
                    committed = candidate if len(candidate) > len(self.epoch_committed) else self.epoch_committed
                    self._set_output(committed, draft=text[len(committed):])
            if self.state != "DRAINING":
                self.state = "RECEIVING"
        return True

    def _set_output(self, committed, draft="", final=False, conflict=False):
        previous = self.epoch_committed
        commit_id = f"{self.session_id}:{self.epoch_id}"
        self.output_version += 1
        if not committed.startswith(previous):
            if not final:
                raise RuntimeError("Only explicit final Correction can rewrite a committed prefix")
            kind, payload = "Correction", {"commit_id": commit_id, "replacement_text": committed,
                                            "reason": "final_retranslation_conflict"}
        elif committed != previous:
            kind, payload = "CommitAppend", {"commit_id": commit_id, "text": committed[len(previous):],
                                              "reason": "final_boundary" if final else "local_agreement"}
        else:
            kind, payload = "DraftSnapshot", {"reason": "provisional_update"}
        if committed != previous:
            if not previous:
                self.commits.append({"commit_id": commit_id, "text": committed})
            else:
                self.commits[-1]["text"] = committed
            if self.first_commit_seconds is None and committed:
                self.first_commit_seconds = self.clock() - self.started_at
        self.epoch_committed, self.draft = committed, draft
        if self.first_draft_seconds is None and draft:
            self.first_draft_seconds = self.clock() - self.started_at
        self.emit(kind, **payload, committed_text="\n".join(item["text"] for item in self.commits),
                  draft_text=draft, conflict=conflict, final_boundary=final)

    def _close(self):
        self.state = "CLOSED"
        duration = self.audio.received_end / SAMPLE_RATE
        self.emit("StreamEnd", status="COMPLETE", complete=True, final_end_sample=self.audio.received_end,
                  committed_text="\n".join(item["text"] for item in self.commits),
                  metrics={"unique_audio_seconds": duration, "inference_seconds": self.inference_seconds,
                           "stream_compute_RTF": self.inference_seconds / duration if duration else None,
                           "first_draft_wall_seconds": self.first_draft_seconds,
                           "first_commit_wall_seconds": self.first_commit_seconds,
                           "semantic_commit_latency": None})
