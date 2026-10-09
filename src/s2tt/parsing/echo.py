from __future__ import annotations

import re

from s2tt.types import AudioSnapshot, Cue, Hypothesis

STAMP = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)-(\d+):(\d+(?:\.\d+)?)\]$")


def parse_echo(raw_text: str, snapshot: AudioSnapshot, *, stop_reason="unknown", model_kind="real",
               generated_tokens=None, metadata=None):
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    cues, issues = [], []
    last_complete_line_end = -1
    index = 0
    while index < len(lines):
        match = STAMP.fullmatch(lines[index])
        if not match:
            issues.append(f"unexpected_line:{index}")
            index += 1
            continue
        if index + 2 >= len(lines) or STAMP.fullmatch(lines[index + 1]) or STAMP.fullmatch(lines[index + 2]):
            issues.append(f"incomplete_cue:{index}")
            index += 1
            continue
        minute_a, second_a, minute_b, second_b = match.groups()
        begin = int(minute_a) * 60 + float(second_a)
        end = int(minute_b) * 60 + float(second_b)
        a = snapshot.start_sample + round(begin * snapshot.sample_rate)
        b = snapshot.start_sample + round(end * snapshot.sample_rate)
        if float(second_a) >= 60 or float(second_b) >= 60 or b <= a:
            issues.append(f"invalid_timestamp:{index}")
        elif a < snapshot.start_sample or b > snapshot.end_sample + round(0.02 * snapshot.sample_rate):
            issues.append(f"outside_snapshot:{index}")
        elif cues and a < cues[-1].end_sample:
            issues.append(f"overlapping_or_reversed_cue:{index}")
        else:
            # Preserve predicted boundaries; the 20ms tolerance is reported, never clamped into alignment truth.
            cues.append(Cue(a, b, lines[index + 1], lines[index + 2]))
            last_complete_line_end = index + 3
        index += 3
    if stop_reason == "length":
        issues.append("generation_length_limit")
        if cues and last_complete_line_end == len(lines):
            cues.pop()  # Last target line may itself be truncated; do not stabilize it.
    incomplete = any(issue.startswith("incomplete_cue") or issue == "generation_length_limit" for issue in issues)
    status = "complete" if cues and not issues else "partial" if cues or incomplete else "invalid"
    return Hypothesis(raw_text, tuple(cues), status, tuple(issues), stop_reason,
                      generated_tokens, model_kind, metadata or {})


def covers_observed_speech(cues, voiced_intervals, tolerance_samples=4000):
    """Acoustic coverage gate, explicitly not proof of semantic completeness."""
    for begin, end in voiced_intervals:
        cursor = begin
        for cue in cues:
            if cue.start_sample - tolerance_samples <= cursor and cue.end_sample + tolerance_samples > cursor:
                cursor = max(cursor, cue.end_sample + tolerance_samples)
        if cursor < end:
            return False
    return True
