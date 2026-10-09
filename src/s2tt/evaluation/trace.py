import json
from pathlib import Path


class TraceWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8")

    def write(self, event):
        self.stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


def summarize_trace(path):
    last_seq = 0
    model_kinds, terminal, drafts, commits, corrections, errors = set(), None, 0, 0, 0, []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event_seq"] <= last_seq:
                raise ValueError("Nonmonotonic event sequence")
            last_seq = event["event_seq"]
            if not event["evict_before_sample"] <= event["used_audio_end_sample"] <= event["received_end_sample"]:
                raise ValueError("Audio watermark invariant violated")
            model_kinds.add(event["model_kind"])
            drafts += event["type"] == "DraftSnapshot"
            commits += event["type"] == "CommitAppend"
            corrections += event["type"] == "Correction"
            if event["type"] == "Error":
                errors.append(event)
            if event["type"] == "StreamEnd":
                terminal = event
    return {"model_kinds": sorted(model_kinds), "events": last_seq, "draft_events": drafts,
            "commit_events": commits, "correction_events": corrections, "errors": errors,
            "terminal": terminal, "semantic_quality_verified": False,
            "status": "FAIL" if errors else "COMPLETE" if terminal and terminal.get("complete") else "INCOMPLETE"}
