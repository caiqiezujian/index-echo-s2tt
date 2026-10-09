from s2tt.types import Cue, Hypothesis


class MockBackend:
    """Explicit transport/logic demonstration. Does not translate speech."""

    model_kind = "mock"
    revision = "mock-v1"

    def infer(self, task):
        if not task.voiced_intervals:
            return Hypothesis("", (), "complete", stop_reason="eos", model_kind="mock")
        text = "【模拟输出，非翻译】已接收音频。"
        cue = Cue(task.snapshot.start_sample, task.snapshot.end_sample, "[MOCK AUDIO]", text)
        return Hypothesis(text, (cue,), "complete", stop_reason="eos", model_kind="mock")
