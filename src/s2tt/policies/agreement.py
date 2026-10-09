from __future__ import annotations

from collections import deque
from os.path import commonprefix


class LocalAgreement:
    version = "la2-unicode-v1"

    def __init__(self, confirmations=2, holdback_chars=6, target_language="zh"):
        if confirmations < 2 or holdback_chars < 0:
            raise ValueError("At least two confirmations and a nonnegative holdback are required")
        self.confirmations, self.holdback_chars = confirmations, holdback_chars
        self.target_language = target_language
        self._history = deque(maxlen=confirmations)
        self._key = None
        self._last_end = -1

    def observe(self, text, used_end, key):
        if key != self._key:
            self._history.clear()
            self._last_end = -1
            self._key = key
        if used_end <= self._last_end:
            return ""
        self._last_end = used_end
        self._history.append(text)
        if len(self._history) < self.confirmations:
            return ""
        prefix = commonprefix(list(self._history))
        prefix = prefix[:max(0, len(prefix) - self.holdback_chars)]
        if self.target_language != "zh" and prefix and not prefix[-1].isspace():
            prefix = prefix.rsplit(" ", 1)[0] if " " in prefix else ""
        return prefix


class BoundaryOnly:
    version = "boundary-only-v1"

    def observe(self, text, used_end, key):
        return ""
