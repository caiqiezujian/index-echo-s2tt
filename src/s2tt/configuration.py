"""Read and validate local JSON settings without contacting a model registry."""
from __future__ import annotations

import json
import math
from pathlib import Path


SECTIONS = {
    "backend": {"kind": str, "model_dir": str, "size": str, "device": str, "max_new_tokens": int},
    "session": {
        "source_language": str, "target_language": str, "sample_rate": int, "channels": int,
        "update_seconds": (int, float), "max_buffer_seconds": (int, float),
        "silence_seconds": (int, float), "vad_threshold": (int, float), "policy": str,
        "holdback_chars": int, "context_windows": int, "glossary": list,
    },
    "server": {"host": str, "port": int, "origins": list, "report_dir": str, "idle_timeout": (int, float)},
    "web": {"host": str, "port": int, "websocket_url": str},
    "replay": {"input": str, "out": str, "mode": str, "packet_ms": int},
    "probe": {"input": str, "out": str, "prefix_seconds": str},
}
PATH_FIELDS = {"backend": ("model_dir",), "server": ("report_dir",),
               "replay": ("input", "out"), "probe": ("input", "out")}


def local_path(value, *, base=None):
    """Configuration paths resolve against the file; CLI paths use the current directory."""
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise ValueError("Use a nonempty local filesystem path, not a URL")
    path = Path(value).expanduser()
    if base is not None and not path.is_absolute():
        path = base / path
    return path.resolve()


def load_settings(config_path):
    if config_path is None:
        return {}
    path = local_path(str(config_path))
    settings = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a JSON object")
    for section, values in settings.items():
        if section not in SECTIONS or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid configuration section: {section}")
        for key, value in values.items():
            expected = SECTIONS[section].get(key)
            if expected is None:
                raise ValueError(f"Unknown configuration field: {section}.{key}")
            if not isinstance(value, expected) or isinstance(value, bool):
                raise ValueError(f"Invalid type for configuration field: {section}.{key}")
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{section}.{key} must be finite")
            if isinstance(value, list) and not all(isinstance(item, str) for item in value):
                raise ValueError(f"{section}.{key} must be a list of strings")
        for key in PATH_FIELDS.get(section, ()):
            if key in values:
                values[key] = str(local_path(values[key], base=path.parent))
    backend = settings.get("backend", {})
    if backend.get("kind", "echo") not in ("echo", "mock"):
        raise ValueError("backend.kind must be echo or mock")
    if backend.get("size", "2B") not in ("2B", "9B"):
        raise ValueError("backend.size must be 2B or 9B")
    for section in ("server", "web"):
        if not 1 <= settings.get(section, {}).get("port", 8765) <= 65535:
            raise ValueError(f"{section}.port must be between 1 and 65535")
    if settings.get("server", {}).get("idle_timeout", 60) <= 0:
        raise ValueError("server.idle_timeout must be positive")
    replay = settings.get("replay", {})
    if replay.get("mode", "causal_fast") not in ("causal_fast", "wallclock_1x"):
        raise ValueError("Invalid replay.mode")
    if not 1 <= replay.get("packet_ms", 40) <= 2000:
        raise ValueError("replay.packet_ms must be between 1 and 2000")
    return settings


def option(args, settings, section, field, default=None, *, argument=None):
    value = getattr(args, argument or field, None)
    return value if value is not None else settings.get(section, {}).get(field, default)
