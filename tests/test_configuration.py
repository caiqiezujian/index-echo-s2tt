import builtins
import json
import os
import subprocess
import sys
import wave

import pytest

from s2tt.cli import build_backend, configuration, main, parser
from s2tt.configuration import load_settings


def profile(tmp_path, settings):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path


def test_relative_paths_use_config_directory_after_cwd_changes(tmp_path, monkeypatch):
    path = profile(tmp_path, {"backend": {"model_dir": "models/Echo"},
                              "server": {"report_dir": "reports/live"},
                              "replay": {"input": "recording.wav", "out": "reports/replay.jsonl"}})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    settings = load_settings(path)
    assert settings["backend"]["model_dir"] == str(tmp_path / "models/Echo")
    assert settings["server"]["report_dir"] == str(tmp_path / "reports/live")
    assert settings["replay"]["input"] == str(tmp_path / "recording.wav")


def test_config_overrides_environment_and_cli_overrides_config(tmp_path, monkeypatch):
    import s2tt.backends.index_echo as implementation
    observed = []
    monkeypatch.setattr(implementation, "IndexEchoBackend", lambda *args: observed.append(args))
    monkeypatch.setenv("ECHO_MODEL_DIR", str(tmp_path / "environment"))
    path = profile(tmp_path, {"backend": {"model_dir": "configured", "size": "9B", "device": "cuda:1",
                                         "max_new_tokens": 256}, "session": {"policy": "boundary"}})
    args = parser().parse_args(["serve", "--config", str(path)])
    settings, session = configuration(args)
    build_backend(args, settings)
    assert observed.pop() == (tmp_path / "configured", "9B", "cuda:1", 256)
    assert session.policy == "boundary"
    args = parser().parse_args(["serve", "--config", str(path), "--model-dir", str(tmp_path / "explicit"),
                                "--size", "2B", "--max-new-tokens", "64", "--policy", "la2"])
    settings, session = configuration(args)
    build_backend(args, settings)
    assert observed.pop() == (tmp_path / "explicit", "2B", "cuda:1", 64)
    assert session.policy == "la2"
    args = parser().parse_args(["serve"])
    build_backend(args, {})
    assert observed.pop()[0] == tmp_path / "environment"


@pytest.mark.parametrize("settings", [
    [], {"backned": {}}, {"backend": {"model_dri": "models"}}, {"server": []},
    {"backend": {"model_dir": "https://modelscope.cn/models/IndexTeam/Echo"}},
    {"backend": {"kind": "api"}}, {"backend": {"size": "7B"}},
    {"server": {"port": "8765"}}, {"server": {"port": 0}}, {"server": {"port": True}},
    {"server": {"origins": [1]}}, {"server": {"idle_timeout": 0}},
    {"session": {"glossary": "a:b"}}, {"replay": {"mode": "noncausal"}}, {"replay": {"packet_ms": 0}},
    {"session": {"update_seconds": float("nan")}},
])
def test_invalid_profile_fails_before_model_loading(tmp_path, settings):
    with pytest.raises(ValueError):
        load_settings(profile(tmp_path, settings))


def test_zero_token_override_is_rejected_instead_of_using_default(tmp_path, capsys):
    path = profile(tmp_path, {"backend": {"model_dir": "models"}})
    assert main(["serve", "--config", str(path), "--max-new-tokens", "0"]) == 1
    assert "max_new_tokens" in capsys.readouterr().err


def test_service_receives_profile_and_explicit_port_override(tmp_path, monkeypatch):
    from s2tt.transports import websocket
    observed = {}

    class RecordingService:
        def __init__(self, backend, config, **kwargs):
            observed.update(model_kind=backend.model_kind, config=config, **kwargs)

        async def serve(self, host, port, origins):
            observed.update(host=host, port=port, origins=origins)

    monkeypatch.setattr(websocket, "WebSocketService", RecordingService)
    path = profile(tmp_path, {"backend": {"kind": "mock"}, "server": {
        "host": "127.0.0.1", "port": 19001, "report_dir": "traces", "idle_timeout": 42,
        "origins": ["http://localhost:19002"]}, "session": {"glossary": ["Echo:回声"]}})
    assert main(["serve", "--config", str(path), "--port", "19003"]) == 0
    assert observed["model_kind"] == "mock" and observed["port"] == 19003
    assert observed["report_dir"] == str(tmp_path / "traces")
    assert observed["idle_timeout"] == 42
    assert observed["origins"] == [None, "http://localhost:19002"]
    assert observed["config"].glossary == ("Echo → 回声",)


def test_config_only_probe_runs_with_all_socket_connections_forbidden(tmp_path, speech_pcm):
    recording = tmp_path / "recording.wav"
    with wave.open(str(recording), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(speech_pcm(0.3))
    path = profile(tmp_path, {"backend": {"kind": "mock"}, "probe": {
        "input": "recording.wav", "out": "probe.jsonl", "prefix_seconds": "full"}})
    script = (
        "import socket, sys\n"
        "def deny(*args, **kwargs): raise AssertionError('Unexpected outbound network connection')\n"
        "socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny\n"
        "from s2tt.cli import main\n"
        "sys.exit(main(['probe', '--config', sys.argv[1]]))\n"
    )
    process = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["model_kind"] == "mock" and result["semantic_quality_verified"] is False
    trace = json.loads((tmp_path / "probe.jsonl").read_text(encoding="utf-8"))
    assert trace["model_kind"] == "mock" and trace["prefix_samples"] == 4800


def test_real_missing_local_package_is_blocked_without_network(tmp_path, monkeypatch, capsys):
    import socket

    def deny(*args, **kwargs):
        raise AssertionError("Missing local models must not trigger a download")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    path = profile(tmp_path, {"backend": {"model_dir": "missing-package"},
                              "probe": {"input": "recording.wav", "out": "trace.jsonl"}})
    assert main(["probe", "--config", str(path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "BLOCKED" and result["model_kind"] == "real"
    assert not (tmp_path / "trace.jsonl").exists()


def test_offline_flags_are_set_before_framework_imports(tmp_path, monkeypatch):
    import s2tt.backends.index_echo as implementation

    flags = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
    for name in flags:
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(implementation, "verify_package", lambda *args: {"revision": "test-no-weights"})
    original_import = builtins.__import__

    class FrameworkImportReached(Exception):
        pass

    def intercept(name, *args, **kwargs):
        if name == "torch":
            assert all(os.environ[key] == "1" for key in flags)
            raise FrameworkImportReached
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", intercept)
    with pytest.raises(FrameworkImportReached):
        implementation.IndexEchoBackend(tmp_path)


def test_cli_replay_can_take_input_and_output_from_json(tmp_path, speech_pcm, capsys):
    recording = tmp_path / "recording.wav"
    with wave.open(str(recording), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(speech_pcm(0.3))
    path = profile(tmp_path, {"backend": {"kind": "mock"}, "session": {"update_seconds": 0.1},
                              "replay": {"input": "recording.wav", "out": "replay.jsonl", "packet_ms": 100}})
    assert main(["replay", "--config", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "COMPLETE" and result["model_kinds"] == ["mock"]
    events = [json.loads(line) for line in (tmp_path / "replay.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "StreamEnd" and events[-1]["complete"]
