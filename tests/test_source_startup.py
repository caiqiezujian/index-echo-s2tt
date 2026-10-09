import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]


def source_process(arguments, cwd):
    # -S skips editable-install .pth files; expose dependencies without processing those files.
    bootstrap = (
        "import runpy, sys\n"
        "sys.path.append(sys.argv.pop(1))\n"
        "sys.argv = sys.argv[1:]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-S", "-c", bootstrap, sysconfig.get_path("purelib"),
                           str(ROOT / "run_s2tt.py"), *arguments], cwd=cwd, env=environment,
                          capture_output=True, text=True, timeout=15)


def test_source_help_from_another_directory_without_editable_install(tmp_path):
    result = source_process(["--help"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert all(command in result.stdout for command in ("serve", "web", "replay", "probe"))


def test_source_probe_without_project_installation(tmp_path, speech_pcm):
    recording = tmp_path / "recording.wav"
    with wave.open(str(recording), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(speech_pcm(0.2))
    configuration = tmp_path / "local.json"
    configuration.write_text(json.dumps({"backend": {"kind": "mock"}, "probe": {
        "input": "recording.wav", "out": "probe.jsonl", "prefix_seconds": "full"}}), encoding="utf-8")
    result = source_process(["probe", "--config", str(configuration)], tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model_kind"] == "mock"
    assert json.loads((tmp_path / "probe.jsonl").read_text(encoding="utf-8"))["prefix_samples"] == 3200


def test_source_missing_dependency_reports_package_without_installation(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-S", str(ROOT / "run_s2tt.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert "Missing Python dependency: numpy" in result.stderr


def bash_executable():
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            root = Path(git).resolve().parents[1]
            for candidate in (root / "bin/bash.exe", root / "usr/bin/bash.exe"):
                if candidate.is_file():
                    return str(candidate)
        pytest.skip("Bash launch contract requires Git Bash on Windows or Bash on Linux")
    executable = shutil.which("bash")
    if not executable:
        pytest.skip("Bash launch contract requires Bash")
    return executable


@pytest.mark.parametrize("script,command", [("start_server.sh", "serve"), ("start_web.sh", "web")])
@pytest.mark.parametrize("custom", [False, True])
def test_bash_uses_python3_and_preserves_config_and_arguments(tmp_path, script, command, custom):
    executable = bash_executable()
    bin_directory = tmp_path / "container bin"
    bin_directory.mkdir()
    observer = tmp_path / "observe.py"
    observer.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['STARTUP_REPORT']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n",
        encoding="utf-8")
    shim = bin_directory / "python3"
    shim.write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(Path(sys.executable).as_posix())} "
                    f"{shlex.quote(observer.as_posix())} \"$@\"\n", encoding="utf-8")
    shim.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = str(bin_directory) + os.pathsep + environment.get("PATH", "")
    report = tmp_path / "arguments.json"
    environment["STARTUP_REPORT"] = str(report)
    arguments = []
    if custom:
        configuration = tmp_path / "configuration with spaces.json"
        arguments = [str(configuration), "--port", "18001"]
    result = subprocess.run([executable, str(ROOT / "deploy" / script), *arguments],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    received = json.loads(report.read_text(encoding="utf-8"))
    assert received[0] == "-u" and Path(received[1]).resolve() == ROOT / "run_s2tt.py"
    assert received[2:4] == [command, "--config"]
    expected = configuration if custom else ROOT / "configs/5090_2b.json"
    assert Path(received[4]).resolve() == expected
    assert received[5:] == (["--port", "18001"] if custom else [])
