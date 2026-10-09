from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from s2tt.core.session import SessionConfig
from s2tt.types import ModelBlocked


def build_backend(args, settings):
    if args.backend == "mock":
        from s2tt.backends.mock import MockBackend
        return MockBackend()
    if not args.model_dir:
        raise ModelBlocked("Provide --model-dir or ECHO_MODEL_DIR with the complete S2TT package")
    from s2tt.backends.index_echo import IndexEchoBackend
    backend = settings.get("backend", {})
    try:
        return IndexEchoBackend(args.model_dir, args.size or backend.get("size", "2B"),
                                args.device or backend.get("device", "cuda:0"),
                                args.max_new_tokens or backend.get("max_new_tokens", 512))
    except ImportError as error:
        raise ModelBlocked(f"Real-model runtime dependency unavailable: {error}") from error


def configuration(args):
    settings = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    session = settings.get("session", {}).copy()
    for name in ("source_language", "target_language", "policy"):
        if getattr(args, name, None) is not None:
            session[name] = getattr(args, name)
    if "glossary" in session:
        session["glossary"] = tuple(session["glossary"])
    return settings, SessionConfig(**session)


def parser():
    root = argparse.ArgumentParser(description="Index-Echo causal speech translation")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("serve", "replay", "probe"):
        command = commands.add_parser(name)
        command.add_argument("--backend", choices=("echo", "mock"), default="echo")
        command.add_argument("--model-dir", default=os.environ.get("ECHO_MODEL_DIR"))
        command.add_argument("--size", choices=("2B", "9B"))
        command.add_argument("--device")
        command.add_argument("--max-new-tokens", type=int)
        command.add_argument("--config")
        command.add_argument("--source-language")
        command.add_argument("--target-language")
        command.add_argument("--policy", choices=("la2", "boundary"))
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8765)
            command.add_argument("--origin", action="append")
            command.add_argument("--report-dir", default="reports/live")
        else:
            command.add_argument("input", type=Path)
            command.add_argument("--out", required=True, type=Path)
            if name == "replay":
                command.add_argument("--mode", choices=("causal_fast", "wallclock_1x"), default="causal_fast")
                command.add_argument("--packet-ms", type=int, default=40)
            else:
                command.add_argument("--prefix-seconds", default="2,4,8,full")
    web = commands.add_parser("web")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8080)
    summary = commands.add_parser("summarize")
    summary.add_argument("trace", type=Path)
    score = commands.add_parser("score")
    score.add_argument("--hypotheses", required=True, type=Path)
    score.add_argument("--references", required=True, type=Path)
    return root


def run_probe(args, backend, config):
    import time
    import wave

    from s2tt.audio.normalizer import AudioNormalizer
    from s2tt.audio.vad import EnergyVAD
    from s2tt.core.session import fingerprint
    from s2tt.evaluation.trace import TraceWriter
    from s2tt.types import AudioSnapshot, DecodeTask, SAMPLE_RATE

    with wave.open(str(args.input), "rb") as recording:
        if recording.getsampwidth() != 2:
            raise ValueError("Probe input must be PCM16 WAV")
        normalizer = AudioNormalizer(recording.getframerate(), recording.getnchannels())
        pieces = []
        while pcm := recording.readframes(recording.getframerate()):
            pieces.append(normalizer.push_pcm16(pcm))
        pieces.append(normalizer.flush())
    import numpy as np
    waveform = np.concatenate(pieces)
    writer = TraceWriter(args.out)
    count = 0
    try:
        for value in args.prefix_seconds.split(","):
            end = len(waveform) if value == "full" else min(round(float(value) * SAMPLE_RATE), len(waveform))
            if end <= 0:
                continue
            snapshot = AudioSnapshot(0, end, waveform[:end])
            vad = EnergyVAD(config.vad_threshold, config.silence_seconds)
            vad.push(snapshot.samples, final=True)
            task = DecodeTask("offline-prefix-probe", 0, count, snapshot, True, True, (), config.glossary,
                              config.source_language, config.target_language, fingerprint(()),
                              fingerprint(config.glossary), backend.revision, "probe-no-policy", vad.intervals(0, end),
                              time.monotonic())
            result = backend.infer(task)
            writer.write({"input_mode": "offline_prefix_probe", "prefix_samples": end,
                          "audio_sha256": snapshot.digest, "model_kind": backend.model_kind,
                          "model_revision": backend.revision, "raw_text": result.raw_text,
                          "source_text": result.source_text, "target_text": result.target_text,
                          "parse_status": result.parse_status, "issues": result.issues,
                          "stop_reason": result.stop_reason, "metadata": result.metadata})
            count += 1
    finally:
        writer.close()
    return {"status": "EXECUTED", "model_kind": backend.model_kind, "probes": count,
            "semantic_quality_verified": False, "trace": str(args.out)}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "web":
            from functools import partial
            from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
            from importlib.resources import files
            directory = str(files("s2tt").joinpath("web"))
            with ThreadingHTTPServer((args.host, args.port), partial(SimpleHTTPRequestHandler, directory=directory)) as http:
                print(f"Browser demo: http://{args.host}:{args.port}", flush=True)
                http.serve_forever()
            return 0
        if args.command == "summarize":
            from s2tt.evaluation.trace import summarize_trace
            print(json.dumps(summarize_trace(args.trace), ensure_ascii=False, indent=2))
            return 0
        if args.command == "score":
            from s2tt.evaluation.quality import score_text_files
            print(json.dumps(score_text_files(args.hypotheses, args.references), ensure_ascii=False, indent=2))
            return 0
        settings, config = configuration(args)
        backend = build_backend(args, settings)
        if args.command == "serve":
            from s2tt.transports.websocket import WebSocketService
            service = WebSocketService(backend, config, report_dir=args.report_dir,
                                       auth_token=os.environ.get("S2TT_AUTH_TOKEN"))
            origins = [None, *(args.origin or ["http://127.0.0.1:8080", "http://localhost:8080"])]
            print(f"Backend={backend.model_kind}; WebSocket ws://{args.host}:{args.port}", flush=True)
            asyncio.run(service.serve(args.host, args.port, origins))
            return 0
        if args.command == "replay":
            from s2tt.evaluation.replay import replay_wav
            result = asyncio.run(replay_wav(args.input, backend, config, args.out,
                                           mode=args.mode, packet_ms=args.packet_ms))
        else:
            result = run_probe(args, backend, config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("status") == "FAIL" else 0
    except ModelBlocked as error:
        print(json.dumps({"status": "BLOCKED", "model_kind": "real", "reason": str(error)}, ensure_ascii=False))
        return 2
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, ImportError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
