"""Check the candidate RTX 5090 environment without loading model weights."""

import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "stage": "environment_smoke_test",
        "model_inference_tested": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
        "checks": [],
    }

    def record(name, passed, detail):
        report["checks"].append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    record("python_3_12", sys.version_info[:2] == (3, 12), platform.python_version())
    for distribution, module_name, pinned in [
        ("torch", "torch", "2.11.0"),
        ("torchaudio", "torchaudio", "2.11.0"),
        ("transformers", "transformers", "5.6.0"),
        ("safetensors", "safetensors", None),
        ("librosa", "librosa", None),
        ("soundfile", "soundfile", None),
        ("silero-vad", "silero_vad", None),
    ]:
        try:
            version = importlib.metadata.version(distribution)
            importlib.import_module(module_name)
            report["packages"][distribution] = version
            record(distribution, pinned is None or version.split("+")[0] == pinned, version)
        except Exception as error:
            record(distribution, False, f"{type(error).__name__}: {error}")

    ffmpeg = shutil.which("ffmpeg")
    record("ffmpeg", ffmpeg is not None, ffmpeg)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            process = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20, check=True,
            )
            report["nvidia_smi"] = process.stdout.strip()
        except Exception as error:
            report["nvidia_smi_error"] = f"{type(error).__name__}: {error}"

    try:
        import torch

        record("torch_cuda_12_8", torch.version.cuda == "12.8", torch.version.cuda)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check GPU start mode, driver and wheel")
        properties = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        report["gpu"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_gib": total_bytes / 1024**3,
            "free_gib": free_bytes / 1024**3,
            "wheel_architectures": torch.cuda.get_arch_list(),
        }
        record("rtx_5090_32gb", "5090" in properties.name and total_bytes >= 30 * 1024**3, report["gpu"])
        record("compute_capability_12_0", torch.cuda.get_device_capability(0) == (12, 0), report["gpu"]["capability"])
        record("bf16_support", torch.cuda.is_bf16_supported(), torch.cuda.is_bf16_supported())
        with torch.inference_mode():
            matrix = torch.randn(128, 128, device="cuda:0", dtype=torch.bfloat16)
            product = matrix @ matrix.T
            query = torch.randn(1, 2, 32, 64, device="cuda:0", dtype=torch.bfloat16)
            attention = torch.nn.functional.scaled_dot_product_attention(query, query, query)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(product).all() and torch.isfinite(attention).all())
        record("bf16_matmul_and_attention", finite, "CUDA operations completed")
    except Exception as error:
        record("cuda_execution", False, f"{type(error).__name__}: {error}")

    try:
        from transformers import AutoConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoderConfig,
        )

        config = AutoConfig.for_model("qwen3_5")
        record("echo_model_classes", True, [
            Qwen3OmniMoeAudioEncoder.__name__, Qwen3OmniMoeAudioEncoderConfig.__name__, type(config).__name__,
        ])
    except Exception as error:
        record("echo_model_classes", False, f"{type(error).__name__}: {error}")

    passed = all(check["status"] == "PASS" for check in report["checks"])
    report["status"] = "PASS" if passed else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
