"""Array adapter based on IndexTeam's Apache-2.0 S2TT loading/generation recipe.

See THIRD_PARTY_NOTICES.md and configs/upstream_manifest.json for provenance.
No file-wide VAD, temporary WAV, or cross-update cache is used here.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from importlib.resources import files
from pathlib import Path

from s2tt.parsing.echo import parse_echo
from s2tt.types import Hypothesis, ModelBlocked

LANGUAGE_NAMES = {"en": "English", "zh": "Chinese", "ja": "Japanese", "es": "Spanish"}


def enable_offline_runtime():
    # Set before importing Transformers / Hub: their offline flags are cached at import.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"


def verify_package(root: Path, size: str, manifest_path=None):
    manifest = (Path(manifest_path).read_text(encoding="utf-8") if manifest_path
                else files("s2tt").joinpath("resources/model_snapshots.json").read_text(encoding="utf-8"))
    snapshot = json.loads(manifest)["snapshots"][size]
    for entry in snapshot["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ModelBlocked(f"Missing or incorrect model artifact: {entry['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ModelBlocked(f"Model artifact checksum mismatch: {entry['path']}")
    return snapshot


class IndexEchoBackend:
    model_kind = "real"

    def __init__(self, model_dir, size="2B", device="cuda:0", max_new_tokens=512, manifest_path=None):
        self.root = Path(model_dir).resolve()
        if not self.root.is_dir():
            raise ModelBlocked(f"Model directory unavailable: {self.root}")
        snapshot = verify_package(self.root, size, manifest_path)
        self.revision, self.size = snapshot["revision"], size
        self.max_new_tokens = max_new_tokens
        if not 32 <= max_new_tokens <= 8192:
            raise ValueError("max_new_tokens must be between 32 and 8192")
        enable_offline_runtime()
        import torch
        import transformers
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoderConfig,
        )

        from s2tt.devices.cuda import CUDAProvider

        if transformers.__version__ != "5.6.0":
            raise ModelBlocked("Echo reference runtime requires transformers 5.6.0")
        self.provider = CUDAProvider(device)
        self.torch, self.device, self.dtype = torch, device, self.provider.dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.root / "llm", local_files_only=True)
        self.llm, loading = AutoModelForCausalLM.from_pretrained(
            self.root / "llm", dtype=self.dtype, local_files_only=True, output_loading_info=True,
        )
        missing = loading.get("missing_keys", [])
        if missing == ["lm_head.weight"] and getattr(self.llm.config, "tie_word_embeddings", False):
            if self.llm.get_input_embeddings().weight.data_ptr() == self.llm.get_output_embeddings().weight.data_ptr():
                missing = []
        if missing or loading.get("unexpected_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ModelBlocked(f"Strict decoder loading failed: {loading}")
        expected_dim = 2048 if size == "2B" else 4096
        if self.llm.config.hidden_size != expected_dim:
            raise ModelBlocked(f"Unexpected {size} decoder hidden size")
        self.llm.eval().to(device)
        config = json.loads((self.root / "audio_config.json").read_text(encoding="utf-8"))
        self.tower = Qwen3OmniMoeAudioEncoder(Qwen3OmniMoeAudioEncoderConfig(**config)).eval()
        self.tower.load_state_dict(load_file(str(self.root / "audio_tower.safetensors")), strict=True)
        self.tower.to(device, self.dtype)
        connector_state = load_file(str(self.root / "connector.safetensors"))
        self.connector = self._connector(connector_state, expected_dim).eval().to(device, self.dtype)
        self.connector.load_state_dict(connector_state, strict=True)
        self.extractor = WhisperFeatureExtractor(
            feature_size=128, hop_length=160, n_fft=400, sampling_rate=16000,
            padding_value=0.0, return_attention_mask=True,
        )
        self.extractor.n_samples, self.extractor.nb_max_frames = 4800000, 30000
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        for token in ("<|audio_pad|>", "<|audio_start|>", "<|audio_end|>", "<|im_end|>"):
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1 or ids[0] == self.tokenizer.unk_token_id:
                raise ModelBlocked(f"Audio/template special token is invalid: {token}")
        self.provider.synchronize()

    def _connector(self, state, dim):
        torch = self.torch

        class ResidualConnector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.log_alpha = torch.nn.Parameter(torch.zeros(()))
                self.w1 = torch.nn.Linear(dim, dim, bias=False)
                self.w2 = torch.nn.Linear(dim, dim, bias=False)
                self.beta = torch.nn.Parameter(torch.zeros(()))

            def forward(self, value):
                return torch.exp(self.log_alpha) * (value + self.beta * self.w2(torch.nn.functional.gelu(self.w1(value))))

        class ProjectionConnector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(2048, dim, bias=False)

            def forward(self, value):
                return self.proj(value)

        expected = {"log_alpha", "w1.weight", "w2.weight", "beta"} if self.size == "2B" else {"proj.weight"}
        if set(state) != expected:
            raise ModelBlocked(f"Unexpected {self.size} connector keys: {sorted(state)}")
        return ResidualConnector() if self.size == "2B" else ProjectionConnector()

    def encode_audio_array(self, samples):
        with self.torch.inference_mode():
            features = self.extractor(samples, sampling_rate=16000,
                                      return_tensors="pt", return_attention_mask=True)
            valid_frames = int(features.attention_mask.sum(-1)[0])
            tower_input = features.input_features[0][:, :valid_frames].to(self.device, self.dtype)
            output = self.tower(input_features=tower_input,
                                feature_lens=self.torch.tensor([valid_frames], device=self.device))
            hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
            return self.connector(hidden.to(self.dtype)), valid_frames

    def infer(self, task):
        if not task.voiced_intervals:
            return Hypothesis("", (), "complete", stop_reason="eos", model_kind="real",
                              metadata={"generation_skipped": "observed_silence"})
        if task.target_language not in LANGUAGE_NAMES:
            raise ValueError("Unsupported target language")
        torch = self.torch
        self.provider.reset_peak()
        started = time.monotonic()
        with torch.inference_mode():
            embedding, valid_frames = self.encode_audio_array(task.snapshot.samples)
            slots = []
            if task.context:
                slots.append("[Context]\n" + "\n".join(task.context))
            if task.glossary:
                slots.append("[Glossary]\n" + "\n".join(task.glossary))
            instruction = ("For each sentence, output three lines: the [MM:SS.CC-MM:SS.CC] timestamp, "
                           f"the transcript, then the {LANGUAGE_NAMES[task.target_language]} translation.")
            audio = "<|audio_start|>" + "<|audio_pad|>" * embedding.shape[0] + "<|audio_end|>"
            content = audio + "\n" + "".join(slot + "\n\n" for slot in slots) + instruction
            prompt = ("<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"
                      "<think>\n\n</think>\n\n")
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            inputs = self.llm.get_input_embeddings()(ids).clone()
            mask = ids == self.pad_id
            if int(mask.sum()) != embedding.shape[0] or embedding.shape[-1] != inputs.shape[-1]:
                raise RuntimeError("Audio placeholder count or connector dimension mismatch")
            inputs[mask] = embedding.to(inputs.dtype)
            generated = self.llm.generate(
                inputs_embeds=inputs, attention_mask=torch.ones_like(ids), max_new_tokens=self.max_new_tokens,
                do_sample=False, eos_token_id=[self.tokenizer.eos_token_id, self.im_end],
                pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True,
            )
            sequence = generated.sequences[0]
            token_count = len(sequence)
            stopped = token_count > 0 and int(sequence[-1]) in (self.tokenizer.eos_token_id, self.im_end)
            reason = "eos" if stopped else "length" if token_count >= self.max_new_tokens else "unknown"
            raw = self.tokenizer.decode(sequence, skip_special_tokens=True).strip()
        self.provider.synchronize()
        metadata = self.provider.metadata() | {
            "model_size": self.size, "model_revision": self.revision, "audio_frames": valid_frames,
            "audio_tokens": len(embedding), "model_elapsed_seconds": time.monotonic() - started,
            "direction_status": "experimental", "cross_update_cache": False,
            "local_files_only": True, "external_api_used": False,
        }
        return parse_echo(raw, task.snapshot, stop_reason=reason, generated_tokens=token_count, metadata=metadata)
