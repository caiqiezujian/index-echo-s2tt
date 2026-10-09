from s2tt.types import ModelBlocked


class CUDAProvider:
    def __init__(self, device="cuda:0"):
        import torch

        if not device.startswith("cuda:") or not torch.cuda.is_available():
            raise ModelBlocked("CUDA device unavailable; CPU fallback is disabled")
        if torch.__version__.split("+")[0] != "2.11.0":
            raise ModelBlocked("Echo reference runtime requires torch 2.11.0")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ModelBlocked("Selected CUDA device does not support BF16")
        self.torch, self.device, self.dtype = torch, device, torch.bfloat16

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def reset_peak(self):
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def metadata(self):
        return {
            "device": self.device,
            "device_name": self.torch.cuda.get_device_name(self.device),
            "dtype": "bfloat16",
            "torch": self.torch.__version__,
            "cuda_runtime": self.torch.version.cuda,
            "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
            "cpu_fallback": False,
        }
