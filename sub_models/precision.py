"""Keep the original BF16 policy except on TITAN RTX."""

import torch


def get_amp_dtype():
    """Inspect the current CUDA device (respecting CUDA_VISIBLE_DEVICES)."""
    if torch.cuda.is_available():
        gpu_name = " ".join(torch.cuda.get_device_name().upper().split())
        if gpu_name in ("TITAN RTX", "NVIDIA TITAN RTX"):
            return torch.float16
    return torch.bfloat16
