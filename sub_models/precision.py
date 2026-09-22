"""Select AMP precision, with an optional per-process override."""

import os
import torch


def get_amp_dtype():
    """Honor STORM_AMP_DTYPE or inspect the current visible CUDA device."""
    requested = os.environ.get("STORM_AMP_DTYPE", "auto").strip().lower()
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if requested != "auto":
        raise ValueError(
            f"Invalid STORM_AMP_DTYPE={requested!r}; choose auto, fp16, or bf16"
        )
    if torch.cuda.is_available():
        gpu_name = " ".join(torch.cuda.get_device_name().upper().split())
        if gpu_name in ("TITAN RTX", "NVIDIA TITAN RTX"):
            return torch.float16
    return torch.bfloat16
