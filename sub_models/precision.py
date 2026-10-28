"""Select AMP precision, with an optional per-process override."""

import os
import random
from contextlib import contextmanager

import numpy as np
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


def evaluation_precisions(precision):
    if precision == "both":
        return ("bf16", "fp32")
    if precision not in ("auto", "bf16", "fp32"):
        raise ValueError("Evaluation precision must be auto, bf16, fp32, or both")
    return (precision,)


@contextmanager
def evaluation_context(world_model, agent, precision="auto", seed=None):
    """Temporarily select inference precision without changing training state."""
    if precision not in ("auto", "bf16", "fp32"):
        raise ValueError("A single evaluation requires auto, bf16, or fp32 precision")
    if seed is not None and not (0 <= seed <= 2**32 - 1):
        raise ValueError("Evaluation seed must be between 0 and 2**32 - 1")

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    matmul_precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    models = (world_model, agent)
    settings = [(model, model.use_amp, model.amp_dtype, model.tensor_dtype)
                for model in models]
    modes = [(module, module.training) for model in models for module in model.modules()]
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if cuda_rng is not None:
                torch.cuda.manual_seed_all(seed)
        if precision != "auto":
            # Keep TF32 disabled in both arms of the precision comparison.
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
            for model in models:
                model.use_amp = precision == "bf16"
                model.amp_dtype = torch.bfloat16
                model.tensor_dtype = torch.bfloat16 if model.use_amp else torch.float32
        for model in models:
            model.eval()
        yield
    finally:
        for model, use_amp, amp_dtype, tensor_dtype in settings:
            model.use_amp = use_amp
            model.amp_dtype = amp_dtype
            model.tensor_dtype = tensor_dtype
        for module, training in modes:
            module.training = training
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
