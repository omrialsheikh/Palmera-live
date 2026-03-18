"""
Device management utilities.
"""

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("CUDA GPU required. No GPU detected.")


def get_dtype(config_dtype: str) -> torch.dtype:
    if config_dtype == "fp16":
        return torch.float16
    elif config_dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def gpu_memory_info() -> dict:
    if not torch.cuda.is_available():
        return {"error": "No GPU"}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "total_gb": torch.cuda.get_device_properties(0).total_mem / 1024**3,
    }


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
