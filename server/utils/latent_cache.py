"""
GPU-resident latent cache for reference image embeddings.
Stores reference latents in VRAM to avoid re-encoding every frame.
"""

import torch
from typing import Optional


class LatentCache:
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self._cache: dict[str, torch.Tensor] = {}

    def store(self, key: str, tensor: torch.Tensor):
        self._cache[key] = tensor.to(device=self.device, dtype=self.dtype)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self._cache.get(key)

    def has(self, key: str) -> bool:
        return key in self._cache

    def clear(self):
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def memory_usage_mb(self) -> float:
        total = 0
        for tensor in self._cache.values():
            total += tensor.element_size() * tensor.nelement()
        return total / 1024 / 1024
