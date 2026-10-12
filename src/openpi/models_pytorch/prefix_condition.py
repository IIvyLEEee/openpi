"""Transportable prefix K/V tensors. Consumers must not mutate these tensors."""

import dataclasses

import torch
from transformers.cache_utils import DynamicCache


@dataclasses.dataclass(frozen=True)
class PrefixCondition:
    # Per-layer K/V in [batch, kv_heads, prefix_length, head_dim], with RoPE already applied to K.
    key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    pad_mask: torch.Tensor

    def __post_init__(self):
        if self.pad_mask.ndim != 2 or self.pad_mask.dtype != torch.bool or not self.key_values:
            raise ValueError("A prefix requires a boolean [B, P] mask and nonempty layer K/V")
        for key, value in self.key_values:
            if key.ndim != 4 or key.shape != value.shape:
                raise ValueError("Prefix K/V must have matching [B, heads, P, D] shapes")
            if (key.shape[0], key.shape[2]) != tuple(self.pad_mask.shape):
                raise ValueError("Prefix K/V do not match the padding mask")
            if key.device != self.pad_mask.device or value.device != key.device:
                raise ValueError("Prefix tensors must be on the same device")

    def to(self, device: str | torch.device) -> "PrefixCondition":
        return PrefixCondition(
            tuple((key.to(device), value.to(device)) for key, value in self.key_values), self.pad_mask.to(device)
        )

    def as_cache(self) -> DynamicCache:
        # A new container per sampling call. Expert attention uses use_cache=False,
        # concatenating its own K/V without updating these shared prefix tensors.
        return DynamicCache.from_legacy_cache(self.key_values)
