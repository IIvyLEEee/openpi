import dataclasses
import enum
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LoRAMergeMode(enum.Enum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"


@dataclasses.dataclass(frozen=True)
class LoRAMergeSummary:
    mode: LoRAMergeMode
    merged_count: int
    stripped_config: bool

    @property
    def merged(self) -> bool:
        return self.merged_count > 0


def normalize_lora_merge_mode(mode: LoRAMergeMode | str) -> LoRAMergeMode:
    if isinstance(mode, LoRAMergeMode):
        return mode
    return LoRAMergeMode(mode)


def has_lora_params(params: dict[str, Any]) -> bool:
    flat_params = _flatten_dict(params)
    return any("/lora_" in f"/{key}" or key.endswith(("_lora_a", "_lora_b")) for key in flat_params)


def model_config_uses_lora(model_config: Any) -> bool:
    return any(
        isinstance(getattr(model_config, attr, None), str) and "lora" in getattr(model_config, attr)
        for attr in ("paligemma_variant", "action_expert_variant")
    )


def strip_lora_model_config(model_config: Any) -> tuple[Any, bool]:
    updates = {}
    for attr in ("paligemma_variant", "action_expert_variant"):
        value = getattr(model_config, attr, None)
        if isinstance(value, str) and value.endswith("_lora"):
            updates[attr] = value.removesuffix("_lora")
    if not updates:
        return model_config, False
    return dataclasses.replace(model_config, **updates), True


def merge_lora_params(params: dict[str, Any]) -> tuple[dict[str, Any], LoRAMergeSummary]:
    flat_params = _flatten_dict(params)
    merged = dict(flat_params)
    merged_count = 0

    for key, value in list(flat_params.items()):
        if key == "lora_a":
            base_key = "w"
            b_key = "lora_b"
        elif key.endswith("/lora_a"):
            base_key = key.removesuffix("/lora_a") + "/w"
            b_key = key.removesuffix("/lora_a") + "/lora_b"
        elif key.endswith("_lora_a"):
            base_key = key.removesuffix("_lora_a")
            b_key = key.removesuffix("_lora_a") + "_lora_b"
        else:
            continue

        if base_key not in flat_params or b_key not in flat_params:
            continue

        delta = _lora_delta(value, flat_params[b_key], dtype=flat_params[base_key].dtype)
        merged[base_key] = flat_params[base_key] + delta
        merged.pop(key, None)
        merged.pop(b_key, None)
        merged_count += 1

    summary = LoRAMergeSummary(mode=LoRAMergeMode.ON, merged_count=merged_count, stripped_config=False)
    return _unflatten_dict(merged), summary


def maybe_merge_lora_params(
    params: dict[str, Any],
    model_config: Any,
    *,
    mode: LoRAMergeMode | str,
) -> tuple[dict[str, Any], Any, LoRAMergeSummary]:
    merge_mode = normalize_lora_merge_mode(mode)
    if merge_mode is LoRAMergeMode.OFF:
        return params, model_config, LoRAMergeSummary(mode=merge_mode, merged_count=0, stripped_config=False)

    if not model_config_uses_lora(model_config) or not has_lora_params(params):
        if merge_mode is LoRAMergeMode.ON:
            raise ValueError("LoRA merge was requested, but the model config or checkpoint params do not contain LoRA.")
        return params, model_config, LoRAMergeSummary(mode=merge_mode, merged_count=0, stripped_config=False)

    merged_params, summary = merge_lora_params(params)
    if summary.merged_count == 0:
        if merge_mode is LoRAMergeMode.ON:
            raise ValueError("LoRA merge was requested, but no mergeable LoRA parameter pairs were found.")
        return params, model_config, LoRAMergeSummary(mode=merge_mode, merged_count=0, stripped_config=False)

    merged_config, stripped_config = strip_lora_model_config(model_config)
    if not stripped_config:
        if merge_mode is LoRAMergeMode.ON:
            raise ValueError(
                "LoRA params were merged, but the model config could not be converted to a non-LoRA variant."
            )
        return params, model_config, LoRAMergeSummary(mode=merge_mode, merged_count=0, stripped_config=False)

    logger.info("Merged %d LoRA parameter pairs into base weights.", summary.merged_count)
    return (
        merged_params,
        merged_config,
        LoRAMergeSummary(mode=merge_mode, merged_count=summary.merged_count, stripped_config=stripped_config),
    )


def _lora_delta(lora_a, lora_b, *, dtype):
    xp = _array_namespace(lora_a, lora_b)
    return xp.einsum("...ir,...ro->...io", lora_a, lora_b).astype(dtype)


def _array_namespace(*arrays):
    if any(type(array).__module__.startswith(("jax", "jaxlib")) for array in arrays):
        import jax.numpy as jnp  # noqa: PLC0415

        return jnp
    return np


def _flatten_dict(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, path))
        else:
            flat[path] = value
    return flat


def _unflatten_dict(flat: dict[str, Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path, value in flat.items():
        cursor = root
        parts = path.split("/")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root
