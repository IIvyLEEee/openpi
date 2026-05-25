import dataclasses

import numpy as np
import pytest

from openpi.models import lora_merge


@dataclasses.dataclass(frozen=True)
class DummyModelConfig:
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"


def test_merge_lora_params_folds_einsum_lora_pair_and_removes_adapter_weights():
    params = {
        "attn": {
            "qkv_einsum": {
                "w": np.ones((3, 2, 4, 5), dtype=np.float32),
                "lora_a": np.full((3, 2, 4, 2), 2.0, dtype=np.float32),
                "lora_b": np.full((3, 2, 2, 5), 3.0, dtype=np.float32),
            }
        }
    }

    merged, summary = lora_merge.merge_lora_params(params)

    assert summary.merged_count == 1
    expected_delta = np.einsum(
        "...ir,...ro->...io",
        params["attn"]["qkv_einsum"]["lora_a"],
        params["attn"]["qkv_einsum"]["lora_b"],
    )
    np.testing.assert_allclose(merged["attn"]["qkv_einsum"]["w"], params["attn"]["qkv_einsum"]["w"] + expected_delta)
    assert "lora_a" not in merged["attn"]["qkv_einsum"]
    assert "lora_b" not in merged["attn"]["qkv_einsum"]


def test_merge_lora_params_folds_feedforward_lora_pairs():
    params = {
        "mlp": {
            "gating_einsum": np.ones((2, 4, 6), dtype=np.float32),
            "gating_einsum_lora_a": np.full((2, 4, 2), 2.0, dtype=np.float32),
            "gating_einsum_lora_b": np.full((2, 2, 6), 3.0, dtype=np.float32),
            "linear": np.ones((6, 4), dtype=np.float32),
            "linear_lora_a": np.full((6, 2), 4.0, dtype=np.float32),
            "linear_lora_b": np.full((2, 4), 5.0, dtype=np.float32),
        }
    }

    merged, summary = lora_merge.merge_lora_params(params)

    assert summary.merged_count == 2
    np.testing.assert_allclose(
        merged["mlp"]["gating_einsum"],
        params["mlp"]["gating_einsum"]
        + np.einsum(
            "...ir,...ro->...io",
            params["mlp"]["gating_einsum_lora_a"],
            params["mlp"]["gating_einsum_lora_b"],
        ),
    )
    np.testing.assert_allclose(
        merged["mlp"]["linear"],
        params["mlp"]["linear"]
        + np.einsum("...ir,...ro->...io", params["mlp"]["linear_lora_a"], params["mlp"]["linear_lora_b"]),
    )
    assert "gating_einsum_lora_a" not in merged["mlp"]
    assert "linear_lora_b" not in merged["mlp"]


def test_maybe_merge_lora_params_strips_lora_model_variants():
    params = {
        "module": {
            "w": np.ones((2, 2), dtype=np.float32),
            "lora_a": np.ones((2, 1), dtype=np.float32),
            "lora_b": np.ones((1, 2), dtype=np.float32),
        }
    }

    merged_params, merged_config, summary = lora_merge.maybe_merge_lora_params(
        params, DummyModelConfig(), mode=lora_merge.LoRAMergeMode.ON
    )

    assert summary.merged_count == 1
    assert summary.stripped_config is True
    assert merged_config.paligemma_variant == "gemma_2b"
    assert merged_config.action_expert_variant == "gemma_300m"
    assert "lora_a" not in merged_params["module"]


def test_lora_merge_on_requires_lora_params_and_config():
    with pytest.raises(ValueError, match="do not contain LoRA"):
        lora_merge.maybe_merge_lora_params(
            {"module": {"w": np.ones((2, 2), dtype=np.float32)}},
            DummyModelConfig(),
            mode=lora_merge.LoRAMergeMode.ON,
        )
