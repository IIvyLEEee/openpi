import copy
import dataclasses

import numpy as np
import pytest
import safetensors.torch
import torch
from torch import nn
from transformers import GemmaConfig
from transformers import PaliGemmaConfig
from transformers import SiglipVisionConfig

from openpi import transforms
from openpi.models import gemma
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import gemma_pytorch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies.dual_system_policy import DualSystemPolicy
from openpi.shared.normalize import NormStats
from openpi.training.pytorch_optimizer import parameter_groups


@pytest.fixture
def tiny_model_factory(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda variant: gemma.Config(
            width=64 if variant == "dummy" else 32,
            depth=2,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        ),
    )
    monkeypatch.setattr(
        gemma_pytorch,
        "CONFIG_MAPPING",
        {
            "gemma": GemmaConfig,
            "paligemma": lambda: PaliGemmaConfig(
                vision_config=SiglipVisionConfig(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    image_size=224,
                    patch_size=32,
                )
            ),
        },
    )

    def make(**overrides):
        config = Pi0Config(
            pi05=True,
            continuous_state_input=True,
            prefix_delay_max=5,
            paligemma_variant="dummy",
            action_expert_variant="gemma_300m",
            dtype="float32",
            pytorch_compile_mode=None,
            action_dim=4,
            action_horizon=3,
            max_token_len=8,
        )
        model = PI0Pytorch(dataclasses.replace(config, **overrides))
        # Genuine SigLIP/Gemma, with smaller vocabulary heads and vision projection.
        vlm = model.paligemma_with_expert.paligemma
        vlm.language_model.embed_tokens = nn.Embedding(128, 64)
        vlm.lm_head = nn.Linear(64, 128, bias=False)
        vlm.model.multi_modal_projector.linear = nn.Linear(32, 64)
        model.paligemma_with_expert.gemma_expert.lm_head = nn.Linear(32, 128, bias=False)
        return model.eval()

    return make


@pytest.fixture
def tiny_model(tiny_model_factory):
    return tiny_model_factory()


def observation():
    return _model.Observation(
        images={name: torch.rand(1, 3, 224, 224) * 2 - 1 for name in _model.IMAGE_KEYS},
        image_masks={name: torch.tensor([name != "right_wrist_0_rgb"]) for name in _model.IMAGE_KEYS},
        state=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        tokenized_prompt=torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]]),
        tokenized_prompt_mask=torch.tensor([[True] * 4 + [False] * 4]),
        prefix_age=torch.tensor([2]),
    )


def test_combined_and_cached_velocity_match(tiny_model):
    obs = observation()
    x = torch.randn(1, 3, 4)
    time = torch.tensor([0.4])
    images, masks, tokens, token_masks, state = tiny_model._preprocess_observation(obs, train=False)  # noqa: SLF001
    prefix, pm, pa = tiny_model.embed_prefix(images, masks, tokens, token_masks)
    suffix, sm, sa, cond = tiny_model.embed_suffix(state, x, time, obs.prefix_age)
    mask = make_att_2d_masks(torch.cat((pm, sm), 1), torch.cat((pa, sa), 1))
    positions = torch.cat((pm, sm), 1).cumsum(1) - 1
    with torch.no_grad():
        (_, out), _ = tiny_model.paligemma_with_expert(
            inputs_embeds=[prefix, suffix],
            attention_mask=tiny_model._prepare_attention_masks_4d(mask),  # noqa: SLF001
            position_ids=positions,
            use_cache=False,
            adarms_cond=[None, cond],
        )
        expected = tiny_model.action_out_proj(out[:, -3:])
        cached = tiny_model.encode_prefix(obs)
        actual = tiny_model.denoise_step(state, cached.pad_mask, cached.as_cache(), x, time, obs.prefix_age)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    # The added state block can read prefix but cannot read future action tokens.
    state_index = pm.shape[1]
    assert not mask[0, state_index, state_index + 1 :].any()
    assert mask[0, -1, state_index]


def test_split_sampling_cache_reuse_and_current_state(tiny_model):
    obs = observation()
    noise = torch.randn(1, 3, 4)
    expected = tiny_model.sample_actions("cpu", obs, noise=noise, num_steps=2)
    system2, system1 = tiny_model.split_for_inference("cpu", "cpu")
    assert not hasattr(system2.paligemma_with_expert, "gemma_expert")
    assert not hasattr(system1.paligemma_with_expert, "paligemma")
    condition = system2.encode_prefix(obs)
    before = tuple((k.clone(), v.clone()) for k, v in condition.key_values)
    for _ in range(2):
        actual = system1.sample_actions_from_prefix(
            obs.state,
            condition,
            noise=noise,
            num_steps=2,
            prefix_age=obs.prefix_age,
        )
        torch.testing.assert_close(actual, expected)
    changed = system1.sample_actions_from_prefix(
        obs.state + 1,
        condition,
        noise=noise,
        num_steps=2,
        prefix_age=obs.prefix_age,
    )
    assert not torch.allclose(changed, actual)
    for old, new in zip(before, condition.key_values, strict=True):
        for original, current in zip(old, new, strict=True):
            torch.testing.assert_close(original, current, atol=0, rtol=0)
    with pytest.raises(ValueError, match="age"):
        system1.sample_actions_from_prefix(obs.state, condition, prefix_age=torch.tensor([6]))


def test_joint_gradients_and_lr_groups(tiny_model):
    tiny_model.train()
    obs = observation()
    loss = tiny_model(obs, torch.randn(1, 3, 4), noise=torch.randn(1, 3, 4), time=torch.tensor([0.5])).mean()
    loss.backward()
    parameters = (
        tiny_model.state_proj.weight,
        tiny_model.delay_embedding.weight,
        tiny_model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.v_proj.weight,
        tiny_model.paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.patch_embedding.weight,
        tiny_model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    groups = parameter_groups(tiny_model, 1e-4, 0.1)
    assert [group["lr"] for group in groups] == pytest.approx([1e-5, 1e-4])
    assert sum(len(group["params"]) for group in groups) == len(list(tiny_model.parameters()))
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.AdamW(groups)
    optimizer.step()
    for original, parameter in zip(before, parameters, strict=True):
        assert not torch.equal(original, parameter)


def test_checkpoint_allowlist_and_strict_roundtrip(tiny_model, tmp_path):
    # Released PaliGemma weights have shared input/output embeddings.
    vlm = tiny_model.paligemma_with_expert.paligemma
    vlm.lm_head.weight = vlm.language_model.embed_tokens.weight
    base = copy.deepcopy(tiny_model)
    del base.state_proj
    del base.delay_embedding
    base.config = dataclasses.replace(base.config, continuous_state_input=False, prefix_delay_max=0)
    base.continuous_state_input = False
    base_path = tmp_path / "base.safetensors"
    safetensors.torch.save_model(base, base_path)
    with pytest.raises(RuntimeError, match="state_proj"):
        tiny_model.load_pretrained_weights(base_path)
    tiny_model.load_pretrained_weights(base_path, allow_new_parameters=True)
    full_path = tmp_path / "full.safetensors"
    safetensors.torch.save_model(tiny_model, full_path)
    tiny_model.load_pretrained_weights(full_path)
    broken = safetensors.torch.load_file(str(base_path))
    del broken["action_out_proj.bias"]
    bad_path = tmp_path / "bad.safetensors"
    safetensors.torch.save_file(broken, bad_path)
    with pytest.raises(RuntimeError, match=r"action_out_proj\.bias"):
        tiny_model.load_pretrained_weights(bad_path, allow_new_parameters=True)


def test_legacy_pi05_and_jax_guard(tiny_model):
    # Existing Pi05 keeps action-only suffix and no new checkpoint parameters.
    tiny_model.continuous_state_input = False
    del tiny_model.state_proj
    del tiny_model.delay_embedding
    obs = observation()
    suffix, _, _, _ = tiny_model.embed_suffix(obs.state, torch.zeros(1, 3, 4), torch.ones(1))
    assert suffix.shape == (1, 3, 32)
    assert not any(name.startswith(("state_proj", "delay_embedding")) for name in tiny_model.state_dict())
    with pytest.raises(NotImplementedError, match="PyTorch"):
        Pi0Config(pi05=True, continuous_state_input=True).create(None)


def test_bfloat16_split_sampling(tiny_model):
    tiny_model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    obs = observation()
    noise = torch.randn(1, 3, 4)
    expected = tiny_model.sample_actions("cpu", obs, noise=noise, num_steps=2)
    system2, system1 = tiny_model.split_for_inference("cpu", "cpu")
    actual = system1.sample_actions_from_prefix(
        obs.state, system2.encode_prefix(obs), noise=noise, num_steps=2, prefix_age=obs.prefix_age
    )
    torch.testing.assert_close(actual, expected)


def test_state_only_policy_normalization_and_output(tiny_model):
    obs = observation()
    condition = tiny_model.encode_prefix(obs)
    system2, system1 = tiny_model.split_for_inference("cpu", "cpu")
    stats = NormStats(mean=np.ones(4, dtype=np.float32), std=np.full(4, 2, dtype=np.float32))
    backend = DualSystemPolicy(
        system2,
        system1,
        prefix_transforms=[],
        state_transforms=[transforms.Normalize({"state": stats}), transforms.PadStatesAndActions(4)],
        output_transforms=[transforms.Unnormalize({"actions": stats})],
        num_steps=2,
    )
    raw = {"observation/state": np.asarray([1, 2, 3, 4], dtype=np.float32)}
    torch.manual_seed(11)
    actual = backend.sample(raw, condition, 2)  # No images or prompt present.
    normalized = torch.as_tensor((raw["observation/state"] - 1) / (2 + 1e-6))[None]
    torch.manual_seed(11)
    expected = system1.sample_actions_from_prefix(normalized, condition, prefix_age=torch.tensor([2]), num_steps=2)
    np.testing.assert_allclose(actual, expected[0].numpy() * (2 + 1e-6) + 1, rtol=1e-5)
    np.testing.assert_array_equal(raw["observation/state"], [1, 2, 3, 4])


@pytest.mark.parametrize("pi05", [False, True])
def test_legacy_models_sample_and_load_without_new_parameters(tiny_model_factory, pi05, tmp_path):
    model = tiny_model_factory(pi05=pi05, continuous_state_input=False, prefix_delay_max=0)
    obs = dataclasses.replace(observation(), prefix_age=None)
    suffix, _, _, _ = model.embed_suffix(obs.state, torch.zeros(1, 3, 4), torch.ones(1))
    assert suffix.shape[1] == 3 + int(not pi05)
    result = model.sample_actions("cpu", obs, num_steps=2)
    assert result.shape == (1, 3, 4)
    assert torch.isfinite(result).all()
    path = tmp_path / "legacy.safetensors"
    safetensors.torch.save_model(model, path)
    model.load_pretrained_weights(path)
    assert not hasattr(model, "delay_embedding")
