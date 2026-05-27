from openpi.serving.server_metadata import build_policy_server_metadata


def test_build_policy_server_metadata_includes_policy_and_denoise_settings():
    metadata = build_policy_server_metadata(
        base_metadata={"task": "ur5e"},
        policy_config="pi0_umi_ur5e_pick_place_lora",
        policy_dir="checkpoints/pi0/29999",
        default_prompt="pick",
        num_steps=12,
        log_denoise_steps=True,
        lora_merge="off",
    )

    assert metadata["task"] == "ur5e"
    assert metadata["serve_policy"] == {
        "policy_config": "pi0_umi_ur5e_pick_place_lora",
        "policy_dir": "checkpoints/pi0/29999",
        "default_prompt": "pick",
        "num_steps": 12,
        "log_denoise_steps": True,
        "lora_merge": "off",
    }
