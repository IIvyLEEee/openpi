from typing import Any


def build_policy_server_metadata(
    *,
    base_metadata: dict[str, Any] | None,
    policy_config: str | None,
    policy_dir: str | None,
    default_prompt: str | None,
    num_steps: int | None,
    log_denoise_steps: bool,
    lora_merge: str,
) -> dict[str, Any]:
    metadata = dict(base_metadata or {})
    metadata["serve_policy"] = {
        "policy_config": policy_config,
        "policy_dir": policy_dir,
        "default_prompt": default_prompt,
        "num_steps": num_steps,
        "log_denoise_steps": bool(log_denoise_steps),
        "lora_merge": lora_merge,
    }
    return metadata
