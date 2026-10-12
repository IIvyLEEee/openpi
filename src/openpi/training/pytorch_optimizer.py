"""PyTorch parameter groups for joint VLM/expert fine-tuning."""


def parameter_groups(model, lr: float, vlm_lr_scale: float):
    if vlm_lr_scale <= 0:
        raise ValueError("pytorch_vlm_lr_scale must be positive")
    if vlm_lr_scale == 1:
        return [{"params": list(model.parameters()), "lr": lr, "lr_scale": 1.0, "name": "all"}]
    vlm, expert = [], []
    for name, parameter in model.named_parameters():
        group = vlm if name.startswith("paligemma_with_expert.paligemma.") else expert
        group.append(parameter)
    return [
        {"params": vlm, "lr": lr * vlm_lr_scale, "lr_scale": vlm_lr_scale, "name": "vlm"},
        {"params": expert, "lr": lr, "lr_scale": 1.0, "name": "expert"},
    ]
