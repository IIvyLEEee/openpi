"""PyTorch backend with separate VLM and expert modules/devices."""

import jax
import numpy as np
import torch

from openpi.models import model as _model
from openpi.models_pytorch.prefix_condition import PrefixCondition
from openpi.transforms import compose


class DualSystemPolicy:
    def __init__(
        self, system2, system1, *, prefix_transforms, state_transforms, output_transforms, num_steps: int = 10
    ):
        if not system1.continuous_state_input:
            raise ValueError("The expert needs continuous_state_input for asynchronous proprioception")
        self.system2, self.system1 = system2, system1
        self._prefix_transform = compose(prefix_transforms)
        self._state_transform = compose(state_transforms)
        self._output_transform = compose(output_transforms)
        self._vlm_device = next(system2.parameters()).device
        self._expert_device = next(system1.parameters()).device
        self.max_prefix_age = system1.config.prefix_delay_max
        self._num_steps = num_steps
        self._source_prefix = self._expert_prefix = None

    def encode(self, observation: dict) -> PrefixCondition:
        inputs = self._prefix_transform(jax.tree.map(lambda x: x, observation))
        inputs = jax.tree.map(lambda x: torch.as_tensor(np.asarray(x), device=self._vlm_device)[None], inputs)
        condition = self.system2.encode_prefix(_model.Observation.from_dict(inputs))
        if self._vlm_device.type == "cuda":
            torch.cuda.synchronize(self._vlm_device)
        return condition

    def sample(self, observation: dict, condition: PrefixCondition, prefix_age: int) -> np.ndarray:
        # No image preprocessing/tokenization/VLM work in the fast path.
        inputs = self._state_transform({"state": np.asarray(observation["observation/state"]).copy()})
        state = torch.as_tensor(inputs["state"], device=self._expert_device)[None]
        # Transfer once per prefix update, not once per expert call.
        if condition is not self._source_prefix:
            self._expert_prefix = condition.to(self._expert_device)
            self._source_prefix = condition
        actions = self.system1.sample_actions_from_prefix(
            state,
            self._expert_prefix,
            prefix_age=torch.tensor([prefix_age], device=self._expert_device),
            num_steps=self._num_steps,
        )
        outputs = {"state": inputs["state"], "actions": actions[0].cpu().numpy()}
        return self._output_transform(outputs)["actions"]
