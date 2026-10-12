"""Sample old prefix observations while keeping proprioception and action targets current."""

import numpy as np


class DelayedObservationDataset:
    def __init__(self, dataset, transform, max_delay: int):
        if max_delay < 0:
            raise ValueError("max_delay must be nonnegative")
        self._dataset = dataset
        self._transform = transform
        self._max_delay = max_delay

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        index = int(index)
        current = self._dataset[index]
        frame = int(current["frame_index"])
        delay = min(int(np.random.randint(self._max_delay + 1)), frame)
        old = self._dataset[index - delay] if delay else current
        if int(old["episode_index"]) != int(current["episode_index"]):
            raise ValueError("Historical observation crossed an episode boundary")
        if int(old["frame_index"]) != frame - delay:
            raise ValueError("Delayed sampling requires contiguous LeRobot frame indices")
        if "task_index" in current and int(old["task_index"]) != int(current["task_index"]):
            # Do not condition on an instruction from before a task switch.
            old, delay = current, 0
        # Transform independently: in particular, the tokenizer reads old state,
        # while normalization of current state/actions uses the same norm stats.
        fast = self._transform(dict(current))
        slow = self._transform(dict(old)) if delay else fast
        result = dict(fast)
        for key in ("image", "image_mask", "tokenized_prompt", "tokenized_prompt_mask"):
            result[key] = slow[key]
        result["prefix_age"] = np.asarray(delay, dtype=np.int32)
        return result
