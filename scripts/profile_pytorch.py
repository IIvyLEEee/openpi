"""Run the real trainer for memory calibration without writing large checkpoints.

Accepts the same CLI as train_pytorch.py. Use a small num_train_steps and a distinct
experiment name; optimizer updates and W&B logging remain enabled.
"""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts import train_pytorch


def skip_checkpoint(*_args, **_kwargs):
    pass


if __name__ == "__main__":
    train_pytorch.save_checkpoint = skip_checkpoint
    try:
        train_pytorch.main()
    finally:
        if torch.cuda.is_available():
            print(
                "PROFILE_GPU="
                + json.dumps(
                    {
                        "rank": int(os.environ.get("LOCAL_RANK", "0")),
                        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                    }
                ),
                flush=True,
            )
