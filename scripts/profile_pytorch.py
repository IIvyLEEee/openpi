"""Run the real trainer for memory calibration without writing large checkpoints.

Accepts the same CLI as train_pytorch.py. Use a small num_train_steps and a distinct
experiment name; optimizer updates and W&B logging remain enabled.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import train_pytorch


def skip_checkpoint(*_args, **_kwargs):
    pass


if __name__ == "__main__":
    train_pytorch.save_checkpoint = skip_checkpoint
    train_pytorch.main()
