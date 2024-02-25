import pathlib
from typing import Optional

import torch
import torchvision.utils as vutils
import wandb
from torch import Tensor


def create_checkpoint(
        checkpoint_path: str,
        global_step: int,
        model, optimizer,
        grid: Optional[Tensor] = None,
        **kwargs
):

    path = pathlib.Path(checkpoint_path)

    if grid:
        vutils.save_image(grid, path / "image" / f"{global_step}.png")

    checkpoint = {
        'global_step': global_step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(checkpoint, str(path / 'checkpoint.pt.tar'))


def load_checkpoint(
        checkpoint_path: str,
        model_name: str,
        param_path: Optional[str] = "parameters/"
):
    pass
    # model.load_state_dict(checkpoint['model'])
