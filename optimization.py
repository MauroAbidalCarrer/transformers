import warnings

import torch

from config import TrainingConfig

warnings.filterwarnings(
    "ignore",
    message=(
        r"The epoch parameter in `scheduler\.step\(\)` was not necessary "
        r"and is being deprecated where possible.*"
    ),
    category=UserWarning,
    module="torch.optim.lr_scheduler"
)


def mk_scheduler(optimizer: torch.optim.Optimizer, train_conf: TrainingConfig):
    n_cosine_steps = train_conf.n_training_steps - train_conf.n_warmup_steps
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,   # ~0
        end_factor=1.0,
        total_iters=train_conf.n_training_steps,
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_cosine_steps,
        eta_min=train_conf.min_lr,
        # last_epoch=None,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[train_conf.n_warmup_steps]  # when to switch from warmup → cosine
    )

    return scheduler