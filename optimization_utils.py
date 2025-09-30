import warnings
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.optim import Optimizer, AdamW, Adam
import matplotlib.pyplot as plt

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

@dataclass
class TrainingConfig:
    n_training_steps: int = field(default=19000)
    max_lr: float = field(default=6e-4)
    n_warmup_steps: int = field(default=715)
    min_lr: float = field(default=1e-6)


def mk_scheduler(optimizer: Optimizer, train_conf: TrainingConfig):
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

def mk_optimizer(model: nn.Module, train_conf: TrainingConfig) -> list[dict]:
    learnable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    decay_params = [param for param in learnable_params if param.dim() >= 2]
    nodecay_params = [param for param in learnable_params if param.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": train_conf.weight_decay},
        {"params": nodecay_params, "weight_decay": 0},
    ]
    
    return AdamW(
        optim_groups,
        train_conf.max_lr,
        train_conf.betas,
        train_conf.eps,
        fused=True,
    )
    
if __name__ == "__main__":
    # Setup
    train_conf = TrainingConfig()
    model = torch.nn.Linear(10, 10)  # dummy model
    optimizer = Adam(model.parameters(), lr=train_conf.max_lr)

    scheduler = mk_scheduler(optimizer, train_conf)

    # Track LR over training steps
    lrs = []
    for step in range(train_conf.n_training_steps):
        optimizer.step()  # fake update
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # Plot
    plt.plot(range(train_conf.n_training_steps), lrs)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Warmup + Cosine Annealing Schedule")
    plt.grid(True)
    plt.show()
