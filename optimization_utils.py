import math
import warnings
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.optim import Optimizer, AdamW, Adam
from torch.optim.lr_scheduler import _LRScheduler
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

# @dataclass
# class TrainingConfig:
#     n_training_steps: int = field(default=19000)
#     max_lr: float = field(default=6e-4)
#     n_warmup_steps: int = field(default=715)
#     min_lr: float = field(default=1e-6)


# def mk_scheduler(optimizer: Optimizer, train_conf: TrainingConfig):
#     n_cosine_steps = train_conf.n_training_steps - train_conf.n_warmup_steps
#     warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
#         optimizer,
#         start_factor=1e-8,   # ~0
#         end_factor=1.0,
#         total_iters=train_conf.n_training_steps,
#     )

#     cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=n_cosine_steps,
#         eta_min=train_conf.min_lr,
#         # last_epoch=None,
#     )

#     scheduler = torch.optim.lr_scheduler.SequentialLR(
#         optimizer,
#         schedulers=[warmup_scheduler, cosine_scheduler],
#         milestones=[train_conf.n_warmup_steps]  # when to switch from warmup → cosine
#     )

#     return scheduler

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
    
class WarmupCosineScheduler(_LRScheduler):
    """
    Scheduler implementing the exact get_lr(it) logic:
      1) linear warmup for warmup_steps (it from 0..warmup_steps-1): lr = max_lr * (it+1)/warmup_steps
      2) cosine decay from max_lr -> min_lr for it in [warmup_steps .. max_steps]
         where max_steps = n_training_steps - 1
      3) for it > max_steps return min_lr
    """

    def __init__(self, optimizer, n_training_steps: int, n_warmup_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        if n_training_steps <= 0:
            raise ValueError("n_training_steps must be > 0")
        if not (0 <= n_warmup_steps < n_training_steps):
            raise ValueError("0 <= n_warmup_steps < n_training_steps required")

        self.n_training_steps = n_training_steps
        self.warmup_steps = n_warmup_steps
        # max_steps is the last iteration index where scheduler still computes decay (inclusive)
        self.max_steps = n_training_steps - 1
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        it = self.last_epoch  # this will be -1 initially before any .step()
        # If scheduler called for initial lr (last_epoch == -1), return base_lrs unchanged
        if it < 0:
            return [base_lr for base_lr in self.base_lrs]

        lrs = []
        for base_lr in self.base_lrs:
            # Treat base_lr as the "max_lr" for this param group
            max_lr = base_lr
            if it < self.warmup_steps:
                # linear warmup (it in [0 .. warmup_steps-1])
                lr = max_lr * float(it + 1) / float(self.warmup_steps)
            elif it > self.max_steps:
                # after the schedule finishes
                lr = float(self.min_lr)
            else:
                # cosine decay between warmup_steps .. max_steps (inclusive)
                denom = float(self.max_steps - self.warmup_steps)
                # denom should be >= 1 because warmup_steps < n_training_steps
                decay_ratio = float(it - self.warmup_steps) / denom
                # numeric safety clamp
                decay_ratio = min(max(decay_ratio, 0.0), 1.0)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # goes from 1 -> 0
                lr = float(self.min_lr) + coeff * (max_lr - float(self.min_lr))
            lrs.append(lr)
        return lrs


if __name__ == "__main__":
    # hyperparameters (as requested)
    train_conf = TrainingConfig()

    # Dummy model + optimizer: set optimizer lr to train_conf.max_lr so scheduler uses that as "max_lr"
    model = torch.nn.Linear(10, 10)
    optimizer = Adam(model.parameters(), lr=train_conf.max_lr)

    # create scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        n_training_steps=train_conf.n_training_steps,
        n_warmup_steps=train_conf.n_warmup_steps,
        min_lr=train_conf.min_lr,
    )

    # collect learning rates for plotting
    lrs = []
    steps = list(range(train_conf.n_training_steps))

    for _ in steps:
        # Step scheduler first so last_epoch becomes 0..n_training_steps-1
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # Quick sanity prints (first, warmup-end, last)
    print(f"step 0 lr: {lrs[0]:.6e}")
    warm_end_idx = train_conf.n_warmup_steps - 1
    if 0 <= warm_end_idx < len(lrs):
        print(f"warmup end (step {warm_end_idx}) lr: {lrs[warm_end_idx]:.6e}")
    print(f"last step (step {train_conf.n_training_steps-1}) lr: {lrs[-1]:.6e}")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(steps, lrs, label="LR")
    plt.axvline(train_conf.n_warmup_steps - 0.5, color="gray", linestyle="--", linewidth=0.6, label="warmup end")
    plt.xlabel("Step")
    plt.ylabel("Learning rate")
    plt.title("Warmup + Cosine LR schedule (corrected)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
