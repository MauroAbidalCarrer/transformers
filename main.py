# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# OMP_NUM_THREADS=1 torchrun --standalone  --nproc_per_node=4 main.py
import os
from time import time
from contextlib import nullcontext

import torch
import tiktoken
from torch import nn
from torch import Tensor
import torch.distributed as dist
from torch.optim import Optimizer
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPT
from config import (
    GPTConfig,
    TorchConfig,
    TrainingConfig,
    ENCODING_NAME,
)
from data_utils import DataLoaderLite
from optimization_utils import mk_scheduler, mk_optimizer


def setup_torch(torch_config: TorchConfig):
    if torch_config.using_ddp:
        init_process_group(backend="nccl")
    torch.cuda.set_device(torch_config.device)
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

def master_print(*args, **kwargs):
    if torch_config.is_master_process:
        print(*args, **kwargs)

def training_step(
        model: nn.Module,
        data_loader: DataLoaderLite,
        torch_config: TorchConfig,
        optimizer: Optimizer,
        scheduler: LRScheduler,
    ) -> dict:
    model = model.train()
    optimizer.zero_grad()
    batch_loss = 0
    for micro_step in range(train_conf.grad_accum_step):
        x, y_true = data_loader.next_batch()
        x, y_true = x.to(torch_config.device), y_true.to(torch_config.device)
        # sample a batch of data
        y_true = y_true.reshape(train_conf.micro_batch_size * model_conf.attention_window_size)
        use_no_sync_ctx = (micro_step != (train_conf.grad_accum_step - 1)) and torch_config.using_ddp
        sync_ctx = model.no_sync if use_no_sync_ctx else nullcontext
        with sync_ctx(), torch.autocast(device_type=torch_config.device.type, dtype=torch.bfloat16):
            y_pred = model(x).reshape(train_conf.micro_batch_size * model_conf.attention_window_size, model_conf.vocab_size)
            micro_batch_loss = F.cross_entropy(y_pred, y_true) / train_conf.grad_accum_step
            micro_batch_loss.backward()
            # Use detach instead of item because we may need to call dist all reduce on it
        batch_loss += micro_batch_loss.detach() 
    if torch_config.using_ddp:
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
    loss_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    if torch_config.device.type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    return {
        "loss": batch_loss,
        "loss_norm": loss_norm,
    }


def eval_model(model: nn.Module, x: Tensor, y_true: Tensor) -> dict[str, float]:
    model = model.eval()
    with torch.no_grad(), torch.autocast(device_type=torch_config.device.type, dtype=torch.bfloat16):
        y_pred = model(x) # (batch size, window size, n embeding dims)
        y_pred = y_pred.reshape(train_conf.micro_batch_size * model_conf.attention_window_size, model_conf.vocab_size) # (batch size * window size, n embeding dims)
        y_true = y_true.reshape(train_conf.micro_batch_size * model_conf.attention_window_size)
        return {
            "loss": F.cross_entropy(y_pred, y_true).cpu().item(),
            "accuracy": (torch.argmax(y_pred, dim=1) == y_true).float().mean().cpu().item(),
        }

torch_config = TorchConfig()
setup_torch(torch_config)
model_conf = GPTConfig(vocab_size=50304)
train_conf = TrainingConfig(model_conf, torch_config.ddp_world_size)

data_loader = DataLoaderLite(
    train_conf.micro_batch_size,
    model_conf.attention_window_size,
    torch_config.ddp_rank,
    torch_config.ddp_world_size,
    "train",
    torch_config.is_master_process,
)
torch.set_float32_matmul_precision('high')
model = raw_model = torch.compile(GPT(model_conf).to(torch_config.device))
param_stats = model.get_params_stats()
master_print(f"number of parameters: {param_stats['count']:.2f}M, model memory usage: {param_stats['mem_usage']:.2f}MB")
if torch_config.using_ddp:
    model = DDP(model, device_ids=[torch_config.ddp_local_rank], find_unused_parameters=True) # Allows us to perform weight updates among the devices.

optimizer = mk_optimizer(model, train_conf)
scheduler = mk_scheduler(optimizer, train_conf)

last_step_time = time()
for step in range(train_conf.n_training_steps):    
    step_stats = training_step(model, data_loader, torch_config, optimizer, scheduler)
    # logging
    current_time = time()
    step_dt_ms = (current_time - last_step_time) * 1000
    tokens_per_sec = train_conf.tokens_per_step / (current_time - last_step_time)
    lr = scheduler.get_last_lr()
    master_print(f"step {step:4d} | batch loss {step_stats['loss']:5.3f} | batch loss norm {step_stats['loss_norm']:3.1f} | lr {lr[0]:10.7f} | dt {step_dt_ms:5.3f}ms | {tokens_per_sec:5.1f} tokens/s")
    last_step_time = current_time
    # checkpoints
    if torch_config.is_master_process:
        if step > 0 and (step % 100 == 0 or step == train_conf.n_training_steps - 1):
            os.makedirs("checkpoints", exist_ok=True)
            # optionally write model checkpoints
            checkpoint_path = os.path.join("checkpoints", f"model_{step:05d}.pt")
            checkpoint = {
                "model": raw_model.state_dict(),
                "config": raw_model.config,
                "step": step,
                "val_loss": batch_loss.item(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                # rng states (optional but useful if you want full reproducibility)
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            # you might also want to add optimizer.state_dict() and
            # rng seeds etc., if you wanted to more exactly resume training
            torch.save(checkpoint, checkpoint_path)

    
model_state = model.state_dict()
torch.save(model_state, "latest_model_params.pth")
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=torch_config.device)

tokenizer = tiktoken.get_encoding(ENCODING_NAME)
master_print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))

if torch_config.using_ddp:
    destroy_process_group()