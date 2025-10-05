# OMP_NUM_THREADS=1 torchrun --standalone  --nproc_per_node=4 main.py
import os
import math
from time import time
from functools import partial
from contextlib import nullcontext

import wandb
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

from config import (
    GPTConfig,
    TorchConfig,
    ENCODING_NAME,
    TrainingConfig,
)
from model import GPT
from data_utils import DataLoaderLite
from hella_swag import iterate_examples, render_example
from optimization_utils import mk_optimizer


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

@torch.no_grad()
def validation_step(model: nn.Module, data_loader: DataLoaderLite, torch_conf: TorchConfig, train_conf: TrainingConfig) -> dict:
    model.eval()
    data_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y_true = data_loader.next_batch()
            x, y_true = x.to(torch_conf.device), y_true.to(torch_conf.device)
            y_true = y_true.reshape(train_conf.micro_batch_size * model_conf.attention_window_size)
            with torch.autocast(device_type=torch_conf.device_type, dtype=torch.bfloat16):
                y_pred = model(x).reshape(train_conf.micro_batch_size * model_conf.attention_window_size, model_conf.model_vocab_size)
            micro_batch_loss = F.cross_entropy(y_pred, y_true) / val_loss_steps
            val_loss_accum += micro_batch_loss.detach()
    if torch_conf.using_ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    master_print(f"validation loss: {val_loss_accum.item():.4f}")
    if torch_conf.is_master_process and train_conf.use_wandb:
        wandb.log({"val/loss": val_loss_accum, "step": train_conf.step})

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

def hella_swag_eval(model: nn.Module, torch_config: TorchConfig, train_conf: TrainingConfig):
    eval_start_time = time()
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % torch_config.ddp_world_size != torch_config.ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(torch_config.device)
        mask = mask.to(torch_config.device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=torch_config.device_type, dtype=torch.bfloat16):
                logits = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    # reduce the stats across all processes
    if torch_config.using_ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=torch_config.device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=torch_config.device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    time_to_eval_ms = (time() - eval_start_time) * 1000
    master_print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}, time to eval: {time_to_eval_ms:4.0f}ms")
    if torch_config.is_master_process and train_conf.use_wandb:
        wandb.log({"eval/hellaswag_acc": acc_norm, "step": train_conf.step})

# --- enable bf16 only if device supports it (Ampere or newer) ---
def _can_use_bfloat16(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        # get capability for current device index
        major, minor = torch.cuda.get_device_capability(device.index)
    except Exception:
        major, minor = torch.cuda.get_device_capability()
    # Ampere (sm_80) and newer support efficient bfloat16
    return major >= 8

global printed_dtypes
printed_dtypes = False
def training_step(
        model: nn.Module,
        data_loader: DataLoaderLite,
        torch_config: TorchConfig,
        optimizer: Optimizer,
        train_conf: TrainingConfig,
    ) -> dict:
    model.train()
    optimizer.zero_grad()
    batch_loss = 0.0
    global printed_dtypes
    for micro_step in range(train_conf.grad_accum_step):
        x, y_true = data_loader.next_batch()
        x, y_true = x.to(torch_config.device), y_true.to(torch_config.device)
        y_true = y_true.reshape(train_conf.micro_batch_size * model_conf.attention_window_size) #.long()

        use_no_sync_ctx = (micro_step != (train_conf.grad_accum_step - 1)) and torch_config.using_ddp
        sync_ctx = model.no_sync if use_no_sync_ctx else nullcontext

        with sync_ctx():
            with autocast_ctx():
                y_pred = model(x)
            y_pred = y_pred.reshape(train_conf.micro_batch_size * model_conf.attention_window_size,
                                    model_conf.model_vocab_size).float()
            micro_batch_loss = F.cross_entropy(y_pred, y_true) / train_conf.grad_accum_step
            micro_batch_loss.backward()

        batch_loss += micro_batch_loss.detach()

    # Average the loss across all ranks
    if torch_config.using_ddp:
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)

    # --- GRADIENT AND PARAMETER CHECKS ---
    if torch_config.is_master_process and not printed_dtypes:

        grad_dtypes = {}
        non_finite_count = 0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            grad_dtypes.setdefault(p.grad.dtype, 0)
            grad_dtypes[p.grad.dtype] += 1

            if not torch.isfinite(p.grad).all():
                non_finite_count += 1
                print(f"rank{torch_config.ddp_rank}: [WARN] Non-finite gradients in '{name}'")

            print(f"rank{torch_config.ddp_rank}: Gradient for '{name}' has dtype {p.grad.dtype}")
            print(f"rank{torch_config.ddp_rank}: Parameter '{name}' has dtype {p.dtype}")

        # --- check optimizer state dtypes (first param group only) ---
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    for key, val in optimizer.state[p].items():
                        if torch.is_tensor(val) and val.dtype != torch.float32:
                            print(f"rank{torch_config.ddp_rank}: [WARN] Optimizer state '{key}' for param '{p.shape}' has dtype {val.dtype}")
        printed_dtypes = True

    # Clip gradients and compute their norm
    loss_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(train_conf.step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Abort step if grads exploded
    if not math.isfinite(loss_norm):
        master_print(f"[WARN] Non-finite grad norm ({loss_norm}); skipping optimizer.step() at step {train_conf.step}")
        optimizer.zero_grad()
        return {"loss": batch_loss, "loss_norm": loss_norm}

    optimizer.step()

    if torch_config.device.type == "cuda":
        torch.cuda.synchronize()

    return {
        "loss": batch_loss,
        "loss_norm": loss_norm,
    }

def save_checkpoint(raw_model: nn.Module, optimizer: Optimizer, train_conf: TrainingConfig, last_train_step_stats: dict):
    os.makedirs("checkpoints", exist_ok=True)
    # optionally write model checkpoints
    checkpoint_path = os.path.join("checkpoints", f"model_{train_conf.step:05d}.pt")
    checkpoint = {
        "model": raw_model.state_dict(),
        "model_config": raw_model.config,
        "step": train_conf.step,
        "last_train_loss": last_train_step_stats['loss'].item(),
        "optimizer": optimizer.state_dict(),
        # "scheduler": scheduler.state_dict(),
        # rng states (optional but useful if you want full reproducibility)
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    # you might also want to add optimizer.state_dict() and
    # rng seeds etc., if you wanted to more exactly resume training
    torch.save(checkpoint, checkpoint_path)
    master_print("Saved step", train_conf.step, "checkpoint.")

def generate_text(raw_model: GPT, tokenizer, torch_conf: TorchConfig, train_conf: TrainingConfig) -> Tensor:
    raw_model.eval()
    num_return_sequences = 4
    max_length = 32
    tokens = tokenizer.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(torch_conf.device)
    sample_rng = torch.Generator(device=torch_conf.device)
    sample_rng.manual_seed(42 + torch_conf.ddp_rank)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=torch_conf.device_type, dtype=torch.bfloat16):
                logits = raw_model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :raw_model.config.tokenizer_vocab_size] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    generations = []
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)
        generations.append(decoded)
        print(f"rank {torch_conf.ddp_rank} sample {i}: {decoded}")
    if train_conf.use_wandb and torch_conf.is_master_process:
        table = wandb.Table(columns=["step", "sample_id", "text"])
        for i, gen in enumerate(generations):
            table.add_data(train_conf.step, i, gen)
        wandb.log({"generated_text": table}, step=train_conf.step)

# setup
torch_config = TorchConfig()
setup_torch(torch_config)
model_conf = GPTConfig(
    model_vocab_size=50304,
    tokenizer_vocab_size=tiktoken.get_encoding(ENCODING_NAME).max_token_value + 1
)
train_conf = TrainingConfig(model_conf, torch_config.ddp_world_size)
master_print("gradient accumulation steps:", train_conf.grad_accum_step)
USE_BFLOAT16 = _can_use_bfloat16(torch_config.device)
master_print("USE_BFLOAT16 autocast:", USE_BFLOAT16)
# autocast context factory (nullcontext if not available)
autocast_ctx = partial(torch.autocast, device_type=torch_config.device_type, dtype=torch.bfloat16) if USE_BFLOAT16 else nullcontext

if train_conf.starting_checkpoint is not None:
    master_print("starting checkpoint path = ", train_conf.checkpoint_path)
    model_conf = train_conf.model_config   
mk_data_loader = partial(
    DataLoaderLite,
    train_conf.micro_batch_size,
    model_conf.attention_window_size,
    torch_config.ddp_rank,
    torch_config.ddp_world_size,
    master_process=torch_config.is_master_process,
)
train_data_loader = mk_data_loader("train")
val_data_loader = mk_data_loader("val")
torch.set_float32_matmul_precision('high')
model = raw_model = GPT(model_conf).to(torch_config.device)
if train_conf.starting_checkpoint is not None:
    master_print(
        "starting from checkpoint at step",
        train_conf.starting_step,
        "and train loss",
        train_conf.starting_checkpoint["last_train_loss"]
    )
    model.load_state_dict(train_conf.starting_checkpoint["model"])
# model = raw_model = torch.compile(GPT(model_conf).to(torch_config.device))
param_stats = model.get_params_stats()
master_print(f"number of parameters: {param_stats['count']:.2f}M, model memory usage: {param_stats['mem_usage']:.2f}MB")

if torch_config.using_ddp:
    model = DDP(model, device_ids=[torch_config.ddp_local_rank], find_unused_parameters=True) # Allows us to perform weight updates among the devices.
optimizer = mk_optimizer(model, train_conf)

def _get_lr(it: int):
    # 1) linear warmup for warmup_iters steps
    if it < train_conf.n_warmup_steps:
        return train_conf.max_lr * (it+1) / train_conf.n_warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > train_conf.n_training_steps:
        return train_conf.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - train_conf.n_warmup_steps) / (train_conf.n_training_steps - train_conf.n_warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return train_conf.min_lr + coeff * (train_conf.max_lr - train_conf.min_lr)


def get_lr(it: int) -> float:
    return _get_lr(it) / 1.5

# scheduler = mk_scheduler(optimizer, train_conf)
if train_conf.starting_checkpoint:
    optimizer.load_state_dict(train_conf.starting_checkpoint["optimizer"])
    # scheduler.load_state_dict(train_conf.starting_checkpoint["scheduler"])
    torch.set_rng_state(train_conf.starting_checkpoint["rng_state"])
    # torch.cuda.set_rng_state_all(train_conf.starting_checkpoint["cuda_rng_state"])

tokenizer = tiktoken.get_encoding(ENCODING_NAME)
if torch_config.is_master_process and train_conf.use_wandb:
    wandb.init(
        project="gpt-training",
        config={
            "model": model_conf.__dict__,
            "training": train_conf.__dict__,
            "torch": torch_config.__dict__,
        }
    )
# Training loop
last_step_time = time()
for _step in range(train_conf.starting_step, train_conf.n_training_steps):    
    train_conf.step = _step
    is_last_step = train_conf.step == train_conf.n_training_steps - 1
    # Generate text
    # if is_last_step or train_conf.step % train_conf.text_gen_freq == 0:
    #     generate_text(raw_model, tokenizer, torch_config, train_conf)
    # validation loss
    # if is_last_step or train_conf.step % train_conf.validation_freq == 0:
    #     validation_step(model, val_data_loader, torch_config, train_conf)
    # hella swag eval
    # if is_last_step or train_conf.step % train_conf.hella_swag_eval_freq == 0:
    #     hella_swag_eval(model, torch_config, train_conf)
    # Training step
    step_stats = training_step(model, train_data_loader, torch_config, optimizer, train_conf)
    # step_stats = training_step(model, train_data_loader, torch_config, optimizer, scheduler)
    # logging
    current_time = time()
    step_dt_ms = (current_time - last_step_time) * 1000
    tokens_per_sec = train_conf.tokens_per_step / (current_time - last_step_time)
    # lr = scheduler.get_last_lr()
    lr = get_lr(train_conf.step)
    master_print(f"step {train_conf.step:4d} | batch loss {step_stats['loss']:5.3f} | batch loss norm {step_stats['loss_norm']:3.1f} | lr {lr:10.7f} | dt {step_dt_ms:5.3f}ms | {tokens_per_sec:5.1f} tokens/s")
    last_step_time = current_time
    if torch_config.is_master_process and train_conf.use_wandb:
        wandb.log({
            "train/loss": step_stats['loss'].item(),
            "train/loss_norm": step_stats['loss_norm'].item(),
            "train/lr": get_lr(train_conf.step),
            # "train/lr": lr[0],
            "train/tokens_per_sec": tokens_per_sec,
            "train/step_time_ms": step_dt_ms,
            "step": train_conf.step,
        })
    # checkpoints
    in_checkpoint_step = train_conf.step > 0 and (train_conf.step % train_conf.save_checkpoint_freq == 0 or train_conf.step == train_conf.n_training_steps - 1)
    if torch_config.is_master_process and in_checkpoint_step:
        save_checkpoint(raw_model, optimizer, train_conf, step_stats)

model_state = model.state_dict()
torch.save(model_state, "latest_model_params.pth")
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=torch_config.device)

# generate_text(model, tokenizer, torch_config)

if torch_config.using_ddp:
    destroy_process_group()
