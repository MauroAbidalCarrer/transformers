# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# torchrun --standalone  --nproc_per_node=4 main.py
import os
from time import time

import torch
import tiktoken
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group

from model import GPT
from config import (
    GPTConfig,
    TrainingConfig,
    ENCODING_NAME,
)
from optimization_utils import mk_scheduler, mk_optimizer


ddp_rank = os.environ.get("RANK", -1)
using_ddp = ddp_rank != -1
def master_print(*args, **kwargs):
    if master_process:
        print(*args, **kwargs)

if using_ddp:
    init_process_group(backend="nccl")
ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
master_process = (ddp_local_rank == 0)
ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
device = torch.device(f"cuda:{ddp_local_rank}")
torch.cuda.set_device(device)
print("local rank:", ddp_local_rank, "master_process:", master_process)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

model_conf = GPTConfig(vocab_size=50304)
train_conf = TrainingConfig(model_conf)

def get_random_batch(split: Tensor) -> tuple[Tensor, Tensor]:
    rand_idx = torch.randint(high=len(split) - model_conf.attention_window_size, size=(train_conf.micro_batch_size, ))
    x = torch.stack([split[idx:idx + model_conf.attention_window_size] for idx in rand_idx])
    y = torch.stack([split[idx + 1:idx + model_conf.attention_window_size + 1] for idx in rand_idx])
    x, y = x.to(device), y.to(device)
    return x, y

def eval_model(model: nn.Module, x: Tensor, y_true: Tensor) -> dict[str, float]:
    model = model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        y_pred = model(x) # (batch size, window size, n embeding dims)
        y_pred = y_pred.reshape(train_conf.micro_batch_size * model_conf.attention_window_size, model_conf.vocab_size) # (batch size * window size, n embeding dims)
        y_true = y_true.reshape(train_conf.micro_batch_size * model_conf.attention_window_size)
        return {
            "loss": F.cross_entropy(y_pred, y_true).cpu().item(),
            "accuracy": (torch.argmax(y_pred, dim=1) == y_true).float().mean().cpu().item(),
        }

with open("input.txt", 'r', encoding='utf-8') as f:
    shakespeare_txt = f.read()
tokenizer = tiktoken.get_encoding(ENCODING_NAME)
encoded_txt = tokenizer.encode(shakespeare_txt)
dataset = torch.tensor(encoded_txt, dtype=torch.long)
n_test_samples = int(train_conf.train_test_split_ratio * len(shakespeare_txt))
train = dataset[:-n_test_samples]
test = dataset[-n_test_samples:]

torch.set_float32_matmul_precision('high')
model = torch.compile(GPT(model_conf).to(device))
parmaters_count = 0
model_memory_usage = 0
for param in model.parameters():
    model_memory_usage += param.nelement() * param.element_size()
    parmaters_count += param.nelement()
parmaters_count /= 1e6
model_memory_usage /= 1024 ** 2
master_print(f"number of parameters: {parmaters_count:.2f}M, model memory usage: {model_memory_usage:.2f}MB")

optimizer = mk_optimizer(model, train_conf)
scheduler = mk_scheduler(optimizer, train_conf)

last_log_step = 0
last_log_iter_start = time()
last_step_time = time()
for step in range(train_conf.n_training_steps):

    model = model.train()
    optimizer.zero_grad()
    batch_loss = 0
    for micro_step in range(train_conf.grad_accum_step):
        # sample a batch of data
        x, y_true = get_random_batch(train)
        y_true = y_true.reshape(train_conf.micro_batch_size * model_conf.attention_window_size)
        # evaluate the loss
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y_pred = model(x).reshape(train_conf.micro_batch_size * model_conf.attention_window_size, model_conf.vocab_size)
            micro_batch_loss = F.cross_entropy(y_pred, y_true) / train_conf.grad_accum_step
            micro_batch_loss.backward()
            batch_loss += micro_batch_loss.item()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    current_time = time()
    step_dt_ms = (current_time - last_step_time) * 1000
    tokjens_per_sec = train_conf.tokens_per_step / (current_time - last_step_time)
    lr = scheduler.get_last_lr()
    master_print(f"step {step:4d} | batch loss {batch_loss:5.3f} | batch loss norm {norm:3.1f} | lr {lr[0]:10.7f} | dt {step_dt_ms:5.3f}ms | {tokjens_per_sec:5.1f} tokens/s")
    last_step_time = current_time

    
model_state = model.state_dict()
torch.save(model_state, "latest_model_params.pth")
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)

master_print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))

if using_ddp:
    destroy_process_group()