from tqdm import tqdm
from time import time

import torch
import tiktoken
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from model import GPT
from config import (
    GPTConfig,
    TrainingConfig,
    OptimizerConfig,
    ENCODING_NAME,
    device,
)


model_conf = GPTConfig()
train_conf = TrainingConfig(model_conf)
optim_conf = OptimizerConfig()


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

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", 'r', encoding='utf-8') as f:
    shakespeare_txt = f.read()
tokenizer = tiktoken.get_encoding(ENCODING_NAME)
encoded_txt = tokenizer.encode(shakespeare_txt)
dataset = torch.tensor(encoded_txt, dtype=torch.long)
n_test_samples = int(train_conf.train_test_split_ratio * len(shakespeare_txt))
train = dataset[:-n_test_samples]
test = dataset[-n_test_samples:]

torch.set_float32_matmul_precision('high')
print("precision set to high")
model = torch.compile(GPT(model_conf).to(device))
parmaters_count = 0
model_memory_usage = 0
for param in model.parameters():
    model_memory_usage += param.nelement() * param.element_size()
    parmaters_count += param.nelement()
parmaters_count /= 1e6
model_memory_usage /= 1024 ** 2
print(f"number of parameters: {parmaters_count:.2f}M, model memory usage: {model_memory_usage:.2f}MB")

optimizer = torch.optim.AdamW(model.parameters(), lr=optim_conf.learning_rate, fused=True)

last_log_step = 0
last_log_iter_start = time()
last_step_time = time()
for step in range(train_conf.n_training_steps):
    if step % train_conf.log_interval == 0 or step == train_conf.n_training_steps - 1:
        n_processed_tokens = (step - last_log_step) * train_conf.micro_batch_size * model_conf.attention_window_size * train_conf.grad_accum_step
        time_to_last_log_step_ms = (time() - last_log_iter_start) * 1000
        with torch.no_grad():
            model = model.eval()
            train_batch = get_random_batch(train)
            train_metrics = eval_model(model, *train_batch)
            test_batch = get_random_batch(test)
            test_metrics = eval_model(model, *test_batch)
            logging_format = "step: {step:4d} | train loss: {train_loss:5.3f} | val loss: {test_loss:5.3f} | dt: {dt:5.0f}ms | tokens/s: {tokens_per_sec:5.0f}"
            print(
                logging_format.format(
                    step=step,
                    train_loss=train_metrics["loss"],
                    test_loss=test_metrics["loss"],
                    dt=time_to_last_log_step_ms,
                    tokens_per_sec= 1000 * n_processed_tokens / time_to_last_log_step_ms
                )
            )
            last_log_step = step
            last_log_iter_start = time()

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

    current_time = time()
    step_dt_ms = (current_time - last_step_time) * 1000
    tokjens_per_sec = train_conf.tokens_per_batch / (current_time - last_step_time)
    print(f"step {step:4d} | batch loss {batch_loss:5.3f} | batch loss norm {norm:3.1f} | dt {step_dt_ms:5.3f}ms | {tokjens_per_sec:5.1f} tokens/s")
    last_step_time = current_time

model_state = model.state_dict()
torch.save(model_state, "latest_model_params.pth")
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)

print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
