from time import time

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from config import *
from model import GPTConfig, GPT


def get_random_batch(split: Tensor, config: GPTConfig) -> tuple[Tensor, Tensor]:
    rand_idx = torch.randint(high=len(split) - config.attention_window_size, size=(BATCH_SIZE, ))
    x = torch.stack([split[idx:idx + config.attention_window_size] for idx in rand_idx])
    y = torch.stack([split[idx + 1:idx + config.attention_window_size + 1] for idx in rand_idx])
    x, y = x.to(device), y.to(device)
    return x, y

def eval_model(model: nn.Module, x: Tensor, y_true: Tensor, config: GPTConfig, device: torch.device) -> dict[str, float]:
    model = model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        y_pred = model(x) # (batch size, window size, n embeding dims)
        y_pred = y_pred.reshape(BATCH_SIZE * config.attention_window_size, config.vocab_size) # (batch size * window size, n embeding dims)
        y_true = y_true.reshape(BATCH_SIZE * config.attention_window_size)
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
n_test_samples = int(TEST_SPLIT_RATIO * len(shakespeare_txt))
train = dataset[:-n_test_samples]
test = dataset[-n_test_samples:]

config = GPTConfig()
model = GPT(config).to(device)
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
size_all_mb = param_size / 1024 ** 2
print('model size: {:.3f}MB'.format(size_all_mb))

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

last_log_step = 0
last_log_iter_start = time()
for step in range(N_TRAINING_STEPS):
    if step % LOGGING_INTERVAL == 0 or step == N_TRAINING_STEPS - 1:
        n_processed_tokens = (step - last_log_step) * BATCH_SIZE * config.attention_window_size
        time_to_last_log_step_ms = (time() - last_log_iter_start) * 1000
        with torch.no_grad():
            model = model.eval()
            train_batch = get_random_batch(train, config)
            train_metrics = eval_model(model, *train_batch, config, device)
            test_batch = get_random_batch(test, config)
            test_metrics = eval_model(model, *test_batch, config, device)
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
    # sample a batch of data
    x, y_true = get_random_batch(train, config)
    y_true = y_true.reshape(BATCH_SIZE * config.attention_window_size)
    # evaluate the loss
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        y_pred = model(x).reshape(BATCH_SIZE * config.attention_window_size, config.vocab_size)
        loss = F.cross_entropy(y_pred, y_true)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model_state = model.state_dict()
torch.save(model_state, "latest_model_params.pth")
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)

print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
