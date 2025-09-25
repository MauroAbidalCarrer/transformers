from time import time
from math import sqrt
from functools import partial
from dataclasses import dataclass

import torch
import tiktoken
from torch import nn
from torch import Tensor
from torch.nn import functional as F


# Data hyper parameters
TEST_SPLIT_RATIO = 0.1
BATCH_SIZE = 1
LEARNING_RATE = 3e-4
N_TRAINING_STEPS = 2000
LOGGING_INTERVAL = 500

device = "cuda" if torch.cuda.is_available() else "cpu"

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", 'r', encoding='utf-8') as f:
    shakespeare_txt = f.read()
tokenizer = tiktoken.get_encoding("gpt2")
encoded_txt = tokenizer.encode(shakespeare_txt)
dataset = torch.tensor(encoded_txt, dtype=torch.long) #.to(device)
n_test_samples = int(TEST_SPLIT_RATIO * len(shakespeare_txt))
train = dataset[:-n_test_samples]
test = dataset[-n_test_samples:]

@dataclass
class GPTConfig:
    attention_window_size: int = 1024 
    vocab_size: int = tokenizer.max_token_value + 1
    n_transformer_blocks: int = 12 
    n_heads: int = 12 
    n_embed_dim: int = 768 

def get_random_batch(split: Tensor, config: GPTConfig) -> tuple[Tensor, Tensor]:
    rand_idx = torch.randint(high=len(split) - config.attention_window_size, size=(BATCH_SIZE, ))
    x = torch.stack([split[idx:idx + config.attention_window_size] for idx in rand_idx])
    y = torch.stack([split[idx + 1:idx + config.attention_window_size + 1] for idx in rand_idx])
    x, y = x.to(device), y.to(device)
    return x, y

def eval_model(model: nn.Module, x: Tensor, y_true: Tensor, config: GPTConfig) -> dict[str, float]:
    model = model.eval()
    with torch.no_grad():
        y_pred = model(x) # (batch size, window size, n embeding dims)
        y_pred = y_pred.reshape(BATCH_SIZE * config.attention_window_size, config.vocab_size) # (batch size * window size, n embeding dims)
        y_true = y_true.reshape(BATCH_SIZE * config.attention_window_size)
        return {
            "loss": F.cross_entropy(y_pred, y_true).cpu().item(),
            "accuracy": (torch.argmax(y_pred, dim=1) == y_true).float().mean().cpu().item(),
        }

class MaskedAttentionHead(nn.Module):
    def __init__(self, config: GPTConfig):
        """
        ### Args:
        head_size: number of dimensions for key, query and values vectors.

        The forward call will project the tokens back to their embeding size.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.n_embed_dim)
        self.head_size = config.n_embed_dim // config.n_heads
        self.keys_projection = nn.Linear(config.n_embed_dim, self.head_size, bias=False)
        self.queries_projection = nn.Linear(config.n_embed_dim, self.head_size, bias=False)
        self.values_projection = nn.Linear(config.n_embed_dim, self.head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(config.attention_window_size, config.attention_window_size)))

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        x = self.layer_norm(x)
        keys: Tensor = self.keys_projection(x)
        queries = self.queries_projection(x)
        values = self.values_projection(x)
        attention_weights = queries @ keys.swapaxes(1, 2)
        attention_weights /= sqrt(self.head_size)
        attention_weights = torch.masked_fill(attention_weights, self.mask[:seq_len, :seq_len] == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        out = attention_weights @ values

        return out

class MultiHeadMaskedAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert (config.n_embed_dim % config.n_heads) == 0, "config.n_embed_dim must be dividable by n_heads"
        head_size = config.n_embed_dim // config.n_heads
        self.heads = nn.ModuleList([MaskedAttentionHead(config) for _ in range(config.n_heads)])
        self.post_head_projection = nn.Linear(config.n_embed_dim, config.n_embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        attended = [head(x) for head in self.heads]
        attended = torch.cat(attended, dim=2)
        attended = self.post_head_projection(attended)
        return attended

class MLPBlock(nn.Sequential):
    def __init__(self, config: GPTConfig):
        n_expanded_dims = config.n_embed_dim * 4
        super().__init__(
            nn.LayerNorm(config.n_embed_dim),
            nn.Linear(config.n_embed_dim, n_expanded_dims),
            nn.ReLU(),
            nn.Linear(n_expanded_dims, config.n_embed_dim),
        )

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_head = MultiHeadMaskedAttention(config)
        self.mlp = MLPBlock(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention_head(x)
        x = x + self.mlp(x)
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed_dim)
        self.positional_embedding = nn.Embedding(config.attention_window_size, config.n_embed_dim)
        mk_t_block = partial(TransformerBlock, config)
        self.transformer_blocks = nn.Sequential(*[mk_t_block() for _ in range(config.n_transformer_blocks)])
        self.un_embedding_layer = nn.Linear(config.n_embed_dim, config.vocab_size)
        # self.un_embedding_layer = self.token_embedding
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens_idx: Tensor) -> Tensor:
        seq_len = tokens_idx.shape[1]
        token_postions = torch.arange(seq_len, device=device)
        positional_embedded_tokens = self.positional_embedding(token_postions)
        value_embedded_tokens = self.token_embedding(tokens_idx)
        embedded_tokens = positional_embedded_tokens + value_embedded_tokens
        processed_stream = self.transformer_blocks(embedded_tokens)
        output_tokens_probabilities = self.un_embedding_layer(processed_stream)
        return output_tokens_probabilities

    def generate(self, tokens: Tensor, max_new_tokens: int) -> Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = tokens[:, -self.config.attention_window_size:]
            logits = self(idx_cond)
            # only get the next prediction of the last token, i.e the pred for the next token (B, C)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            tokens = torch.cat((tokens, idx_next), dim=1) # (B, T+1)
        return tokens

if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config).to(device)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    size_all_mb = param_size / 1024**2
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
                train_metrics = eval_model(model, *train_batch, config)
                test_batch = get_random_batch(test, config)
                test_metrics = eval_model(model, *test_batch, config)
                # print(f"step {step}: val loss , val accuracy {test_metrics['accuracy']:.4f}")
                logging_format = "step: {step:4d} | train loss: {train_loss:5.3f} | val loss: {test_loss:5.3f} | dt: {dt:5.0f}ms | tokens/s: {tokens_per_sec:5.0f}"
                print(logging_format.format(
                    step=step,
                    train_loss=train_metrics["loss"],
                    test_loss=test_metrics["loss"],
                    dt=time_to_last_log_step_ms,
                    tokens_per_sec=n_processed_tokens / time_to_last_log_step_ms
                ))
                last_log_step = step
                last_log_iter_start = time()

        model = model.train()
        # sample a batch of data
        x, y_true = get_random_batch(train, config)
        # evaluate the loss
        y_pred = model(x).reshape(BATCH_SIZE * config.attention_window_size, config.vocab_size)
        y_true = y_true.reshape(BATCH_SIZE * config.attention_window_size)
        loss = F.cross_entropy(y_pred, y_true)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model_state = model.state_dict()
    torch.save(model_state, "latest_model_params.pth")
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
