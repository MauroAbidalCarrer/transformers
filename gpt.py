from math import sqrt
from functools import partial

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


# Data hyper parameters
TEST_SPLIT_RATIO = 0.1
# Model hyper parameters
ATTENTION_WINDOW_SIZE = 256
N_EMBEDING_DIMS = 384
N_HEADS = 6
N_TRANSFORMER_BLOCKS = 6
# ATTENTION_EXPANSION_RATIO = 4
ATTENTION_DROPOUT = 0
MLP_EXPANSION_RATIO = 4
MLP_DROPOUT = 0.15
# training hyper parameters
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
N_TRAINING_STEPS = 2000
LOGGING_INTERVAL = 500

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O ~/input.txt

with open("/root/input.txt", 'r', encoding='utf-8') as f:
    shakespeare_txt = f.read()

vocab = sorted(list(set(shakespeare_txt)))
vocab_len = len(vocab)
str_to_char_idx = { ch:i for i,ch in enumerate(vocab) }
token_idx_to_str = dict(enumerate(vocab))
encode = lambda string: [str_to_char_idx[char] for char in string]
decode = lambda tokens_idx: "".join([token_idx_to_str[token_idx] for token_idx in tokens_idx])

device = "cuda" if torch.cuda.is_available() else "cpu"

encoded_txt = encode(shakespeare_txt)
dataset = torch.tensor(encoded_txt, dtype=torch.long).to(device)
n_test_samples = int(TEST_SPLIT_RATIO * len(shakespeare_txt))
train = dataset[:-n_test_samples]
test = dataset[-n_test_samples:]


def get_random_batch(split: Tensor) -> tuple[Tensor, Tensor]:
    rand_idx = torch.randint(high=len(split) - ATTENTION_WINDOW_SIZE, size=(BATCH_SIZE, ))
    x = torch.stack([split[idx:idx + ATTENTION_WINDOW_SIZE] for idx in rand_idx])
    y = torch.stack([split[idx + 1:idx + ATTENTION_WINDOW_SIZE + 1] for idx in rand_idx])
    # y = F.one_hot(y, num_classes=vocab_len).float()
    x, y = x.to(device), y.to(device)
    return x, y

def eval_model(model: nn.Module, x: Tensor, y_true: Tensor) -> dict[str, float]:
    model = model.eval()
    with torch.no_grad():
        y_pred = model(x) # (batch size, window size, n embeding dims)
        y_pred = y_pred.reshape(BATCH_SIZE * ATTENTION_WINDOW_SIZE, vocab_len) # (batch size * window size, n embeding dims)
        y_true = y_true.reshape(BATCH_SIZE * ATTENTION_WINDOW_SIZE)
        return {
            "loss": F.cross_entropy(y_pred, y_true).cpu().item(),
            "accuracy": (torch.argmax(y_pred, dim=1) == y_true).float().mean().cpu().item(),
        }


class MaskedAttentionHead(nn.Module):
    def __init__(self, head_size:int, dropout_ratio: float=0):
        """
        ### Args:
        head_size: number of dimensions for key, query and values vectors.

        The forward call will project the tokens back to their embeding size.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(N_EMBEDING_DIMS)
        self.head_size = head_size
        self.keys_projection = nn.Linear(N_EMBEDING_DIMS, head_size, bias=False)
        self.queries_projection = nn.Linear(N_EMBEDING_DIMS, head_size, bias=False)
        self.values_projection = nn.Linear(N_EMBEDING_DIMS, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(ATTENTION_WINDOW_SIZE, ATTENTION_WINDOW_SIZE)))
        self.dropout = nn.Dropout(dropout_ratio)

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
        attention_weights = self.dropout(attention_weights)
        out = attention_weights @ values

        return out

class MultiHeadMaskedAttention(nn.Module):
    def __init__(self, n_heads: int, dropout: float):
        super().__init__()
        assert (N_EMBEDING_DIMS % n_heads) == 0, "N_EMBEDING_DIMS must be dividable by n_heads"
        head_size = N_EMBEDING_DIMS // n_heads
        self.heads = nn.ModuleList([MaskedAttentionHead(head_size, dropout) for _ in range(n_heads)])
        self.post_head_projection = nn.Linear(N_EMBEDING_DIMS, N_EMBEDING_DIMS)

    def forward(self, x: Tensor) -> Tensor:
        attended = [head(x) for head in self.heads]
        attended = torch.cat(attended, dim=2)
        attended = self.post_head_projection(attended)
        return attended

class MLPBlock(nn.Sequential):
    def __init__(self, expantion_ratio: int, dropout_ratio: float):
        n_expanded_dims = N_EMBEDING_DIMS * expantion_ratio
        super().__init__(
            nn.LayerNorm(N_EMBEDING_DIMS),
            nn.Linear(N_EMBEDING_DIMS, n_expanded_dims),
            nn.ReLU(),
            # Small diviation from Andrej Karpathy's repo where the dropout is at the end.
            nn.Linear(n_expanded_dims, N_EMBEDING_DIMS),
            nn.Dropout(dropout_ratio), 
        )

class TransformerBlock(nn.Module):
    def __init__(self, n_heads: int, head_dropout: float, mlp_expansion: int, mlp_dropout: float):
        super().__init__()
        self.attention_head = MultiHeadMaskedAttention(n_heads, head_dropout)
        self.mlp = MLPBlock(mlp_expansion, mlp_dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention_head(x)
        x = x + self.mlp(x)
        return x

class GPT(nn.Module):
    def __init__(self, n_transformer_blocks: int=3):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_len, N_EMBEDING_DIMS)
        self.positional_embedding = nn.Embedding(ATTENTION_WINDOW_SIZE, N_EMBEDING_DIMS)
        mk_t_block = partial(TransformerBlock, N_HEADS, ATTENTION_DROPOUT, MLP_EXPANSION_RATIO, MLP_DROPOUT)
        self.transformer_blocks = nn.Sequential(*[mk_t_block() for _ in range(n_transformer_blocks)])
        self.un_embedding_layer = nn.Linear(N_EMBEDING_DIMS, vocab_len)

        # better init, not covered in the original GPT video, but important, will cover in followup video
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
            idx_cond = tokens[:, -ATTENTION_WINDOW_SIZE:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            tokens = torch.cat((tokens, idx_next), dim=1) # (B, T+1)
        return tokens

if __name__ == "__main__":
    model = GPT(n_transformer_blocks=N_TRANSFORMER_BLOCKS).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(N_TRAINING_STEPS):

        if iter % LOGGING_INTERVAL == 0 or iter == N_TRAINING_STEPS - 1:
            train_batch = get_random_batch(train)
            train_metrics = eval_model(model, *train_batch)
            print(f"step {iter}: train loss {train_metrics['loss']:.4f}, train accuracy {train_metrics['accuracy']:.4f}")
            test_batch = get_random_batch(test)
            test_metrics = eval_model(model, *test_batch)
            print(f"step {iter}: val loss {test_metrics['loss']:.4f}, val accuracy {test_metrics['accuracy']:.4f}")

        model = model.train()
        # sample a batch of data
        x, y_true = get_random_batch(train)
        # evaluate the loss
        y_pred = model(x).reshape(BATCH_SIZE * ATTENTION_WINDOW_SIZE, vocab_len)
        y_true = y_true.reshape(BATCH_SIZE * ATTENTION_WINDOW_SIZE)
        loss = F.cross_entropy(y_pred, y_true)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model_state = model.state_dict()
    torch.save(model_state, "latest_model_params.pth")
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
