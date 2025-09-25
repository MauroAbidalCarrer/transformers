from math import sqrt
from functools import partial

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


from config import GPTConfig, device

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
