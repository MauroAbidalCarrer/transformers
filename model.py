from functools import partial

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from config import GPTConfig


class MultiHeadMaskedAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert (config.n_embed_dim % config.n_heads) == 0, "config.n_embed_dim must be dividable by n_heads."
        # Output is n_embedding_dim * 3 because query, key and value weights packed into a single linear layer.
        # So the actual n emedding dims of the key, values and queries will equal n_embed_dim
        self.qkv_weights = nn.Linear(config.n_embed_dim, 3 * config.n_embed_dim)
        self.post_head_projection = nn.Linear(config.n_embed_dim, config.n_embed_dim)
        self.post_head_projection.is_proj_layer = True
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, n_embed = x.shape
        # Compute the queries, keys and values in a single linear forward pass and then split them into separate views.
        queries, keys, values = self.qkv_weights(x).split(n_embed, -1)
        shape_for_attention = (batch_size, seq_len, self.config.n_heads, n_embed // self.config.n_heads)
        queries = queries.reshape(*shape_for_attention).transpose(1, 2)
        keys = keys.reshape(*shape_for_attention).transpose(1, 2)
        values = values.reshape(*shape_for_attention).transpose(1, 2)
        y = F.scaled_dot_product_attention(queries, keys, values, is_causal=True) # (batch size, n_heads, seq len, embed // n_heads)
        y = y.transpose(1, 2) # (batch size, n_heads, seq len, embed)
        y = y.reshape(batch_size, seq_len, n_embed) # (batch size, seq len, embed)
        return y

class MLPBlock(nn.Sequential):
    def __init__(self, config: GPTConfig):
        n_expanded_dims = config.n_embed_dim * 4
        proj_layer = nn.Linear(n_expanded_dims, config.n_embed_dim)
        proj_layer.is_proj_layer = True
        super().__init__(
            nn.LayerNorm(config.n_embed_dim),
            nn.Linear(config.n_embed_dim, n_expanded_dims),
            nn.GELU(),
            proj_layer,
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
        self.un_embedding_layer = nn.Linear(config.n_embed_dim, config.vocab_size, bias=False)
        self.un_embedding_layer.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # If the layer is layer that projects the back into the residual stream
            if getattr(module, 'is_proj_layer', False):
                std *= (2 * self.config.n_transformer_blocks) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, tokens_idx: Tensor) -> Tensor:
        seq_len = tokens_idx.shape[1]
        token_postions = torch.arange(seq_len, device=tokens_idx.device)
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