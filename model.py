import inspect
from functools import partial

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from config import GPTConfig, TorchConfig


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
        shape_for_attention = (
            batch_size,
            seq_len,
            self.config.n_heads,
            n_embed // self.config.n_heads,
        )
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
        self.token_embedding = nn.Embedding(config.model_vocab_size, config.n_embed_dim)
        self.positional_embedding = nn.Embedding(config.attention_window_size, config.n_embed_dim)
        mk_t_block = partial(TransformerBlock, config)
        self.transformer_blocks = nn.Sequential(*[mk_t_block() for _ in range(config.n_transformer_blocks)])
        self.un_embedding_layer = nn.Linear(config.n_embed_dim, config.model_vocab_size, bias=False)
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
            # crop idx to the last attention_window_size tokens
            idx_cond = tokens[:, -self.config.attention_window_size:]
            logits = self(idx_cond)
            # only get the next prediction of the last token, i.e the pred for the next token (B, C)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            tokens = torch.cat((tokens, idx_next), dim=1) # (B, T+1)
        return tokens
    
    def get_params_stats(self) -> dict:
        parmaters_count = 0
        model_memory_usage = 0
        for param in self.parameters():
            model_memory_usage += param.nelement() * param.element_size()
            parmaters_count += param.nelement()
        parmaters_count /= 1e6
        model_memory_usage /= 1024 ** 2
        return {
            "count": parmaters_count,
            "mem_usage": model_memory_usage,
        }

# class CausalSelfAttention(nn.Module):

#     def __init__(self, config: GPTConfig):
#         super().__init__()
#         assert config.n_embed_dim % config.n_heads == 0
#         # key, query, value projections for all heads, but in a batch
#         self.c_attn = nn.Linear(config.n_embed_dim, 3 * config.n_embed_dim)
#         # output projection
#         self.c_proj = nn.Linear(config.n_embed_dim, config.n_embed_dim)
#         self.c_proj.NANOGPT_SCALE_INIT = 1
#         # regularization
#         self.n_heads = config.n_heads
#         self.n_embd = config.n_embed_dim

#     def forward(self, x):
#         B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
#         # e.g. in GPT-2 (124M), n_heads=12, hs=64, so nh*hs=C=768 channels in the Transformer
#         qkv = self.c_attn(x)
#         q, k, v = qkv.split(self.n_embd, dim=2)
#         k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
#         q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
#         v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
#         y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
#         y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
#         # output projection
#         y = self.c_proj(y)
#         return y

# class MLP(nn.Module):

#     def __init__(self, config: GPTConfig):
#         super().__init__()
#         self.c_fc    = nn.Linear(config.n_embed_dim, 4 * config.n_embed_dim)
#         self.gelu    = nn.GELU(approximate='tanh')
#         self.c_proj  = nn.Linear(4 * config.n_embed_dim, config.n_embed_dim)
#         self.c_proj.NANOGPT_SCALE_INIT = 1

#     def forward(self, x):
#         x = self.c_fc(x)
#         x = self.gelu(x)
#         x = self.c_proj(x)
#         return x

# class Block(nn.Module):

#     def __init__(self, config: GPTConfig):
#         super().__init__()
#         self.ln_1 = nn.LayerNorm(config.n_embed_dim)
#         self.attn = CausalSelfAttention(config)
#         self.ln_2 = nn.LayerNorm(config.n_embed_dim)
#         self.mlp = MLP(config)

#     def forward(self, x):
#         x = x + self.attn(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x

# class GPT(nn.Module):

#     def __init__(self, config: GPTConfig):
#         super().__init__()
#         self.config = config

#         self.transformer = nn.ModuleDict(dict(
#             wte = nn.Embedding(config.tokenizer_vocab_size, config.n_embed_dim),
#             wpe = nn.Embedding(config.attention_window_size, config.n_embed_dim),
#             h = nn.ModuleList([Block(config) for _ in range(config.n_transformer_blocks)]),
#             ln_f = nn.LayerNorm(config.n_embed_dim),
#         ))
#         self.lm_head = nn.Linear(config.n_embed_dim, config.tokenizer_vocab_size, bias=False)

#         # weight sharing scheme
#         self.transformer.wte.weight = self.lm_head.weight

#         # init params
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             std = 0.02
#             if hasattr(module, 'NANOGPT_SCALE_INIT'):
#                 std *= (2 * self.config.n_transformer_blocks) ** -0.5
#             torch.nn.init.normal_(module.weight, mean=0.0, std=std)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

#     def forward(self, idx, targets=None):
#         # idx is of shape (B, T)
#         B, T = idx.size()
#         assert T <= self.config.attention_window_size, f"Cannot forward sequence of length {T}, block size is only {self.config.attention_window_size}"
#         # forward the token and posisition embeddings
#         pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
#         pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
#         tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
#         x = tok_emb + pos_emb
#         # forward the blocks of the transformer
#         for block in self.transformer.h:
#             x = block(x)
#         # forward the final layernorm and the classifier
#         x = self.transformer.ln_f(x)
#         logits = self.lm_head(x) # (B, T, vocab_size)
#         return logits

#     def configure_optimizers(self, weight_decay, learning_rate, torch_conf: TorchConfig):
#         # start with all of the candidate parameters (that require grad)
#         param_dict = {pn: p for pn, p in self.named_parameters()}
#         param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
#         # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
#         # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
#         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
#         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
#         optim_groups = [
#             {'params': decay_params, 'weight_decay': weight_decay},
#             {'params': nodecay_params, 'weight_decay': 0.0}
#         ]
#         num_decay_params = sum(p.numel() for p in decay_params)
#         num_nodecay_params = sum(p.numel() for p in nodecay_params)
#         if torch_conf.is_master_process:
#             print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
#             print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
#         # Create AdamW optimizer and use the fused version if it is available
#         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
#         use_fused = fused_available and torch_conf.device_type == "cuda"
#         if torch_conf.is_master_process:
#             print(f"using fused AdamW: {use_fused}")
#         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
#         return optimizer

#     def get_params_stats(self) -> dict:
#         parmaters_count = 0
#         model_memory_usage = 0
#         for param in self.parameters():
#             model_memory_usage += param.nelement() * param.element_size()
#             parmaters_count += param.nelement()
#         parmaters_count /= 1e6
#         model_memory_usage /= 1024 ** 2
#         return {
#             "count": parmaters_count,
#             "mem_usage": model_memory_usage,
#         }
