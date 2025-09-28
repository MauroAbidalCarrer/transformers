from dataclasses import dataclass

import torch
import tiktoken


ENCODING_NAME = "gpt2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class GPTConfig:
    attention_window_size: int = 1024 
    vocab_size: int = tiktoken.get_encoding(ENCODING_NAME).max_token_value + 1
    n_transformer_blocks: int = 12 
    n_heads: int = 12 
    n_embed_dim: int = 768

@dataclass
class TrainingConfig:
    # number of batches computed in parallel on the GPU
    # Set it to the maximum batch size the GPU can hold
    micro_batch_size = 24
    # Number of tokens to process before performing backward step
    # 2**19 = ~0.5M of tokens per batch
    tokens_per_batch = 2**19 
    n_training_steps = 1500
    train_test_split_ratio = 0.1
    log_interval = 500

    def __init__(self, model_config: GPTConfig):
        assert self.tokens_per_batch % self.tokens_per_batch == 0, "sequences per batch should be dividable by tokens per batch"
        self.seq_len = model_config.attention_window_size
        self.grad_accum_step = self.tokens_per_batch // (self.micro_batch_size * self.seq_len)

@dataclass
class OptimizerConfig:
    learning_rate = 3e-4
