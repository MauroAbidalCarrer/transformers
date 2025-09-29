import os
from dataclasses import dataclass

import torch
import tiktoken


ENCODING_NAME = "gpt2"

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
    micro_batch_size = 4
    # Number of tokens to process before performing backward step
    # 2**19 = ~0.5M of tokens per batch
    tokens_per_step = 2**19 
    n_training_steps = 19000
    train_test_split_ratio = 0.1
    log_interval = 500
    max_lr = 6e-4
    n_warmup_steps = 715
    weight_decay = 0.1
    betas = (0.9, 0.95)
    eps = 1e-8
    save_checkpoint_freq = 2
    validation_freq = 100
    hella_swag_eval_freq = 300

    def __init__(self, model_config: GPTConfig, n_gpus=0):
        self.seq_len = model_config.attention_window_size
        self.tokens_per_micro_step = self.micro_batch_size * self.seq_len * n_gpus
        assert self.tokens_per_step % self.tokens_per_micro_step == 0, "sequences per batch should be dividable by tokens per batch"
        self.grad_accum_step = self.tokens_per_step // self.tokens_per_micro_step
        self.min_lr = self.max_lr / 10

class TorchConfig:
    def __init__(self):
        self.ddp_rank = int(os.environ.get("RANK", -1))
        self.using_ddp = self.ddp_rank != -1
        self.ddp_rank = self.ddp_rank if self.using_ddp else 0
        self.ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.is_master_process = (self.ddp_local_rank == 0)
        self.device = torch.device(f"cuda:{self.ddp_local_rank}")
        self.device_type = self.device.type    
