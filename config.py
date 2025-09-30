import os
import argparse
from dataclasses import dataclass, field

import torch
import tiktoken


ENCODING_NAME = "gpt2"

@dataclass
class GPTConfig:
    attention_window_size: int = 1024 
    model_vocab_size: int = tiktoken.get_encoding(ENCODING_NAME).max_token_value + 1
    tokenizer_vocab_size: int = tiktoken.get_encoding(ENCODING_NAME).max_token_value + 1
    n_transformer_blocks: int = 12 
    n_heads: int = 12 
    n_embed_dim: int = 768

@dataclass
class TrainingConfig:
    # required first (no defaults)
    model_config: "GPTConfig" = field(repr=False)
    n_gpus: int = field(default=0, repr=False)

    # optional with defaults
    micro_batch_size: int = field(default=4)
    tokens_per_step: int = field(default=2**19)  # ~0.5M tokens
    n_training_steps: int = field(default=19000)
    train_test_split_ratio: float = field(default=0.1)
    log_interval: int = field(default=500)
    max_lr: float = field(default=6e-4)
    n_warmup_steps: int = field(default=715)
    weight_decay: float = field(default=0.1)
    betas: tuple = field(default=(0.9, 0.95))
    eps: float = field(default=1e-8)
    save_checkpoint_freq: int = field(default=2)
    validation_freq: int = field(default=100)
    hella_swag_eval_freq: int = field(default=300)
    text_gen_freq: int = field(default=300)
    starting_step: int = field(default=0, init=False)

    # derived attributes
    seq_len: int = field(init=False)
    tokens_per_micro_step: int = field(init=False)
    grad_accum_step: int = field(init=False)
    min_lr: float = field(init=False)
    use_wandb: bool = field(init=False)

    def __post_init__(self):
        self.seq_len = self.model_config.attention_window_size
        self.tokens_per_micro_step = self.micro_batch_size * self.seq_len * self.n_gpus
        assert self.tokens_per_step % self.tokens_per_micro_step == 0, (
            "sequences per batch should be divisible by tokens per batch"
        )
        self.grad_accum_step = self.tokens_per_step // self.tokens_per_micro_step
        self.min_lr = self.max_lr / 10

        parser = argparse.ArgumentParser()
        parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases tracking")
        args = parser.parse_args()
        self.use_wandb = not args.no_wandb
        self.step = self.starting_step

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
