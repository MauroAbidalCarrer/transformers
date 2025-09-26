from dataclasses import dataclass

import torch
import tiktoken


# Data hyper parameters
TEST_SPLIT_RATIO = 0.1
BATCH_SIZE = 3
LEARNING_RATE = 3e-4
N_TRAINING_STEPS = 600
LOGGING_INTERVAL = 500
ENCODING_NAME = "gpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class GPTConfig:
    attention_window_size: int = 1024 
    vocab_size: int = tiktoken.get_encoding(ENCODING_NAME).max_token_value + 1
    n_transformer_blocks: int = 12 
    n_heads: int = 12 
    n_embed_dim: int = 768