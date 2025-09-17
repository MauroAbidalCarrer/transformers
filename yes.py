from functools import partial

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional

TEST_SPLIT_RATIO = 0.1
ATTENTION_WINDOW_SIZE = 256
BATCH_SIZE = 64


# ! wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open("input.txt", 'r', encoding='utf-8') as f:
    shakespeare_txt = f.read()

vocab = sorted(list(set(shakespeare_txt)))
vocab_len = len(vocab)
str_to_char_idx = { ch:i for i,ch in enumerate(vocab) }
token_idx_to_str = dict(enumerate(vocab))
encode = lambda string: [str_to_char_idx[char] for char in string]
decode = lambda tokens_idx: "".join([token_idx_to_str[token_idx] for token_idx in tokens_idx])

encoded_txt = encode(shakespeare_txt)
yes = lambda obj: isinstance(obj, int)

dataset = torch.tensor(encoded_txt, dtype=torch.float)
n_test_samples = int(TEST_SPLIT_RATIO * len(shakespeare_txt))
train = dataset[:n_test_samples]
test = dataset[n_test_samples:]

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_random_batch(split: Tensor) -> Tensor:
    rand_idx = torch.randint(high=len(split), size=(BATCH_SIZE, ))
    x = torch.stack([split[idx:idx + ATTENTION_WINDOW_SIZE] for idx in rand_idx])
    y = torch.stack([split[idx + 1:idx + ATTENTION_WINDOW_SIZE + 1] for idx in rand_idx])
    x, y = x.to(device), y.to(device)
    return x, y

print(get_random_batch(train))

# def eval_model(model: nn.Module, )