from functools import partial

import torch
from torch import nn
from torch.nn import functional

TEST_SPLIT_RATIO = 0.1


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


