import os

import torch
import tiktoken

from config import ENCODING_NAME

class DataLoaderLite:
    def __init__(self, batch_size:int, seq_len:int, process_rank:int, num_processes:int, split:str, master_process: bool):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.process_rank = process_rank if process_rank != -1 else 0
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        # data_root = "edu_fineweb10B"
        # shards = os.listdir(data_root)
        # shards = [s for s in shards if split in s]
        # shards = sorted(shards)
        # shards = [os.path.join(data_root, s) for s in shards]
        # self.shards = shards
        # assert len(shards) > 0, f"no shards found for split {split}"
        # if master_process:
        #     print(f"found {len(shards)} shards for split {split}")
        with open("input.txt", 'r', encoding='utf-8') as f:
            shakespeare_txt = f.read()
        tokenizer = tiktoken.get_encoding(ENCODING_NAME)
        encoded_txt = tokenizer.encode(shakespeare_txt)
        self.tokens = torch.tensor(encoded_txt, dtype=torch.long)
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        # self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.batch_size * self.seq_len * self.process_rank

    def next_batch(self):
        tokens_per_batch = self.batch_size * self.seq_len
        buf = self.tokens[self.current_position : self.current_position + tokens_per_batch + 1]
        x = (buf[:-1]).view(self.batch_size, self.seq_len) # inputs
        y = (buf[1:]).view(self.batch_size, self.seq_len) # targets
        # advance the position in the tensor
        self.current_position += self.batch_size * self.seq_len * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (self.batch_size * self.seq_len * self.num_processes + 1) > len(self.tokens):
            # self.current_shard = (self.current_shard + 1) % len(self.shards)
            # self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.batch_size * self.seq_len * self.process_rank
        return x, y
