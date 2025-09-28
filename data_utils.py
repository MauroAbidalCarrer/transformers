from typing import Optional

import torch
import tiktoken
from torch.utils.data import (
    TensorDataset,
    DistributedSampler,
    Dataset,
    DataLoader,
)

from config import ENCODING_NAME, TrainingConf

def get_dataset_splits(train_conf: TrainingConf) -> tuple[TensorDataset, TensorDataset]:
    with open("input.txt", 'r', encoding='utf-8') as f:
        shakespeare_txt = f.read()
    tokenizer = tiktoken.get_encoding(ENCODING_NAME)
    encoded_txt = tokenizer.encode(shakespeare_txt)
    dataset = torch.tensor(encoded_txt, dtype=torch.long)
    n_test_samples = int(train_conf.train_test_split_ratio * len(shakespeare_txt))
    train = dataset[:-n_test_samples]
    test = dataset[-n_test_samples:]

def mk_data_loader(dataset: Dataset, train_conf: TrainingConf, n_gpus: int, ddp_rank: Optional[int]=None) -> DataLoader:
    if n_gpus > 1:
        sampler = DistributedSampler(dataset, num_replicas=n_gpus, rank=ddp_rank, shuffle=True)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=train_conf.micro_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True,
        drop_last=True,
    )
