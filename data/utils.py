import collections
import itertools
import math
import numpy as np
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler

class CollateListData:
    def __init__(self, dim=0, keys=['image']):
        self.dim = dim
        self.keys = keys
        assert len(keys)>0
    
    def __call__(self, batch: collections.abc.Sequence):
        for i, item in enumerate(batch):
            # print(f"{i} = {item['image'].shape=} >> {item['image'].keys=}")
            data = item[0]
            for k in self.keys:
                data[k] = torch.stack([ix[k] for ix in item], dim=self.dim)
            # data["patch_location"] = torch.stack([ix["patch_location"] for ix in item], dim=0)
            batch[i] = data
        return default_collate(batch)



###############################################
## Batch sampler for Semi-supervised Learning
###############################################

class TwoStreamSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, primary_batch_size: int = 1, secondary_batch_size: int = 1, shuffle = True, num_samples: Optional[int] = None):
        self.primary_indices = np.array(primary_indices)
        self.secondary_indices = np.array(secondary_indices)
        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = secondary_batch_size
        self.batch_size = primary_batch_size + secondary_batch_size
        self.shuffle = shuffle
        self.num_samples = len(primary_indices) if num_samples is None else num_samples

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        n = len(self.primary_indices)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        def primary_iter(shuffle=True):
            if shuffle:
                for _ in range(self.num_samples // n):
                    yield from self.primary_indices[torch.randperm(n, generator=generator)].tolist()
                yield from self.primary_indices[torch.randperm(n, generator=generator)].tolist()[:self.num_samples % n]
            else:
                for _ in range(self.num_samples // n):
                    yield from self.primary_indices.tolist()
                yield from self.primary_indices.tolist()[:self.num_samples % n]
        
        def secondary_iter(shuffle=True):
            def infinite_shuffles():
                while True:
                    if shuffle:
                        yield self.secondary_indices[torch.randperm(len(self.secondary_indices), generator=generator)]
                    else:
                        yield self.secondary_indices
            return itertools.chain.from_iterable(infinite_shuffles())
        
        sample = itertools.chain(*(
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter(self.shuffle), self.primary_batch_size),
                   grouper(secondary_iter(self.shuffle), self.secondary_batch_size))
        ))        
        
        yield from sample

    def __len__(self):
        k = math.floor(self.num_samples / float(self.primary_batch_size))
        return k * (self.primary_batch_size + self.secondary_batch_size)


def iterate_once(iterable, shuffle=True):
    if shuffle:
        return np.random.permutation(iterable)
    else:
        return iterable


def iterate_eternally(indices, shuffle=True):
    def infinite_shuffles():
        while True:
            if shuffle:
                yield np.random.permutation(indices)
            else:
                yield indices
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class TwoStreamMiniBatchSampler(Sampler):
    """Iterate mini-batch from TwoStreamBatchSampler
    Minibatch_size is the true "batch size" of sampler
    e.g.: if, primary_indices ABC ... primary_batch_size=1 / secondary indices abc ... secondary_batch_size=3 / minibatch=2
        generates -> Aa bc Bd ef ...
    """

    def __init__(self, primary_indices, secondary_indices, batch_size=1, secondary_batch_size=1, minibatch_size=1, shuffle=True, drop_last=False, len_dataset: int=None):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mini = minibatch_size
        self.drop_last = drop_last
        self.len_dataset = len_dataset

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        #primary_iter = iterate_once(self.primary_indices, self.shuffle)
        primary_iter = iterate_eternally(self.primary_indices, self.shuffle)
        secondary_iter = iterate_eternally(self.secondary_indices, self.shuffle) 
        sample = (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )
        
        return (x for x in grouper(itertools.chain.from_iterable(sample), self.mini))
        

    def __len__(self):
        if self.len_dataset is None:
            if self.drop_last:
                return ((len(self.primary_indices) // self.primary_batch_size) * self.batch_size) // self.mini
            else:
                return (((len(self.primary_indices) + self.primary_batch_size - 1) // self.primary_batch_size) * self.batch_size + self.mini - 1) // self.mini
        else:
            if self.drop_last:
                return ((self.len_dataset // self.primary_batch_size) * self.batch_size) // self.mini
            else:
                return (((self.len_dataset + self.primary_batch_size - 1) // self.primary_batch_size) * self.batch_size + self.mini - 1) // self.mini


