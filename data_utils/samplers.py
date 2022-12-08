# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""batch samplers that work with either random or sequential data samplers"""
import math
import os
import sys
from typing import Optional

import torch
from torch.utils import data
import numpy as np

from SwissArmyTransformer import mpu
from utils import print_rank_0


class NewRandomSampler(data.sampler.Sampler):
    def __init__(self, data_source, name="None"):
        super(NewRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.name = name
        print_rank_0(f"{self.name} | Per-Rank Sampler Number of data: {len(data_source)}.")

        self.generator = torch.Generator()
        self.epoch = 0
        self.generator.manual_seed(self.epoch)
        print_rank_0("\n\n")
        print_rank_0(f"{self.name} | Set seed for generator with epoch-seed: {self.epoch}")
        print_rank_0("\n\n")
        self.data_list = torch.randperm(len(self.data_source), generator=self.generator).tolist()

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):
        while True:
            for data in self.data_list:
                yield data
            self.epoch += 1
            self.generator.manual_seed(self.epoch)
            print_rank_0("\n\n")
            print_rank_0(f"{self.name} | Set seed for generator with epoch-seed: {self.epoch}")
            print_rank_0("\n\n")
            self.data_list = torch.randperm(len(self.data_source), generator=self.generator).tolist()

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSequentialSampler(data.sampler.Sampler):
    def __init__(self, num_samples, train_iters, batch_size, rank=-1, world_size=2):
        super().__init__(num_samples)
        if rank == -1:
            rank = 0
            world_size = 1
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.start_iter = 0
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.batch_bias = [i * (num_samples // batch_size) for i in range(batch_size)]

    def __iter__(self):
        for idx in range(self.start_iter, self.train_iters * 10):
            batch = [(idx + bias) % self.num_samples for bias in self.batch_bias]
            tbatch = self._batch(batch)
            yield tbatch

    def __len__(self):
        return self.train_iters

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        start = self.rank*self.batch_size//self.world_size
        end = (self.rank+1)*self.batch_size//self.world_size
        return batch[start:end]


class DistributedBatchSampler(data.sampler.BatchSampler):
    """
    similar to normal implementation of distributed sampler, except implementation is at the
    batch sampler level, instead of just the sampler level. This allows wrapping of arbitrary
    data samplers (sequential, random, WeightedRandomSampler, etc.) with this batch sampler.
    """
    def __init__(self, sampler, batch_size, drop_last, rank=-1, world_size=2, wrap_last=False, gradient_accumulation_steps=None):
        super(DistributedBatchSampler, self).__init__(sampler, batch_size, drop_last)
        if rank == -1:
            assert False, 'should not be here'
        self.rank = rank
        self.world_size = world_size
        self.sampler.wrap_around = 0
        self.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0
        self.effective_batch_size = batch_size if gradient_accumulation_steps is None else batch_size * gradient_accumulation_steps

    def __iter__(self):
        batch = []
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            if len(batch) == self.batch_size:
                tbatch = self._batch(batch)
                if i >= self.start_iter * self.effective_batch_size:
                    yield tbatch
                    self.start_iter = 0
                i += len(batch)
                batch = []
        batch_len = len(batch)
        if batch_len > 0 and not self.drop_last:
            if self.wrap_last:
                self.sampler.wrap_around -= (self.batch_size)
                self.wrap_around += (len(batch))
                self.wrap_around %= self.batch_size
            last_batch = self._batch(batch)
            if last_batch:
                yield last_batch
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around%self.batch_size:
                continue
            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        batch_size = len(batch)
        micro_batch_size = (batch_size - 1) // self.world_size + 1
        start = self.rank * micro_batch_size
        end = (self.rank + 1) * micro_batch_size
        return batch[start:end]
