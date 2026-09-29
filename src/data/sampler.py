"""Distributed batch sampling for dataset mixtures."""
import math
import os
import torch
from torch.utils.data import Sampler


class BatchMixtureSampler(Sampler):
    """
    Multi-node safe dataset mixture sampler.
    Example: datasets = [ds1, ds2], ratio = [4, 1]
    """

    def __init__(self, dataset_lens, mixture_ratios, num_samples_per_epoch, batch_size, seed=42):
        self.dataset_lens = dataset_lens
        self.weights = torch.tensor(mixture_ratios, dtype=torch.double)
        self.weights /= self.weights.sum()
        self.num_samples = num_samples_per_epoch
        self.batch_size = batch_size
        self.seed = seed
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.num_samples_rank = math.ceil(self.num_samples / self.world_size)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        all_dataset_ids = torch.multinomial(self.weights, self.num_samples_rank * self.world_size, replacement=True, generator=g)
        dataset_ids = all_dataset_ids[self.rank::self.world_size]
        dataset_ids_tensor = dataset_ids if isinstance(dataset_ids, torch.Tensor) else torch.tensor(dataset_ids)
        dataset_lens_tensor = torch.tensor(self.dataset_lens)
        lens_for_ids = dataset_lens_tensor[dataset_ids_tensor]
        rand_indices = torch.floor(torch.rand(len(dataset_ids_tensor), generator=g) * lens_for_ids).long()
        indices = list(zip(dataset_ids_tensor.tolist(), rand_indices.tolist()))
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        return iter(batches)

    def __len__(self):
        return math.ceil(self.num_samples_rank / self.batch_size)
