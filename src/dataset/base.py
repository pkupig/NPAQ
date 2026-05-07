import numpy as np
from torch.utils.data import Dataset
from abc import abstractmethod


class BaseDataset(Dataset):
    """Base class for all point cloud datasets for metric prediction."""
    def __init__(self, k_neighbors=32, transform=None, noise_std=0.0, cache=False):
        self.k_neighbors = k_neighbors
        self.transform = transform
        self.noise_std = noise_std
        self.cache = cache
        self.data = []

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Return a dict containing:
            - points: (k, in_dims) local patch in LCF
            - metric: (2,2) target metric at query point
            - basis:  (3,3) LCF basis
            - query_idx: int
        """
        pass

    def _add_noise(self, points):
        if self.noise_std > 0:
            points = points + np.random.normal(0, self.noise_std, points.shape)
        return points
