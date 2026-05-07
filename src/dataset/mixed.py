"""
MixedDataset — combines a synthetic dataset and a real dataset with a
configurable mixing ratio.

At each __getitem__ call the dataset samples from the real set with
probability `real_ratio` and from the synthetic set otherwise.
This is equivalent to a weighted ConcatDataset but without needing
WeightedRandomSampler in the DataLoader.

Typical use (train_mixed.yaml):
    type: mixed
    real_ratio: 0.3          # 30 % real (Stanford), 70 % synthetic
    synthetic_train_points: 50000
    real_root_dir: data/stanford_preprocessed
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class MixedDataset(Dataset):
    """
    Args:
        synthetic_dataset: SyntheticDataset (or any Dataset)
        real_dataset:      StanfordDataset / ABCDataset (or any Dataset)
        real_ratio:        Fraction of samples drawn from the real set [0, 1].
                           Default 0.3 → 70 % synthetic, 30 % real.
        virtual_len:       Length reported by __len__. Defaults to
                           len(synthetic) + len(real).
        seed:              Optional RNG seed for reproducible split choices
                           (not required for training; each worker seeds itself).
    """

    def __init__(
        self,
        synthetic_dataset: Dataset,
        real_dataset: Dataset,
        real_ratio: float = 0.3,
        virtual_len: int = None,
        seed: int = None,
    ):
        self.synthetic = synthetic_dataset
        self.real = real_dataset
        self.real_ratio = float(real_ratio)
        self._len = virtual_len if virtual_len is not None else (
            len(synthetic_dataset) + len(real_dataset)
        )
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        # Each worker has its own process-local RandomState so there is no
        # inter-worker RNG collision even with num_workers > 0.
        if self._rng.random() < self.real_ratio:
            real_idx = self._rng.randint(len(self.real))
            return self.real[real_idx]
        else:
            syn_idx = self._rng.randint(len(self.synthetic))
            return self.synthetic[syn_idx]

    def __repr__(self):
        return (
            f"MixedDataset(len={self._len}, real_ratio={self.real_ratio:.0%}, "
            f"synthetic={self.synthetic}, real={self.real})"
        )
