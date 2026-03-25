import torch
from torch.utils.data import Dataset
import numpy as np
from abc import abstractmethod

class BaseDataset(Dataset):
    """Base class for all point cloud datasets for metric prediction."""
    def __init__(
        self,
        k_neighbors=32,
        transform=None,
        noise_std=0.0,
        cache=False,
        consistency_queries: int = 0,
        consistency_radius_ratio: float = 0.2,
    ):
        self.k_neighbors = k_neighbors
        self.transform = transform
        self.noise_std = noise_std
        self.cache = cache
        self.consistency_queries = int(consistency_queries)
        self.consistency_radius_ratio = float(consistency_radius_ratio)
        self.data = []  # to be filled by subclass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Return a dict containing:
            - points: (k,3) local patch in LCF
            - metric: (2,2) target metric at query point
            - principal_dir1, principal_dir2: (3,) optional
            - basis: (3,3) LCF basis
            - query_idx: int
        """
        pass

    def _add_noise(self, points):
        if self.noise_std > 0:
            points = points + np.random.normal(0, self.noise_std, points.shape)
        return points

    def _select_consistency_query_indices(self, points, query_idx, *, tree=None):
        """
        Pick nearby extra query indices for local field-consistency training.

        Returns indices excluding the anchor query point.
        """
        if self.consistency_queries <= 0:
            return np.empty((0,), dtype=np.int64)

        from scipy.spatial import KDTree

        pts = np.asarray(points, dtype=np.float64)
        if len(pts) <= 1:
            return np.empty((0,), dtype=np.int64)
        if tree is None:
            tree = KDTree(pts)

        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        radius = max(1e-8, self.consistency_radius_ratio * max(diag, 1e-8))
        idxs = tree.query_ball_point(pts[int(query_idx)], r=radius)
        idxs = [int(i) for i in idxs if int(i) != int(query_idx)]

        if len(idxs) < self.consistency_queries:
            k = min(len(pts), self.consistency_queries + 1)
            _, nn = tree.query(pts[int(query_idx)], k=k)
            nn = np.atleast_1d(nn).tolist()
            idxs = [int(i) for i in nn if int(i) != int(query_idx)]

        idxs = idxs[:self.consistency_queries]
        return np.asarray(idxs, dtype=np.int64)
