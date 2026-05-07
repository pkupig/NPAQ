import os
import torch
import numpy as np
import glob
import json
from .base import BaseDataset
from .lcf import compute_local_canonical_frame
from scipy.spatial import KDTree

class ABCDataset(BaseDataset):
    """Load preprocessed ABC dataset (points, normals, metrics)."""
    def __init__(self, root_dir, split='train', k_neighbors=32, noise_std=0.0,
                 transform=None, cache=False, use_normal=True):
        super().__init__(k_neighbors, transform, noise_std, cache)
        self.root_dir = root_dir
        self.split = split
        self.use_normal = use_normal
        self.files = self._get_files()
        if len(self.files) == 0:
            raise RuntimeError(
                f"ABCDataset: no .npz files found for split='{split}' "
                f"under {root_dir!r}.\n"
                f"Download and preprocess ABC data first:\n"
                f"  python scripts/download_abc.py "
                f"--chunks 0 1 --output data/abc --preprocess"
            )
        if cache:
            self._cache_data()
        # Single-slot LRU for the cache=False path. Each worker keeps its own
        # most-recently-loaded (file_idx, data, tree) so that the 200 patches
        # per file in __len__ amortise to one .npz load + one KDTree build per
        # file rather than per call. Without this, each __getitem__ re-decodes
        # the .npz and rebuilds the KDTree on potentially 100k points — the
        # dominant DataLoader cost on Stanford/ABC.
        self._lru_idx = -1
        self._lru_data = None
        self._lru_tree = None

    def _get_files(self):
        index_file = os.path.join(self.root_dir, f'{self.split}_index.json')
        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)
            return index['files']
        else:
            # Assume .npz files in split directory
            pattern = os.path.join(self.root_dir, self.split, '*.npz')
            return sorted(glob.glob(pattern))

    def _cache_data(self):
        self.cached_data = []
        self.cached_trees = []
        for fpath in self.files:
            data = np.load(fpath)
            self.cached_data.append(data)
            self.cached_trees.append(KDTree(data['points']))

    def __len__(self):
        return len(self.files) * 200  # patches per epoch (200×files)

    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        if self.cache:
            data = self.cached_data[file_idx]
            tree = self.cached_trees[file_idx]
        else:
            if file_idx == self._lru_idx and self._lru_data is not None:
                data = self._lru_data
                tree = self._lru_tree
            else:
                data = dict(np.load(self.files[file_idx]))   # materialise into RAM
                tree = KDTree(data['points'])
                self._lru_idx = file_idx
                self._lru_data = data
                self._lru_tree = tree

        points = data['points']
        normals = data.get('normals') if self.use_normal else None
        metrics = data.get('metric')   # (N,2,2)

        query_idx = np.random.randint(len(points))
        local_coords, basis, neighbor_normals, _ = compute_local_canonical_frame(
            points, query_idx, k=self.k_neighbors, normals=normals,
            return_neighbors=True, tree=tree,
        )

        target_metric = metrics[query_idx] if metrics is not None else np.eye(2)

        local_coords = self._add_noise(local_coords)
        if neighbor_normals is not None:
            points_feat = np.concatenate([local_coords, neighbor_normals], axis=-1)
        else:
            points_feat = local_coords

        return {
            'points': torch.from_numpy(points_feat).float(),
            'metric': torch.from_numpy(target_metric).float(),
            'basis':  torch.from_numpy(basis).float(),
            'query_idx': query_idx,
        }
