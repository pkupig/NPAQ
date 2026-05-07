"""
Custom dataset loader for user-provided point clouds with optional precomputed metrics.
Supports loading from various file formats and caching.
"""

import os
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any
from .base import BaseDataset
from .lcf import compute_local_canonical_frame
from ..utils.mesh_io import read_pointcloud


class CustomDataset(BaseDataset):
    """
    Dataset for custom point cloud files with optional precomputed metrics.
    
    Expected file structure:
        - Point cloud files can be .xyz, .ply, .npz (with 'points' key), etc.
        - Metric files (if provided) should be .npz files containing:
            'metric': (N,2,2) metric tensors at each point
            'principal_dir1', 'principal_dir2': (N,3) principal directions (optional)
            'normals': (N,3) normals (optional)
    
    Args:
        file_list: List of paths to point cloud files.
        metric_files: List of paths to corresponding metric files (same length as file_list).
                      If None, metric will be set to identity (not recommended for training).
        k_neighbors: Number of nearest neighbors for LCF construction.
        noise_std: Standard deviation of Gaussian noise to add to points.
        transform: Optional transform to apply to the patch.
        cache: If True, load all point clouds and metrics into memory.
        samples_per_model: Number of query points to sample per model (affects __len__).
    """
    def __init__(
        self,
        file_list: List[str],
        metric_files: Optional[List[str]] = None,
        k_neighbors: int = 32,
        noise_std: float = 0.0,
        transform=None,
        cache: bool = False,
        samples_per_model: int = 1000,
    ):
        super().__init__(k_neighbors, transform, noise_std, cache)
        self.file_list = file_list
        self.samples_per_model = samples_per_model

        # Validate metric files
        if metric_files is not None:
            if len(metric_files) != len(file_list):
                raise ValueError("metric_files must have same length as file_list")
            self.metric_files = metric_files
        else:
            self.metric_files = [None] * len(file_list)
            # Warn if no metrics provided (likely inference only)
            import warnings
            warnings.warn("No metric files provided. Metric tensors will be set to identity.")

        # Cache storage
        if cache:
            self._cache_data()

    def _cache_data(self):
        """Load all point clouds and metrics into memory."""
        self.cached_points = []
        self.cached_normals = []
        self.cached_metrics = []
        self.cached_trees = []

        for pfile, mfile in zip(self.file_list, self.metric_files):
            points = read_pointcloud(pfile)
            self.cached_points.append(points)
            from scipy.spatial import KDTree
            self.cached_trees.append(KDTree(points))

            normals = None
            if pfile.endswith('.npz'):
                data = np.load(pfile)
                if 'normals' in data:
                    normals = data['normals']
            self.cached_normals.append(normals)

            if mfile is not None:
                data = np.load(mfile)
                metric = data['metric'] if 'metric' in data else np.eye(2)
            else:
                metric = np.array([np.eye(2) for _ in range(len(points))])
            self.cached_metrics.append(metric)

    def __len__(self) -> int:
        """Total number of samples = number of models * samples_per_model."""
        return len(self.file_list) * self.samples_per_model

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a training sample (patch + metric) for a random query point."""
        # Determine which model and which sample within model
        model_idx = idx // self.samples_per_model
        # We ignore the within-model sample index because we always randomly pick a query point

        if self.cache:
            points = self.cached_points[model_idx]
            normals = self.cached_normals[model_idx]
            metric = self.cached_metrics[model_idx]
            tree = self.cached_trees[model_idx]
        else:
            pfile = self.file_list[model_idx]
            mfile = self.metric_files[model_idx]
            points = read_pointcloud(pfile)
            from scipy.spatial import KDTree
            tree = KDTree(points)
            normals = None
            if pfile.endswith('.npz'):
                data = np.load(pfile)
                if 'normals' in data:
                    normals = data['normals']

            if mfile is not None:
                data = np.load(mfile)
                metric = data['metric'] if 'metric' in data else np.eye(2)
            else:
                metric = np.array([np.eye(2) for _ in range(len(points))])

        query_idx = np.random.randint(len(points))

        local_coords, basis, neighbor_normals, _ = compute_local_canonical_frame(
            points, query_idx, k=self.k_neighbors, normals=normals,
            return_neighbors=True, tree=tree,
        )
        local_coords = self._add_noise(local_coords)

        target_metric = metric[query_idx] if query_idx < len(metric) else np.eye(2)

        if neighbor_normals is not None:
            points_feat = np.concatenate([local_coords, neighbor_normals], axis=-1)
        else:
            points_feat = local_coords

        return {
            'points': torch.from_numpy(points_feat).float(),
            'metric': torch.from_numpy(target_metric).float(),
            'basis':  torch.from_numpy(basis).float(),
            'query_idx': torch.tensor(query_idx, dtype=torch.long),
            'model_idx': torch.tensor(model_idx, dtype=torch.long),
        }
