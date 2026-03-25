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
        consistency_queries: int = 0,
        consistency_radius_ratio: float = 0.2,
    ):
        super().__init__(
            k_neighbors, transform, noise_std, cache,
            consistency_queries=consistency_queries,
            consistency_radius_ratio=consistency_radius_ratio,
        )
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
        self.cached_dir1 = []
        self.cached_dir2 = []
        self.cached_trees = []

        for pfile, mfile in zip(self.file_list, self.metric_files):
            # Load points
            points = read_pointcloud(pfile)
            self.cached_points.append(points)
            from scipy.spatial import KDTree
            self.cached_trees.append(KDTree(points))

            # Load normals if present (optional)
            normals = None
            # Try to extract normals from point cloud file if it's .npz
            if pfile.endswith('.npz'):
                data = np.load(pfile)
                if 'normals' in data:
                    normals = data['normals']
            self.cached_normals.append(normals)

            # Load metrics if provided
            if mfile is not None:
                data = np.load(mfile)
                metric = data['metric'] if 'metric' in data else np.eye(2)
                dir1 = data['principal_dir1'] if 'principal_dir1' in data else np.zeros((len(points), 3))
                dir2 = data['principal_dir2'] if 'principal_dir2' in data else np.zeros((len(points), 3))
                self.cached_metrics.append(metric)
                self.cached_dir1.append(dir1)
                self.cached_dir2.append(dir2)
            else:
                # No metrics: use identity and zero directions
                self.cached_metrics.append(np.array([np.eye(2) for _ in range(len(points))]))
                self.cached_dir1.append(np.zeros((len(points), 3)))
                self.cached_dir2.append(np.zeros((len(points), 3)))

    def __len__(self) -> int:
        """Total number of samples = number of models * samples_per_model."""
        return len(self.file_list) * self.samples_per_model

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a training sample (patch + metric) for a random query point."""
        # Determine which model and which sample within model
        model_idx = idx // self.samples_per_model
        # We ignore the within-model sample index because we always randomly pick a query point

        # Get data (from cache or load on-the-fly)
        if self.cache:
            points = self.cached_points[model_idx]
            normals = self.cached_normals[model_idx]
            metric = self.cached_metrics[model_idx]
            dir1 = self.cached_dir1[model_idx]
            dir2 = self.cached_dir2[model_idx]
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
                dir1 = data['principal_dir1'] if 'principal_dir1' in data else np.zeros((len(points), 3))
                dir2 = data['principal_dir2'] if 'principal_dir2' in data else np.zeros((len(points), 3))
            else:
                metric = np.array([np.eye(2) for _ in range(len(points))])
                dir1 = np.zeros((len(points), 3))
                dir2 = np.zeros((len(points), 3))

        # Randomly select a query point
        query_idx = np.random.randint(len(points))

        # Compute LCF for the query point
        local_coords, basis, neighbor_normals, neighbor_indices = compute_local_canonical_frame(
            points, query_idx, k=self.k_neighbors, normals=normals, return_neighbors=True, tree=tree
        )

        # Add noise if specified
        local_coords = self._add_noise(local_coords)

        # Get ground truth at query point
        target_metric = metric[query_idx] if query_idx < len(metric) else np.eye(2)
        target_dir1 = dir1[query_idx] if query_idx < len(dir1) else np.zeros(3)
        target_dir2 = dir2[query_idx] if query_idx < len(dir2) else np.zeros(3)

        # Convert to torch tensors
        if neighbor_normals is not None:
            points_feat = np.concatenate([local_coords, neighbor_normals], axis=-1)
        else:
            points_feat = local_coords

        out = {
            'points': torch.from_numpy(points_feat).float(),
            'metric': torch.from_numpy(target_metric).float(),
            'principal_dir1': torch.from_numpy(target_dir1).float(),
            'principal_dir2': torch.from_numpy(target_dir2).float(),
            'basis': torch.from_numpy(basis).float(),
            'query_idx': torch.tensor(query_idx, dtype=torch.long),
            'model_idx': torch.tensor(model_idx, dtype=torch.long),
        }
        extra_idx = self._select_consistency_query_indices(points, query_idx, tree=tree)
        if len(extra_idx) > 0:
            extra_points = []
            extra_metrics = []
            extra_dir1 = []
            extra_dir2 = []
            extra_basis = []
            extra_pos = []
            for qi in extra_idx.tolist():
                c_local, c_basis, c_normals, _ = compute_local_canonical_frame(
                    points, qi, k=self.k_neighbors, normals=normals, return_neighbors=True, tree=tree
                )
                c_local = self._add_noise(c_local)
                c_feat = np.concatenate([c_local, c_normals], axis=-1) if c_normals is not None else c_local
                extra_points.append(c_feat.astype(np.float32))
                extra_metrics.append(metric[qi].astype(np.float32) if qi < len(metric) else np.eye(2, dtype=np.float32))
                extra_dir1.append(dir1[qi].astype(np.float32) if qi < len(dir1) else np.zeros(3, dtype=np.float32))
                extra_dir2.append(dir2[qi].astype(np.float32) if qi < len(dir2) else np.zeros(3, dtype=np.float32))
                extra_basis.append(c_basis.astype(np.float32))
                rel = (points[qi] - points[query_idx]) @ basis
                extra_pos.append(rel.astype(np.float32))
            out['consistency_points'] = torch.from_numpy(np.stack(extra_points, axis=0)).float()
            out['consistency_metric'] = torch.from_numpy(np.stack(extra_metrics, axis=0)).float()
            out['consistency_principal_dir1'] = torch.from_numpy(np.stack(extra_dir1, axis=0)).float()
            out['consistency_principal_dir2'] = torch.from_numpy(np.stack(extra_dir2, axis=0)).float()
            out['consistency_basis'] = torch.from_numpy(np.stack(extra_basis, axis=0)).float()
            out['consistency_query_pos'] = torch.from_numpy(np.stack(extra_pos, axis=0)).float()
        return out
