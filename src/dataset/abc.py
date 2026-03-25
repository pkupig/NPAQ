import os
import torch
import numpy as np
import glob
import json
from .base import BaseDataset
from .lcf import compute_local_canonical_frame
from sklearn.neighbors import KDTree

class ABCDataset(BaseDataset):
    """Load preprocessed ABC dataset (points, normals, metrics)."""
    def __init__(self, root_dir, split='train', k_neighbors=32, noise_std=0.0,
                 transform=None, cache=False, use_normal=True,
                 consistency_queries: int = 0,
                 consistency_radius_ratio: float = 0.2):
        super().__init__(
            k_neighbors, transform, noise_std, cache,
            consistency_queries=consistency_queries,
            consistency_radius_ratio=consistency_radius_ratio,
        )
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
        return len(self.files) * 1000  # approximate number of patches

    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        if self.cache:
            data = self.cached_data[file_idx]
        else:
            data = np.load(self.files[file_idx])

        points = data['points']
        normals = data.get('normals') if self.use_normal else None
        metrics = data.get('metric')   # (N,2,2)
        dir1 = data.get('principal_dir1')
        dir2 = data.get('principal_dir2')
        tree = self.cached_trees[file_idx] if self.cache else KDTree(points)

        # Random query point
        query_idx = np.random.randint(len(points))
        local_coords, basis, neighbor_normals, neighbor_indices = compute_local_canonical_frame(
            points, query_idx, k=self.k_neighbors, normals=normals, return_neighbors=True, tree=tree
        )

        target_metric = metrics[query_idx] if metrics is not None else np.eye(2)
        target_dir1 = dir1[query_idx] if dir1 is not None else np.zeros(3)
        target_dir2 = dir2[query_idx] if dir2 is not None else np.zeros(3)

        local_coords = self._add_noise(local_coords)
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
            'query_idx': query_idx,
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
                if c_normals is not None:
                    c_feat = np.concatenate([c_local, c_normals], axis=-1)
                else:
                    c_feat = c_local
                extra_points.append(c_feat.astype(np.float32))
                extra_metrics.append(metrics[qi].astype(np.float32) if metrics is not None else np.eye(2, dtype=np.float32))
                extra_dir1.append(dir1[qi].astype(np.float32) if dir1 is not None else np.zeros(3, dtype=np.float32))
                extra_dir2.append(dir2[qi].astype(np.float32) if dir2 is not None else np.zeros(3, dtype=np.float32))
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
