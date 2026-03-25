from .base import BaseDataset
from .synthesis import SURFACES, sample_surface
import numpy as np
import torch

class SyntheticDataset(BaseDataset):
    """Generate synthetic patches on-the-fly from analytic surfaces."""
    def __init__(self, proportions=None, points_per_epoch=10000, k_neighbors=32,
                 noise_std=0.0, transform=None, cache=False,
                 consistency_queries: int = 0,
                 consistency_radius_ratio: float = 0.2):
        super().__init__(
            k_neighbors, transform, noise_std, cache,
            consistency_queries=consistency_queries,
            consistency_radius_ratio=consistency_radius_ratio,
        )
        self.proportions = proportions or {'cylinder':0.4, 'torus':0.4, 'saddle':0.2}
        self.points_per_epoch = points_per_epoch
        self.specs = {name: SURFACES[name] for name in self.proportions}
        # Use a process-local seed so DataLoader workers don't share the same RNG state
        self.rng = np.random.RandomState()

    def __len__(self):
        return self.points_per_epoch

    def __getitem__(self, idx):
        # Choose surface
        names = list(self.proportions.keys())
        probs = list(self.proportions.values())
        surf_name = self.rng.choice(names, p=probs)
        spec = self.specs[surf_name]

        # Generate a small patch of points (more than k to allow neighbor selection)
        n_patch = self.k_neighbors * 5
        points, metrics, dir1, dir2, normals, params = sample_surface(
            spec, n_patch, noise_std=self.noise_std, seed=self.rng.randint(1e6)
        )

        # Random query (avoid boundary points that have fewer real neighbors)
        query_idx = self.rng.randint(n_patch)

        # Compute LCF for the query point using the full patch as context.
        # When analytic normals are available, pass them so:
        #   (a) the LCF z-axis is the exact surface normal (not noisy PCA estimate)
        #   (b) neighbor normals (rotated into LCF) are returned as extra features.
        from .lcf import compute_local_canonical_frame
        from scipy.spatial import KDTree
        tree = KDTree(points)
        local_coords, basis, neighbor_normals = compute_local_canonical_frame(
            points, query_idx, k=self.k_neighbors, normals=normals, tree=tree
        )

        target_metric = metrics[query_idx]
        target_dir1 = dir1[query_idx]
        target_dir2 = dir2[query_idx]

        # Concatenate position + normal → (K, 6) input features.
        # neighbor_normals encodes curvature via normal variation across the patch,
        # making principal curvature direction and magnitude directly observable.
        if neighbor_normals is not None:
            points_feat = np.concatenate([local_coords, neighbor_normals], axis=-1)  # (K, 6)
        else:
            points_feat = local_coords  # (K, 3) fallback

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
                c_local, c_basis, c_normals = compute_local_canonical_frame(
                    points, qi, k=self.k_neighbors, normals=normals, tree=tree
                )
                c_feat = np.concatenate([c_local, c_normals], axis=-1) if c_normals is not None else c_local
                extra_points.append(c_feat.astype(np.float32))
                extra_metrics.append(metrics[qi].astype(np.float32))
                extra_dir1.append(dir1[qi].astype(np.float32))
                extra_dir2.append(dir2[qi].astype(np.float32))
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
