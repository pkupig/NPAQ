from .base import BaseDataset
from .synthesis import SURFACES
import numpy as np
import torch

class SyntheticDataset(BaseDataset):
    """Generate synthetic patches on-the-fly from analytic surfaces."""
    def __init__(self, proportions=None, points_per_epoch=10000, k_neighbors=32,
                 noise_std=0.0, transform=None, cache=False):
        super().__init__(k_neighbors, transform, noise_std, cache)
        self.proportions = proportions or {'cylinder':0.4, 'torus':0.4, 'saddle':0.2}
        self.points_per_epoch = points_per_epoch
        self.specs = {name: SURFACES[name] for name in self.proportions}
        # Use a process-local seed so DataLoader workers don't share the same RNG state
        self.rng = np.random.RandomState()

    def __len__(self):
        return self.points_per_epoch

    def __getitem__(self, idx):
        from .lcf import compute_local_canonical_frame
        from scipy.spatial import KDTree

        # Choose surface
        names = list(self.proportions.keys())
        probs = list(self.proportions.values())
        surf_name = self.rng.choice(names, p=probs)
        spec = self.specs[surf_name]

        u_lo, u_hi = spec.u_range
        v_lo, v_hi = spec.v_range

        # ── Step 1: pick query params from interior ──────────────────────────
        # Margin only on non-periodic dimensions (periodic dims have no real boundary).
        u_margin = 0.0 if spec.periodic_u else spec.query_margin_frac * (u_hi - u_lo)
        v_margin = 0.0 if spec.periodic_v else spec.query_margin_frac * (v_hi - v_lo)
        u0 = self.rng.uniform(u_lo + u_margin, u_hi - u_margin)
        v0 = self.rng.uniform(v_lo + v_margin, v_hi - v_margin)

        # ── Step 2: sample context points from LOCAL parameter window ────────
        # Drawing context from [u0 ± patch_half_u] × [v0 ± patch_half_v] ensures
        # all sampled points are geodesically close to the query, so Euclidean
        # k-NN neighbours are not contaminated by "back of the cylinder" points
        # that are 3D-close but geodesically distant.
        n_context = self.k_neighbors * 5 - 1
        du = self.rng.uniform(-spec.patch_half_u, spec.patch_half_u, n_context)
        dv = self.rng.uniform(-spec.patch_half_v, spec.patch_half_v, n_context)
        u_ctx = u0 + du
        v_ctx = v0 + dv

        # Wrap periodic dimensions; clamp non-periodic to domain bounds.
        if spec.periodic_u:
            period_u = u_hi - u_lo
            u_ctx = (u_ctx - u_lo) % period_u + u_lo
        else:
            u_ctx = np.clip(u_ctx, u_lo, u_hi)
        if spec.periodic_v:
            period_v = v_hi - v_lo
            v_ctx = (v_ctx - v_lo) % period_v + v_lo
        else:
            v_ctx = np.clip(v_ctx, v_lo, v_hi)

        # Prepend the exact query point as index 0.
        u_all = np.concatenate([[u0], u_ctx])
        v_all = np.concatenate([[v0], v_ctx])
        query_idx = 0

        # ── Step 3: evaluate analytic surface properties ─────────────────────
        # Per-sample random params (e.g. monge coefficients, bump amplitude).
        params = spec.param_sampler(self.rng) if spec.param_sampler is not None else {}
        points = spec.func(u_all, v_all, **params)                                  # (N, 3)
        normals = spec.normals_fn(u_all, v_all, **params) if spec.normals_fn is not None else None
        if self.noise_std > 0:
            points = points + self.rng.normal(0, self.noise_std, points.shape)
        metrics_all = spec.metric_tensor(u_all, v_all, **params)                    # (N, 3, 3)

        # ── Step 4: LCF + 2D metric projection ───────────────────────────────
        tree = KDTree(points)
        local_coords, basis, neighbor_normals = compute_local_canonical_frame(
            points, query_idx, k=self.k_neighbors, normals=normals, tree=tree
        )

        R = basis[:, :2]                              # (3, 2)
        target_metric = R.T @ metrics_all[0] @ R     # (2, 2)

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
