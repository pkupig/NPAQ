#!/usr/bin/env python
"""
Training script for the neural metric field predictor.
Supports synthetic, ABC, ModelNet, and custom datasets.
"""

import argparse
import os
import sys
import yaml
import logging
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    SyntheticDataset, ABCDataset, ModelNetDataset, CustomDataset
)
from src.models.dgcnn import (
    DGCNN, infer_in_dims_from_checkpoint, infer_predict_singularity_from_checkpoint,
    infer_predict_confidence_from_checkpoint,
)
from src.models.losses import CombinedLoss, TotalEndToEndLoss
from src.geometry.metric_utils import params_to_tensor, eigh2x2


def parse_args():
    parser = argparse.ArgumentParser(description='Train DGCNN for metric prediction')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--reset-epoch', action='store_true',
                        help='Reset epoch counter to 0 when resuming (for fine-tuning)')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def setup_logging(log_dir: str):
    """Configure logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_dataset(config: Dict[str, Any], split: str):
    """Instantiate dataset based on configuration."""
    data_cfg = config['data']
    dataset_type = data_cfg.get('type', 'synthetic').lower()

    common_kwargs = {
        'k_neighbors': data_cfg['k_neighbors'],
        'noise_std': data_cfg.get('noise_std', 0.0),
        'transform': None,  # can add later
        'cache': data_cfg.get('cache', False),
        'consistency_queries': data_cfg.get('consistency_queries', 0),
        'consistency_radius_ratio': data_cfg.get('consistency_radius_ratio', 0.2),
    }

    if dataset_type == 'synthetic':
        kwargs = {
            'proportions': data_cfg.get('proportions'),
            'points_per_epoch': data_cfg[f'synthetic_{split}_points'],
            **common_kwargs
        }
        return SyntheticDataset(**kwargs)

    elif dataset_type == 'abc':
        kwargs = {
            'root_dir': data_cfg['root_dir'],
            'split': split,
            'use_normal': data_cfg.get('use_normal', True),
            **common_kwargs
        }
        return ABCDataset(**kwargs)

    elif dataset_type == 'modelnet':
        kwargs = {
            'root_dir': data_cfg['root_dir'],
            'category': data_cfg.get('category', 'all'),
            **common_kwargs
        }
        return ModelNetDataset(**kwargs)

    elif dataset_type == 'custom':
        kwargs = {
            'file_list': data_cfg[f'{split}_files'],
            'metric_files': data_cfg.get(f'{split}_metrics'),
            **common_kwargs
        }
        return CustomDataset(**kwargs)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def collate_fn(batch):
    """Default collate for dict of tensors."""
    elem = batch[0]
    out = {}
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            out[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            out[key] = [b[key] for b in batch]
    return out


def _decode_from_tensors(
    model,
    points: torch.Tensor,
    metric_gt: torch.Tensor,
    basis: torch.Tensor,
    dir1_gt: Optional[torch.Tensor],
    dir2_gt: Optional[torch.Tensor],
):
    """Decode metric/direction/confidence from already-device tensors."""
    x = points.transpose(2, 1)
    out = model(x)

    s1 = out[:, 0]; s2 = out[:, 1]; c = out[:, 2]; s = out[:, 3]
    M_pred_lcf = params_to_tensor(s1, s2, c, s)

    R = basis[:, :, :2]
    if metric_gt.shape[-1] == 3:
        M_gt_lcf = torch.matmul(R.transpose(-2, -1), torch.matmul(metric_gt, R))
    else:
        M_gt_lcf = metric_gt

    with torch.no_grad():
        eigvals_gt, _ = eigh2x2(M_gt_lcf)
        lam_min = eigvals_gt[..., 0].clamp(min=0.0)
        lam_max = eigvals_gt[..., 1].clamp(min=0.0)
        aniso_weights = (lam_max - lam_min) / (lam_max + lam_min + 1e-8)

    dir1_pred_local = dir2_pred_local = None
    dir1_gt_local = dir2_gt_local = None
    if dir1_gt is not None:
        dir1_gt_local = torch.matmul(R.transpose(-2, -1), dir1_gt.unsqueeze(-1)).squeeze(-1)
        dir2_gt_local = torch.matmul(R.transpose(-2, -1), dir2_gt.unsqueeze(-1)).squeeze(-1)
        dir1_gt_local = F.normalize(dir1_gt_local, p=2, dim=-1)
        dir2_gt_local = F.normalize(dir2_gt_local, p=2, dim=-1)

        norm_cs = torch.sqrt(c ** 2 + s ** 2 + 1e-8)
        c_norm = c / norm_cs
        s_norm = s / norm_cs
        cos_theta = torch.sqrt((1.0 + c_norm) * 0.5 + 1e-8)
        sin_theta = torch.sign(s_norm) * torch.sqrt((1.0 - c_norm) * 0.5 + 1e-8)
        dir1_pred_local = torch.stack([cos_theta, sin_theta], dim=-1)
        dir2_pred_local = None

    conf_gt = aniso_weights
    conf_pred = model.decode_confidence(out)

    return dict(
        M_pred_lcf=M_pred_lcf, M_gt_lcf=M_gt_lcf,
        dir1_pred=dir1_pred_local, dir1_gt=dir1_gt_local,
        dir2_pred=dir2_pred_local, dir2_gt=dir2_gt_local,
        aniso_weights=aniso_weights,
        confidence_pred=conf_pred,
        confidence_gt=conf_gt,
    )


def _forward_and_decode(model, batch, device):
    """
    Shared forward pass for train and validate.

    Returns a dict with M_pred_lcf, M_gt_lcf, dir1/2 pred/gt, confidence.
    """
    points    = batch['points'].to(device)    # (B, K, in_dims)
    metric_gt = batch['metric'].to(device)    # (B, 2, 2) or (B, 2, 3) synthetic
    basis     = batch['basis'].to(device)     # (B, 3, 3)
    dir1_gt   = batch.get('principal_dir1')
    dir2_gt   = batch.get('principal_dir2')
    if dir1_gt is not None:
        dir1_gt = dir1_gt.to(device)
        dir2_gt = dir2_gt.to(device)

    # ── Tangent-plane rotation augmentation (training only) ───────────────
    # The Duff LCF frame builds (e1, e2) purely from the surface normal —
    # unrelated to principal curvature direction.  DGCNN is not 2D-rotation-
    # equivariant in the LCF tangent plane, so without this augmentation the
    # network must memorise a different pattern for every possible Duff-frame
    # orientation relative to the principal axes.  Randomly rotating the frame
    # forces the network to learn rotation-invariant features instead.
    #
    # Consistency requirement: ALL quantities must be expressed in the SAME
    # (rotated) coordinate system:
    #   - points (x,y) and normals (nx,ny): rotate by A = [[c,s],[-s,c]]
    #   - 3D basis columns (e1,e2): R_new = R_old @ R2  (R2=[[c,-s],[s,c]])
    #   - dir1_gt_local: computed from R_new → transforms automatically ✓
    #   - M_gt_lcf (3D metric): computed from R_new → transforms automatically ✓
    #   - M_gt_lcf (2D metric, synthetic): stored in OLD frame → must apply
    #       M_new = R2^T @ M_old @ R2  explicitly  ← the bug that caused
    #       train_loss >> val_loss and the 0.40 training floor.
    aug_R2 = None   # (B, 2, 2) rotation matrix; set below during training
    if model.training:
        B = points.shape[0]
        alpha = torch.rand(B, device=device) * (2.0 * torch.pi)
        cos_a = torch.cos(alpha)   # (B,)
        sin_a = torch.sin(alpha)   # (B,)

        # Rotate LCF (x, y) coordinates of each neighbour
        x_old = points[..., 0].clone()   # (B, K)
        y_old = points[..., 1].clone()   # (B, K)
        points = points.clone()
        points[..., 0] =  cos_a.unsqueeze(1) * x_old + sin_a.unsqueeze(1) * y_old
        points[..., 1] = -sin_a.unsqueeze(1) * x_old + cos_a.unsqueeze(1) * y_old

        # Rotate LCF normal (nx, ny) if in_dims == 6
        if points.shape[-1] >= 6:
            nx_old = points[..., 3].clone()
            ny_old = points[..., 4].clone()
            points[..., 3] =  cos_a.unsqueeze(1) * nx_old + sin_a.unsqueeze(1) * ny_old
            points[..., 4] = -sin_a.unsqueeze(1) * nx_old + cos_a.unsqueeze(1) * ny_old

        # Rotate tangent columns of basis: new_R = old_R @ R2
        # R2 = [[cos_a, -sin_a], [sin_a, cos_a]]  (standard 2-D CCW rotation)
        # basis shape: (B, 3, 3); tangent cols are [:, :, 0:2]
        e1_old = basis[:, :, 0].clone()   # (B, 3)
        e2_old = basis[:, :, 1].clone()   # (B, 3)
        basis = basis.clone()
        basis[:, :, 0] =  cos_a.unsqueeze(1) * e1_old + sin_a.unsqueeze(1) * e2_old
        basis[:, :, 1] = -sin_a.unsqueeze(1) * e1_old + cos_a.unsqueeze(1) * e2_old

        # Build R2 for explicit 2-D metric rotation (used below for synthetic data)
        aug_R2 = torch.stack([
            torch.stack([ cos_a, -sin_a], dim=-1),
            torch.stack([ sin_a,  cos_a], dim=-1),
        ], dim=-2)   # (B, 2, 2)
    # ── end augmentation ──────────────────────────────────────────────────

    # Apply explicit 2D metric rotation for synthetic 2D metrics after augmentation.
    if aug_R2 is not None and metric_gt.shape[-1] == 2:
        metric_gt = torch.matmul(aug_R2.transpose(-2, -1), torch.matmul(metric_gt, aug_R2))

    out = _decode_from_tensors(model, points, metric_gt, basis, dir1_gt, dir2_gt)

    if 'consistency_points' in batch:
        c_points = batch['consistency_points'].to(device)        # (B,Q,K,D)
        c_metric = batch['consistency_metric'].to(device)        # (B,Q,2,2)
        c_basis = batch['consistency_basis'].to(device)          # (B,Q,3,3)
        c_dir1 = batch.get('consistency_principal_dir1')
        c_dir2 = batch.get('consistency_principal_dir2')
        if c_dir1 is not None:
            c_dir1 = c_dir1.to(device)
            c_dir2 = c_dir2.to(device)
        B, Q = c_points.shape[:2]
        c_out = _decode_from_tensors(
            model,
            c_points.reshape(B * Q, *c_points.shape[2:]),
            c_metric.reshape(B * Q, *c_metric.shape[2:]),
            c_basis.reshape(B * Q, *c_basis.shape[2:]),
            None if c_dir1 is None else c_dir1.reshape(B * Q, *c_dir1.shape[2:]),
            None if c_dir2 is None else c_dir2.reshape(B * Q, *c_dir2.shape[2:]),
        )
        out['consistency'] = {
            'dir1_pred': None if c_out['dir1_pred'] is None else c_out['dir1_pred'].reshape(B, Q, -1),
            'dir1_gt': None if c_out['dir1_gt'] is None else c_out['dir1_gt'].reshape(B, Q, -1),
            'dir2_pred': None if c_out['dir2_pred'] is None else c_out['dir2_pred'].reshape(B, Q, -1),
            'dir2_gt': None if c_out['dir2_gt'] is None else c_out['dir2_gt'].reshape(B, Q, -1),
            'basis': c_basis,
            'anchor_basis': basis,
            'query_pos': batch['consistency_query_pos'].to(device),
            'aniso_weights': c_out['aniso_weights'].reshape(B, Q),
            'M_pred_lcf': c_out['M_pred_lcf'].reshape(B, Q, 2, 2),
            'M_gt_lcf': c_out['M_gt_lcf'].reshape(B, Q, 2, 2),
        }

    return out


def train_epoch(model, loader, optimizer, loss_fn, device, config, scaler=None):
    model.train()
    total_loss = total_metric = total_dir = total_sing = total_topo = total_conf = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc='Training', leave=False)
    for batch in pbar:
        fwd = _forward_and_decode(model, batch, device)

        if isinstance(loss_fn, TotalEndToEndLoss):
            loss, terms = loss_fn(
                M_pred=fwd['M_pred_lcf'], M_gt=fwd['M_gt_lcf'],
                sing_logits=None, sing_labels=None, euler_char=0.0,
                dir1_pred=fwd['dir1_pred'], dir1_gt=fwd['dir1_gt'],
                dir2_pred=fwd['dir2_pred'], dir2_gt=fwd['dir2_gt'],
                aniso_weights=fwd.get('aniso_weights'),
                confidence_pred=fwd.get('confidence_pred'),
                confidence_gt=fwd.get('confidence_gt'),
                consistency_dir1_pred=(fwd.get('consistency') or {}).get('dir1_pred'),
                consistency_dir2_pred=(fwd.get('consistency') or {}).get('dir2_pred'),
                consistency_basis=(fwd.get('consistency') or {}).get('basis'),
                consistency_anchor_basis=(fwd.get('consistency') or {}).get('anchor_basis'),
                consistency_query_pos=(fwd.get('consistency') or {}).get('query_pos'),
                consistency_aniso_weights=(fwd.get('consistency') or {}).get('aniso_weights'),
                consistency_M_pred=(fwd.get('consistency') or {}).get('M_pred_lcf'),
                consistency_M_gt=(fwd.get('consistency') or {}).get('M_gt_lcf'),
                consistency_dir1_gt=(fwd.get('consistency') or {}).get('dir1_gt'),
                consistency_dir2_gt=(fwd.get('consistency') or {}).get('dir2_gt'),
            )
            loss_metric_v = terms['metric'].item()
            loss_dir_v    = terms['dir'].item()
            loss_sing_v   = terms['sing'].item()
            topo_base = terms.get('topo', torch.tensor(0.0, device=device))
            topo_field = terms.get('topo_field', torch.tensor(0.0, device=device))
            topo_ring = terms.get('topo_ring', torch.tensor(0.0, device=device))
            topo_charge = terms.get('topo_charge', torch.tensor(0.0, device=device))
            topo_turn = terms.get('topo_turn', torch.tensor(0.0, device=device))
            topo_cons_metric = terms.get('topo_cons_metric', torch.tensor(0.0, device=device))
            topo_cons_dir = terms.get('topo_cons_dir', torch.tensor(0.0, device=device))
            loss_topo_v = (
                topo_base
                + float(getattr(loss_fn, 'topo_field_weight', 0.0)) * topo_field
                + float(getattr(loss_fn, 'topo_ring_weight', 0.0)) * topo_ring
                + float(getattr(loss_fn, 'topo_charge_weight', 0.0)) * topo_charge
                + float(getattr(loss_fn, 'topo_turn_weight', 0.0)) * topo_turn
                + float(getattr(loss_fn, 'topo_cons_metric_weight', 0.0)) * topo_cons_metric
                + float(getattr(loss_fn, 'topo_cons_dir_weight', 0.0)) * topo_cons_dir
            ).item()
            loss_conf_v   = terms.get('conf', torch.tensor(0.0, device=device)).item()
        else:
            # Legacy CombinedLoss
            loss, lm, ld = loss_fn(
                fwd['M_pred_lcf'], fwd['M_gt_lcf'],
                fwd['dir1_pred'], fwd['dir1_gt'],
                fwd['dir2_pred'], fwd['dir2_gt'],
                aniso_weights=fwd.get('aniso_weights'),
            )
            loss_metric_v = lm.item(); loss_dir_v = ld.item()
            loss_sing_v = 0.0; loss_topo_v = 0.0; loss_conf_v = 0.0

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss   += loss.item()
        total_metric += loss_metric_v
        total_dir    += loss_dir_v
        total_sing   += loss_sing_v
        total_topo   += loss_topo_v
        total_conf   += loss_conf_v
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return (total_loss / num_batches, total_metric / num_batches,
            total_dir / num_batches, total_topo / num_batches,
            total_conf / num_batches)


@torch.no_grad()
def validate(model, loader, loss_fn, device, config):
    model.eval()
    total_loss = total_metric = total_dir = total_topo = total_conf = 0.0
    num_batches = len(loader)

    for batch in tqdm(loader, desc='Validating', leave=False):
        fwd = _forward_and_decode(model, batch, device)

        if isinstance(loss_fn, TotalEndToEndLoss):
            loss, terms = loss_fn(
                M_pred=fwd['M_pred_lcf'], M_gt=fwd['M_gt_lcf'],
                sing_logits=None, sing_labels=None, euler_char=0.0,
                dir1_pred=fwd['dir1_pred'], dir1_gt=fwd['dir1_gt'],
                dir2_pred=fwd['dir2_pred'], dir2_gt=fwd['dir2_gt'],
                aniso_weights=fwd.get('aniso_weights'),
                confidence_pred=fwd.get('confidence_pred'),
                confidence_gt=fwd.get('confidence_gt'),
                consistency_dir1_pred=(fwd.get('consistency') or {}).get('dir1_pred'),
                consistency_dir2_pred=(fwd.get('consistency') or {}).get('dir2_pred'),
                consistency_basis=(fwd.get('consistency') or {}).get('basis'),
                consistency_anchor_basis=(fwd.get('consistency') or {}).get('anchor_basis'),
                consistency_query_pos=(fwd.get('consistency') or {}).get('query_pos'),
                consistency_aniso_weights=(fwd.get('consistency') or {}).get('aniso_weights'),
                consistency_M_pred=(fwd.get('consistency') or {}).get('M_pred_lcf'),
                consistency_M_gt=(fwd.get('consistency') or {}).get('M_gt_lcf'),
                consistency_dir1_gt=(fwd.get('consistency') or {}).get('dir1_gt'),
                consistency_dir2_gt=(fwd.get('consistency') or {}).get('dir2_gt'),
            )
            lm = terms['metric'].item(); ld = terms['dir'].item()
            topo_base = terms.get('topo', torch.tensor(0.0, device=device))
            topo_field = terms.get('topo_field', torch.tensor(0.0, device=device))
            topo_ring = terms.get('topo_ring', torch.tensor(0.0, device=device))
            topo_charge = terms.get('topo_charge', torch.tensor(0.0, device=device))
            topo_turn = terms.get('topo_turn', torch.tensor(0.0, device=device))
            topo_cons_metric = terms.get('topo_cons_metric', torch.tensor(0.0, device=device))
            topo_cons_dir = terms.get('topo_cons_dir', torch.tensor(0.0, device=device))
            lt = (
                topo_base
                + float(getattr(loss_fn, 'topo_field_weight', 0.0)) * topo_field
                + float(getattr(loss_fn, 'topo_ring_weight', 0.0)) * topo_ring
                + float(getattr(loss_fn, 'topo_charge_weight', 0.0)) * topo_charge
                + float(getattr(loss_fn, 'topo_turn_weight', 0.0)) * topo_turn
                + float(getattr(loss_fn, 'topo_cons_metric_weight', 0.0)) * topo_cons_metric
                + float(getattr(loss_fn, 'topo_cons_dir_weight', 0.0)) * topo_cons_dir
            ).item()
            lc = terms.get('conf', torch.tensor(0.0, device=device)).item()
        else:
            loss, lm_t, ld_t = loss_fn(
                fwd['M_pred_lcf'], fwd['M_gt_lcf'],
                fwd['dir1_pred'], fwd['dir1_gt'],
                fwd['dir2_pred'], fwd['dir2_gt'],
                aniso_weights=fwd.get('aniso_weights'),
            )
            lm = lm_t.item(); ld = ld_t.item()
            lt = 0.0; lc = 0.0

        total_loss   += loss.item()
        total_metric += lm
        total_dir    += ld
        total_topo   += lt
        total_conf   += lc

    return (total_loss / num_batches, total_metric / num_batches,
            total_dir / num_batches, total_topo / num_batches,
            total_conf / num_batches)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup logging and device
    log_dir = os.path.join(args.logdir, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir)
    logger.info(f"Config: {config}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Build model
    # If resuming, pre-peek at the checkpoint to infer in_dims from the actual
    # conv1 weight shape.  This is robust to checkpoints saved before in_dims
    # was added to train.yaml (the config field may be absent or stale).
    logger.info("Building model...")
    model_cfg = config['model']
    if args.resume and os.path.isfile(args.resume):
        _peek = torch.load(args.resume, map_location='cpu')
        _state = _peek.get('model_state_dict', _peek)
        in_dims = infer_in_dims_from_checkpoint(
            _state)
        predict_singularity = infer_predict_singularity_from_checkpoint(_state)
        predict_confidence = infer_predict_confidence_from_checkpoint(_state)
        del _peek
    else:
        in_dims = model_cfg.get('in_dims', 3)
        predict_singularity = bool(model_cfg.get('predict_singularity', False))
        predict_confidence = bool(model_cfg.get('predict_confidence', True))
    model = DGCNN(
        k=model_cfg['k'],
        emb_dims=model_cfg['emb_dims'],
        dropout=model_cfg['dropout'],
        in_dims=in_dims,
        predict_singularity=predict_singularity,
        predict_confidence=predict_confidence,
        max_log_half=model_cfg.get('max_log_half', 1.5),
    ).to(device)
    logger.info(
        f"  DGCNN in_dims={in_dims}, emb_dims={model_cfg['emb_dims']}, "
        f"predict_confidence={predict_confidence}, predict_singularity={predict_singularity}"
    )

    # Loss function — use TotalEndToEndLoss if singularity/Jacobian weights given
    loss_cfg = config['loss']
    if any(k in loss_cfg for k in ('lambda_sing', 'lambda_ph', 'lambda_j')):
        if float(loss_cfg.get('lambda_sing', 0.0)) != 0.0 or float(loss_cfg.get('lambda_ph', 0.0)) != 0.0:
            logger.warning(
                "Singularity supervision is disabled (no labels / gauge freedom). "
                "Forcing lambda_sing=lambda_ph=0 during training."
            )
        loss_fn = TotalEndToEndLoss(
            lambda_dir=loss_cfg.get('lambda_dir',  0.1),
            dir_stop_threshold=loss_cfg.get('dir_stop_threshold', 0.0),
            lambda_sing=0.0,
            lambda_ph=0.0,
            lambda_j=loss_cfg.get('lambda_j',      0.1),
            lambda_conf=loss_cfg.get('lambda_conf', 0.1),
            lambda_topo=loss_cfg.get('lambda_topo', 0.0),
            topo_det_eps=loss_cfg.get('topo_det_eps', 1e-3),
            topo_cond_max=loss_cfg.get('topo_cond_max', 50.0),
            topo_entropy_weight=loss_cfg.get('topo_entropy_weight', 0.1),
            topo_field_weight=loss_cfg.get('topo_field_weight', 0.0),
            topo_ring_weight=loss_cfg.get('topo_ring_weight', 0.0),
            topo_charge_weight=loss_cfg.get('topo_charge_weight', 0.0),
            topo_turn_weight=loss_cfg.get('topo_turn_weight', 0.0),
            topo_cons_metric_weight=loss_cfg.get('topo_cons_metric_weight', 0.0),
            topo_cons_dir_weight=loss_cfg.get('topo_cons_dir_weight', 0.0),
            topo_consistency_sigma=loss_cfg.get('topo_consistency_sigma', 0.35),
            topo_ring_min_queries=loss_cfg.get('topo_ring_min_queries', 6),
            topo_warmup_epochs=loss_cfg.get('topo_warmup_epochs', 0),
            topo_warmup_start_epoch=loss_cfg.get('topo_warmup_start_epoch', 0),
        )
    else:
        loss_fn = CombinedLoss(
            lambda_dir=loss_cfg.get('lambda_dir', 0.1),
            dir_stop_threshold=loss_cfg.get('dir_stop_threshold', 0.0),
        )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    sched_type = config['training'].get('lr_scheduler', 'step')
    if sched_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training'].get('lr_decay_gamma', 0.5),
            patience=config['training'].get('lr_patience', 30),
            min_lr=config['training'].get('lr_min', 1e-6),
            threshold=1e-4,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['lr_decay_step'],
            gamma=config['training']['lr_decay_gamma']
        )

    # Mixed precision
    use_amp = config['training'].get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # Model-selection / early-stop policy:
    # default to metric-first selection (loss is secondary diagnostics only).
    train_cfg = config['training']
    select_by = str(train_cfg.get('select_by', 'val_metric')).lower()
    if select_by not in ('val_metric', 'val_loss', 'val_dir'):
        logger.warning(f"Unknown training.select_by={select_by!r}; fallback to 'val_metric'.")
        select_by = 'val_metric'
    early_stop_patience = int(train_cfg.get('early_stop_patience', 0))
    early_stop_min_delta = float(train_cfg.get('early_stop_min_delta', 1e-4))
    no_improve_epochs = 0

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_metric = float('inf')
    best_val_dir = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # IMPORTANT: when resuming for finetuning, users often change LR /
            # weight decay / scheduler settings in YAML.  The optimizer state in
            # checkpoint would silently override those values unless we force
            # apply the current config here.
            for pg in optimizer.param_groups:
                pg['lr'] = config['training']['lr']
                pg['weight_decay'] = config['training'].get('weight_decay', 0.0)

            # By default, do NOT restore scheduler state from checkpoint so the
            # current YAML scheduler hyper-parameters take effect immediately.
            # If strict continuation is desired, set:
            #   training.resume_scheduler_state: true
            if config['training'].get('resume_scheduler_state', False):
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except (KeyError, ValueError):
                    logger.warning("Scheduler state incompatible — resetting scheduler.")
            else:
                logger.info("Skipping scheduler state restore; using scheduler from current config.")
            start_epoch = 0 if args.reset_epoch else checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
            best_val_metric = checkpoint.get('best_val_metric', best_val_metric)
            best_val_dir = checkpoint.get('best_val_dir', best_val_dir)
            logger.info(f"Resumed from epoch {start_epoch-1}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        if isinstance(loss_fn, TotalEndToEndLoss):
            loss_fn.set_epoch(epoch)
            logger.info(
                f"  Topology ramp: x{loss_fn.current_topo_multiplier():.3f} "
                f"(lambda_topo={float(loss_cfg.get('lambda_topo', 0.0)):.4f})"
            )

        train_loss, train_metric, train_dir, train_topo, train_conf = train_epoch(
            model, train_loader, optimizer, loss_fn, device, config, scaler
        )
        val_loss, val_metric, val_dir, val_topo, val_conf = validate(
            model, val_loader, loss_fn, device, config
        )

        # Determine monitored score FIRST so ReduceLROnPlateau uses the same
        # signal as early stopping (previously the plateau scheduler always
        # used val_loss while early stopping used val_metric; when directional
        # loss was improving but metric had plateaued, LR stayed high and early
        # stopping still triggered — defeating the purpose of both mechanisms).
        if select_by == 'val_loss':
            sched_score = val_loss
        elif select_by == 'val_dir':
            sched_score = val_dir
        else:
            sched_score = val_metric   # default: val_metric

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(sched_score)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metric/train', train_metric, epoch)
        writer.add_scalar('Metric/val', val_metric, epoch)
        writer.add_scalar('Direction/train', train_dir, epoch)
        writer.add_scalar('Direction/val', val_dir, epoch)
        writer.add_scalar('Confidence/train', train_conf, epoch)
        writer.add_scalar('Confidence/val', val_conf, epoch)
        lambda_topo = float(loss_cfg.get('lambda_topo', 0.0))
        log_topo_proxy = bool(config['training'].get('log_topo_proxy', False))
        if lambda_topo > 0.0 and log_topo_proxy:
            writer.add_scalar('TopologyProxy/train', train_topo, epoch)
            writer.add_scalar('TopologyProxy/val', val_topo, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        if lambda_topo > 0.0 and log_topo_proxy:
            logger.info(
                f"Train Loss: {train_loss:.6f} (M: {train_metric:.6f}, D: {train_dir:.6f}, "
                f"C: {train_conf:.6f}, T: {train_topo:.6f}) | "
                f"Val Loss: {val_loss:.6f} (M: {val_metric:.6f}, D: {val_dir:.6f}, "
                f"C: {val_conf:.6f}, T: {val_topo:.6f})"
            )
        else:
            logger.info(
                f"Train Loss: {train_loss:.6f} (M: {train_metric:.6f}, D: {train_dir:.6f}, "
                f"C: {val_conf:.6f}) | "
                f"Val Loss: {val_loss:.6f} (M: {val_metric:.6f}, D: {val_dir:.6f}, "
                f"C: {val_conf:.6f})"
            )

        # Save checkpoint (metric-first by default)
        if select_by == 'val_loss':
            current_score = val_loss
            best_score = best_val_loss
        elif select_by == 'val_dir':
            current_score = val_dir
            best_score = best_val_dir
        else:
            current_score = val_metric
            best_score = best_val_metric

        is_best = current_score < (best_score - early_stop_min_delta)
        if is_best:
            if select_by == 'val_loss':
                best_val_loss = current_score
            elif select_by == 'val_dir':
                best_val_dir = current_score
            else:
                best_val_metric = current_score
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_metric': best_val_metric,
            'best_val_dir': best_val_dir,
            'config': config
        }
        torch.save(checkpoint, os.path.join(log_dir, 'latest.pth'))

        if is_best:
            torch.save(checkpoint, os.path.join(log_dir, 'best.pth'))
            logger.info(f"Saved best model by {select_by}: {current_score:.6f}")

        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            logger.info(
                f"Early stop: {select_by} did not improve by >{early_stop_min_delta:g} "
                f"for {no_improve_epochs} epochs."
            )
            break

    logger.info("Training finished.")
    writer.close()


if __name__ == '__main__':
    main()
