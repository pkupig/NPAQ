#!/usr/bin/env python
"""
Training script for the neural metric field predictor.
Supports synthetic, ABC, Stanford, mixed, and custom datasets.
"""

import argparse
import os
import sys
import yaml
import logging
import time
from typing import Dict, Any, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    SyntheticDataset, ABCDataset, CustomDataset,
    StanfordDataset, MixedDataset,
)
from src.models.dgcnn import (
    DGCNN, infer_in_dims_from_checkpoint, infer_predict_confidence_from_checkpoint,
)
from src.models.losses import TotalLoss
from src.geometry.metric_utils import params_to_tensor, eigh2x2


def parse_args():
    parser = argparse.ArgumentParser(description='Train DGCNN for metric prediction')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--reset-epoch', action='store_true')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def setup_logging(log_dir: str):
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
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_dataset(config: Dict[str, Any], split: str):
    data_cfg = config['data']
    dataset_type = data_cfg.get('type', 'synthetic').lower()

    common_kwargs = {
        'k_neighbors': data_cfg['k_neighbors'],
        'noise_std': data_cfg.get('noise_std', 0.0),
        'transform': None,
        'cache': data_cfg.get('cache', False),
    }

    if dataset_type == 'synthetic':
        return SyntheticDataset(
            proportions=data_cfg.get('proportions'),
            points_per_epoch=data_cfg[f'synthetic_{split}_points'],
            **common_kwargs,
        )

    if dataset_type == 'abc':
        return ABCDataset(
            root_dir=data_cfg['root_dir'],
            split=split,
            use_normal=data_cfg.get('use_normal', True),
            **common_kwargs,
        )

    if dataset_type == 'stanford':
        return StanfordDataset(
            root_dir=data_cfg['root_dir'],
            split=split,
            use_normal=data_cfg.get('use_normal', True),
            **common_kwargs,
        )

    if dataset_type == 'custom':
        return CustomDataset(
            file_list=data_cfg[f'{split}_files'],
            metric_files=data_cfg.get(f'{split}_metrics'),
            **common_kwargs,
        )

    if dataset_type == 'mixed':
        syn_points_key = f'synthetic_{split}_points'
        synthetic_ds = SyntheticDataset(
            proportions=data_cfg.get('proportions'),
            points_per_epoch=data_cfg.get(syn_points_key, 50000 if split == 'train' else 5000),
            **common_kwargs,
        )
        real_ds = StanfordDataset(
            root_dir=data_cfg['real_root_dir'],
            split=split,
            use_normal=data_cfg.get('use_normal', True),
            **common_kwargs,
        )
        real_ratio = float(data_cfg.get('real_ratio', 0.3))
        virtual_len = data_cfg.get(f'mixed_{split}_len',
                                   len(synthetic_ds) + len(real_ds))
        return MixedDataset(
            synthetic_dataset=synthetic_ds,
            real_dataset=real_ds,
            real_ratio=real_ratio,
            virtual_len=virtual_len,
        )

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def collate_fn(batch):
    elem = batch[0]
    out = {}
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            out[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            out[key] = [b[key] for b in batch]
    return out


def _forward_and_decode(model, batch, device):
    """
    Forward pass with tangent-plane rotation augmentation.

    The Duff LCF builds (e1, e2) purely from the surface normal, unrelated
    to principal curvature direction. Without rotation augmentation, DGCNN
    would have to memorise every Duff-frame orientation relative to the
    principal axes (this caused the historical 0.40 training floor).

    Consistency requirement: ALL quantities must be expressed in the SAME
    (rotated) coordinate system. The 2D synthetic metric is stored in the
    OLD frame, so it must be explicitly rotated by R2^T M R2.
    """
    points    = batch['points'].to(device)
    metric_gt = batch['metric'].to(device)
    basis     = batch['basis'].to(device)

    aug_R2 = None
    if model.training:
        B = points.shape[0]
        alpha = torch.rand(B, device=device) * (2.0 * torch.pi)
        cos_a = torch.cos(alpha)
        sin_a = torch.sin(alpha)

        x_old = points[..., 0].clone()
        y_old = points[..., 1].clone()
        points = points.clone()
        points[..., 0] =  cos_a.unsqueeze(1) * x_old + sin_a.unsqueeze(1) * y_old
        points[..., 1] = -sin_a.unsqueeze(1) * x_old + cos_a.unsqueeze(1) * y_old

        if points.shape[-1] >= 6:
            nx_old = points[..., 3].clone()
            ny_old = points[..., 4].clone()
            points[..., 3] =  cos_a.unsqueeze(1) * nx_old + sin_a.unsqueeze(1) * ny_old
            points[..., 4] = -sin_a.unsqueeze(1) * nx_old + cos_a.unsqueeze(1) * ny_old

        e1_old = basis[:, :, 0].clone()
        e2_old = basis[:, :, 1].clone()
        basis = basis.clone()
        basis[:, :, 0] =  cos_a.unsqueeze(1) * e1_old + sin_a.unsqueeze(1) * e2_old
        basis[:, :, 1] = -sin_a.unsqueeze(1) * e1_old + cos_a.unsqueeze(1) * e2_old

        aug_R2 = torch.stack([
            torch.stack([ cos_a, -sin_a], dim=-1),
            torch.stack([ sin_a,  cos_a], dim=-1),
        ], dim=-2)

    if aug_R2 is not None and metric_gt.shape[-1] == 2:
        metric_gt = torch.matmul(aug_R2.transpose(-2, -1), torch.matmul(metric_gt, aug_R2))

    x = points.transpose(2, 1)
    out = model(x)

    M_pred_lcf = params_to_tensor(out[:, 0], out[:, 1], out[:, 2], out[:, 3])

    R = basis[:, :, :2]
    if metric_gt.shape[-1] == 3:
        M_gt_lcf = torch.matmul(R.transpose(-2, -1), torch.matmul(metric_gt, R))
    else:
        M_gt_lcf = metric_gt

    with torch.no_grad():
        eigvals_gt, _ = eigh2x2(M_gt_lcf)
        lam_min = eigvals_gt[..., 0].clamp(min=0.0)
        lam_max = eigvals_gt[..., 1].clamp(min=0.0)
        aniso = (lam_max - lam_min) / (lam_max + lam_min + 1e-8)

    return {
        'M_pred': M_pred_lcf,
        'M_gt':   M_gt_lcf,
        'conf_pred': model.decode_confidence(out),
        'conf_gt':   aniso,
    }


def _run_loss(loss_fn, fwd):
    return loss_fn(
        M_pred=fwd['M_pred'], M_gt=fwd['M_gt'],
        confidence_pred=fwd['conf_pred'],
        confidence_gt=fwd['conf_gt'],
    )


def train_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    totals = {'loss': 0.0, 'metric': 0.0, 'topo': 0.0, 'conf': 0.0}
    num_batches = len(loader)

    pbar = tqdm(loader, desc='Training', leave=False)
    for batch in pbar:
        fwd = _forward_and_decode(model, batch, device)
        loss, terms = _run_loss(loss_fn, fwd)

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

        totals['loss']   += loss.item()
        totals['metric'] += terms['metric'].item()
        totals['topo']   += terms['topo'].item()
        totals['conf']   += terms['conf'].item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return {k: v / num_batches for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    totals = {'loss': 0.0, 'metric': 0.0, 'topo': 0.0, 'conf': 0.0}
    num_batches = len(loader)

    for batch in tqdm(loader, desc='Validating', leave=False):
        fwd = _forward_and_decode(model, batch, device)
        loss, terms = _run_loss(loss_fn, fwd)

        totals['loss']   += loss.item()
        totals['metric'] += terms['metric'].item()
        totals['topo']   += terms['topo'].item()
        totals['conf']   += terms['conf'].item()

    return {k: v / num_batches for k, v in totals.items()}


def main():
    args = parse_args()
    config = load_config(args.config)

    log_dir = os.path.join(args.logdir, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir)
    logger.info(f"Config: {config}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Creating datasets...")
    train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logger.info("Building model...")
    model_cfg = config['model']
    if args.resume and os.path.isfile(args.resume):
        _peek = torch.load(args.resume, map_location='cpu')
        _state = _peek.get('model_state_dict', _peek)
        in_dims = infer_in_dims_from_checkpoint(_state)
        predict_confidence = infer_predict_confidence_from_checkpoint(_state)
        del _peek
    else:
        in_dims = model_cfg.get('in_dims', 3)
        predict_confidence = bool(model_cfg.get('predict_confidence', True))

    model = DGCNN(
        k=model_cfg['k'],
        emb_dims=model_cfg['emb_dims'],
        dropout=model_cfg['dropout'],
        in_dims=in_dims,
        predict_confidence=predict_confidence,
        max_log_half=model_cfg.get('max_log_half', 1.5),
    ).to(device)
    logger.info(
        f"  DGCNN in_dims={in_dims}, emb_dims={model_cfg['emb_dims']}, "
        f"predict_confidence={predict_confidence}"
    )

    loss_cfg = config['loss']
    if float(loss_cfg.get('lambda_j', 0.0)) > 0.0:
        raise NotImplementedError(
            "loss.lambda_j > 0 requires unrolled parametrization/PD inputs "
            "(V_final, quads_t, M_targets, ref_pinv). Current train.py only "
            "runs metric, anisotropy, saliency, and topology-barrier losses."
        )
    loss_fn = TotalLoss(
        lambda_aniso=loss_cfg.get('lambda_aniso', 0.1),
        lambda_conf=loss_cfg.get('lambda_conf', 0.1),
        lambda_topo=loss_cfg.get('lambda_topo', 0.01),
        lambda_j=loss_cfg.get('lambda_j', 0.0),
        topo_det_eps=loss_cfg.get('topo_det_eps', 1e-3),
        topo_cond_max=loss_cfg.get('topo_cond_max', 50.0),
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.0),
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
            gamma=config['training']['lr_decay_gamma'],
        )

    use_amp = config['training'].get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    writer = SummaryWriter(log_dir)

    train_cfg = config['training']
    select_by = str(train_cfg.get('select_by', 'val_metric')).lower()
    if select_by not in ('val_metric', 'val_loss'):
        logger.warning(f"Unknown training.select_by={select_by!r}; fallback to 'val_metric'.")
        select_by = 'val_metric'
    early_stop_patience = int(train_cfg.get('early_stop_patience', 0))
    early_stop_min_delta = float(train_cfg.get('early_stop_min_delta', 1e-4))
    no_improve_epochs = 0

    start_epoch = 0
    best_val_loss = float('inf')
    best_val_metric = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            for pg in optimizer.param_groups:
                pg['lr'] = config['training']['lr']
                pg['weight_decay'] = config['training'].get('weight_decay', 0.0)

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
            logger.info(f"Resumed from epoch {start_epoch-1}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")

    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")

        train_stats = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_stats = validate(model, val_loader, loss_fn, device)

        sched_score = val_stats['loss'] if select_by == 'val_loss' else val_stats['metric']

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(sched_score)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        writer.add_scalar('Loss/train', train_stats['loss'], epoch)
        writer.add_scalar('Loss/val',   val_stats['loss'],   epoch)
        writer.add_scalar('Metric/train', train_stats['metric'], epoch)
        writer.add_scalar('Metric/val',   val_stats['metric'],   epoch)
        writer.add_scalar('Confidence/train', train_stats['conf'], epoch)
        writer.add_scalar('Confidence/val',   val_stats['conf'],   epoch)
        if loss_cfg.get('lambda_topo', 0.0) > 0.0:
            writer.add_scalar('TopologyProxy/train', train_stats['topo'], epoch)
            writer.add_scalar('TopologyProxy/val',   val_stats['topo'],   epoch)
        writer.add_scalar('LR', current_lr, epoch)

        logger.info(
            f"Train Loss: {train_stats['loss']:.6f} (M: {train_stats['metric']:.6f}, "
            f"C: {train_stats['conf']:.6f}) | "
            f"Val Loss: {val_stats['loss']:.6f} (M: {val_stats['metric']:.6f}, "
            f"C: {val_stats['conf']:.6f})"
        )

        if select_by == 'val_loss':
            current_score, best_score = val_stats['loss'], best_val_loss
        else:
            current_score, best_score = val_stats['metric'], best_val_metric

        is_best = current_score < (best_score - early_stop_min_delta)
        if is_best:
            if select_by == 'val_loss':
                best_val_loss = current_score
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
            'config': config,
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
