#!/usr/bin/env python
"""
Preprocess Stanford / custom real meshes into .npz patches for mixed training.

Reuses the same process_file / _build_2d_metric logic as abc_preprocess.py so
the output .npz format is identical and can be loaded by ABCDataset / StanfordDataset.

Output per mesh (saved as <stem>.npz):
    points         (N, 3)   float32
    normals        (N, 3)   float32
    metric         (N, 2, 2) float32  — 2D LCF metric tensor
    principal_dir1 (N, 3)   float32
    principal_dir2 (N, 3)   float32

Usage:
    python scripts/preprocess_stanford.py \\
        --input  /mnt/e/Users/DELL/Desktop/stanford_data \\
        --output data/stanford_preprocessed \\
        --n_points 4096 --k_fit 20

    # Then train with mixed config:
    python scripts/train.py --config configs/train_mixed.yaml
"""

import argparse
import os
import sys
import json
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the proven preprocessing logic from abc_preprocess.py
from scripts.abc_preprocess import process_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess Stanford (and other real) meshes for NPAQ mixed training'
    )
    parser.add_argument('--input',     required=True,
                        help='Directory containing .ply/.obj/.off/.stl mesh files')
    parser.add_argument('--output',    required=True,
                        help='Output directory for .npz files')
    parser.add_argument('--n_points',  type=int, default=4096,
                        help='Surface samples per mesh (default: 4096)')
    parser.add_argument('--k_fit',     type=int, default=20,
                        help='Neighbours for curvature fitting (default: 20)')
    parser.add_argument('--k_lcf',     type=int, default=32,
                        help='Neighbours for LCF construction (default: 32)')
    parser.add_argument('--rho',       type=float, default=1.0,
                        help='Metric scale factor ρ (default: 1.0)')
    parser.add_argument('--epsilon',   type=float, default=0.05,
                        help='Regularisation ε for flat regions (default: 0.05)')
    parser.add_argument('--seed',      type=int, default=0,
                        help='Base random seed (default: 0)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Fraction of processed files reserved for validation (default: 0.2)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    extensions = {'.obj', '.stl', '.off', '.ply'}
    mesh_files = sorted([
        str(p) for p in Path(args.input).rglob('*')
        if p.suffix.lower() in extensions
    ])

    if not mesh_files:
        print(f"No mesh files found in {args.input}")
        return

    print(f"Found {len(mesh_files)} mesh files in {args.input}")
    print(f"Output → {args.output}")

    ok_files = []
    failed_files = []

    for i, fpath in enumerate(tqdm(mesh_files, desc='Preprocessing')):
        stem = Path(fpath).stem
        out_path = os.path.join(args.output, stem + '.npz')

        if os.path.exists(out_path):
            ok_files.append(out_path)
            continue

        try:
            result = process_file(
                fpath,
                n_points=args.n_points,
                k_fit=args.k_fit,
                k_lcf=args.k_lcf,
                rho=args.rho,
                epsilon=args.epsilon,
                seed=args.seed + i,
            )
        except Exception:
            traceback.print_exc()
            result = None

        if result is None:
            failed_files.append(fpath)
            tqdm.write(f"  FAILED: {fpath}")
            continue

        np.savez_compressed(out_path, **result)
        ok_files.append(out_path)

    ok_files = sorted(ok_files)

    # Write index files compatible with ABCDataset / StanfordDataset.
    index_path = os.path.join(args.output, 'stanford_index.json')
    with open(index_path, 'w') as f:
        json.dump({'files': ok_files}, f, indent=2)

    if len(ok_files) > 1:
        val_ratio = min(0.9, max(0.0, float(args.val_ratio)))
        val_count = max(1, int(round(val_ratio * len(ok_files)))) if val_ratio > 0.0 else 0
        split_at = len(ok_files) - val_count
        train_files = ok_files[:split_at]
        val_files = ok_files[split_at:] if val_count > 0 else ok_files[:0]
    else:
        train_files = ok_files
        val_files = ok_files

    with open(os.path.join(args.output, 'train_index.json'), 'w') as f:
        json.dump({'files': train_files}, f, indent=2)
    with open(os.path.join(args.output, 'val_index.json'), 'w') as f:
        json.dump({'files': val_files}, f, indent=2)

    print(f"\nDone: {len(ok_files)} succeeded, {len(failed_files)} failed.")
    print(f"Index: {index_path}")
    print(f"Split: train={len(train_files)}, val={len(val_files)}")
    if failed_files:
        print("Failed files:")
        for f in failed_files:
            print(f"  {f}")


if __name__ == '__main__':
    main()
