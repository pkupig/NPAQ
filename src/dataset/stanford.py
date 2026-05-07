"""
Stanford dataset loader.
Identical to ABCDataset in I/O format; exists as a named class so that
log output and type-dispatch in train.py are self-explanatory.
"""

from .abc import ABCDataset


class StanfordDataset(ABCDataset):
    """
    Load preprocessed Stanford (or other real-mesh) .npz files.

    Expected .npz keys (produced by scripts/preprocess_stanford.py):
        points         (N, 3)   float32
        normals        (N, 3)   float32
        metric         (N, 2, 2) float32
        principal_dir1 (N, 3)   float32
        principal_dir2 (N, 3)   float32

    The index file is expected at <root_dir>/stanford_index.json
    (written automatically by preprocess_stanford.py).
    If not found, falls back to globbing <root_dir>/**/*.npz.

    Preprocess first:
        python scripts/preprocess_stanford.py \\
            --input  /mnt/e/Users/DELL/Desktop/stanford_data \\
            --output data/stanford_preprocessed
    """

    def _get_files(self):
        import os, json

        # Prefer explicit split indexes when preprocessing wrote them.
        split_index = os.path.join(self.root_dir, f'{self.split}_index.json')
        if os.path.exists(split_index):
            with open(split_index) as f:
                return json.load(f)['files']

        # Backward-compatible single index.  Older preprocessing wrote one
        # stanford_index.json; split it deterministically so validation is not
        # the same real meshes as training.
        index_file = os.path.join(self.root_dir, 'stanford_index.json')
        if os.path.exists(index_file):
            with open(index_file) as f:
                files = sorted(json.load(f)['files'])
            if len(files) <= 1:
                return files
            val_count = max(1, int(round(0.2 * len(files))))
            split_at = max(1, len(files) - val_count)
            if self.split == 'train':
                return files[:split_at]
            if self.split in ('val', 'valid', 'validation', 'test'):
                return files[split_at:]
            return files

        # Fall back to parent logic (train_index.json or glob)
        return super()._get_files()

    def __repr__(self):
        return (
            f"StanfordDataset(root={self.root_dir!r}, "
            f"split={self.split!r}, n_files={len(self.files)})"
        )
