#!/usr/bin/env python
"""
Download a subset of the ABC dataset (A Big CAD model dataset) for NPAQ training.

Dataset home: https://deep-geometry.github.io/abc-dataset/
Paper: Koch et al. (2019), CVPR

ABC v1.0 is hosted on the Internet Archive (~500 MB per chunk, ~100 chunks total).
Each chunk contains ~5 000 OBJ files of CAD surface models.

Usage:
  # Download chunks 0000 and 0001, extract into data/abc/
  python scripts/download_abc.py --chunks 0 1 --output data/abc

  # Download chunk 0000 only, then auto-preprocess for training
  python scripts/download_abc.py --chunks 0 --output data/abc --preprocess

  # Preprocess only (data already downloaded)
  python scripts/download_abc.py --chunks 0 --output data/abc --preprocess --skip-download
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


# ─── URLs ─────────────────────────────────────────────────────────────────────
# ABC v1.0 OBJ chunks on Internet Archive.
# Format: abc_{chunk:04d}_obj_v{ver:02d}.7z
_BASE_URL = "https://archive.org/download/abc_v1.0"
_FILENAME_TEMPLATE = "abc_{chunk:04d}_obj_v00.7z"

# Alternative mirror (Zenodo DOI redirect).  Used only when Archive.org fails.
_ZENODO_URL = "https://zenodo.org/record/3697452/files"


def _url_for_chunk(chunk_id: int) -> str:
    fname = _FILENAME_TEMPLATE.format(chunk=chunk_id)
    return f"{_BASE_URL}/{fname}"


def _zenodo_url_for_chunk(chunk_id: int) -> str:
    fname = _FILENAME_TEMPLATE.format(chunk=chunk_id)
    return f"{_ZENODO_URL}/{fname}"


# ─── Extraction ───────────────────────────────────────────────────────────────

def _extract_7z(archive: str, dest_dir: str) -> bool:
    """Extract a .7z archive using py7zr (pip) or system 7z/7za binary."""
    os.makedirs(dest_dir, exist_ok=True)
    # Try py7zr first (pure Python, no system dependency)
    try:
        import py7zr
        with py7zr.SevenZipFile(archive, mode='r') as z:
            z.extractall(path=dest_dir)
        return True
    except ImportError:
        pass

    # Fall back to system 7z / 7za
    for cmd in ('7z', '7za', '7zr'):
        if shutil.which(cmd):
            ret = subprocess.run([cmd, 'x', archive, f'-o{dest_dir}', '-y'],
                                 capture_output=True)
            if ret.returncode == 0:
                return True
            print(f"  [extract] {cmd} failed: {ret.stderr.decode()[:200]}")

    print("  [extract] ERROR: no extraction tool found.")
    print("  Install py7zr:      pip install py7zr")
    print("  Or system 7-Zip:   sudo apt-get install p7zip-full")
    return False


# ─── Download ─────────────────────────────────────────────────────────────────

class _ProgressBar:
    def __init__(self, total: int):
        self.total = total
        self._last_pct = -1

    def __call__(self, blocks: int, block_size: int, total_size: int):
        if total_size > 0:
            pct = min(100, int(blocks * block_size * 100 / total_size))
        else:
            pct = 0
        if pct != self._last_pct and pct % 5 == 0:
            bar = '#' * (pct // 5) + '.' * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%", end='', flush=True)
            self._last_pct = pct


def _download_file(url: str, dest: str) -> bool:
    """Download url → dest, with a progress bar.  Returns True on success."""
    try:
        print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, dest, reporthook=_ProgressBar(0))
        print()   # newline after progress bar
        return True
    except Exception as exc:
        print(f"\n  Download failed: {exc}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_chunk(chunk_id: int, out_dir: str, skip_download: bool = False) -> bool:
    """
    Download chunk `chunk_id` of ABC v1.0 to out_dir/abc_{chunk_id:04d}/.

    Returns True if the chunk directory is ready (already present or freshly
    downloaded and extracted).
    """
    chunk_dir = os.path.join(out_dir, f"{chunk_id:04d}")
    archive   = os.path.join(out_dir, _FILENAME_TEMPLATE.format(chunk=chunk_id))

    # Already extracted?
    if os.path.isdir(chunk_dir) and any(Path(chunk_dir).rglob('*.obj')):
        print(f"[abc] Chunk {chunk_id:04d} already extracted at {chunk_dir}")
        return True

    if skip_download:
        print(f"[abc] --skip-download set but {chunk_dir} is missing; cannot continue.")
        return False

    # Download (try archive.org, then zenodo as fallback)
    if not os.path.isfile(archive):
        os.makedirs(out_dir, exist_ok=True)
        url = _url_for_chunk(chunk_id)
        print(f"[abc] Downloading chunk {chunk_id:04d} …")
        ok = _download_file(url, archive)
        if not ok:
            print(f"  Retrying with Zenodo mirror …")
            ok = _download_file(_zenodo_url_for_chunk(chunk_id), archive)
        if not ok:
            print(f"[abc] Failed to download chunk {chunk_id:04d}.")
            print( "      You can manually download from:")
            print(f"        {url}")
            print(f"      and place the .7z file at: {archive}")
            return False
    else:
        print(f"[abc] Archive already present: {archive}")

    # Extract
    print(f"[abc] Extracting to {chunk_dir} …")
    if not _extract_7z(archive, chunk_dir):
        return False

    n_obj = len(list(Path(chunk_dir).rglob('*.obj')))
    print(f"[abc] Chunk {chunk_id:04d} ready: {n_obj} OBJ files.")
    return True


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_chunk(
    chunk_id: int,
    abc_dir: str,
    prep_dir: str,
    split: str,
    max_files: int,
    n_points: int,
) -> bool:
    """Run abc_preprocess.py on a downloaded chunk."""
    chunk_dir = os.path.join(abc_dir, f"{chunk_id:04d}")
    out_dir   = os.path.join(prep_dir, split)
    script    = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'abc_preprocess.py'
    )

    cmd = [
        sys.executable, script,
        '--input',     chunk_dir,
        '--output',    out_dir,
        '--n_points',  str(n_points),
        '--max_files', str(max_files),
    ]
    print(f"[abc] Preprocessing chunk {chunk_id:04d} → {out_dir} …")
    print(f"  {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    return ret.returncode == 0


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Download ABC dataset chunks for NPAQ training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--chunks', nargs='+', type=int, default=[0],
                   help='Chunk IDs to download (0-based). E.g. --chunks 0 1')
    p.add_argument('--output', default='data/abc',
                   help='Root directory for downloaded chunks')
    p.add_argument('--preprocess', action='store_true',
                   help='Also run abc_preprocess.py after downloading')
    p.add_argument('--prep-dir', default='data/abc_preprocessed',
                   help='Output root for abc_preprocess.py')
    p.add_argument('--skip-download', action='store_true',
                   help='Skip download; only extract/preprocess')
    p.add_argument('--max-files', type=int, default=5000,
                   help='Max OBJ files to preprocess per chunk')
    p.add_argument('--n-points', type=int, default=4096,
                   help='Surface samples per mesh during preprocessing')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    splits = ['train', 'val', 'val']   # chunk 0 → train, chunk 1 → val, others → val
    for i, chunk_id in enumerate(args.chunks):
        split = splits[min(i, len(splits) - 1)]
        print(f"\n{'='*60}")
        print(f"  ABC Chunk {chunk_id:04d}  (split: {split})")
        print(f"{'='*60}")

        ok = download_chunk(chunk_id, args.output,
                            skip_download=args.skip_download)
        if not ok:
            print(f"[abc] Skipping chunk {chunk_id:04d} due to download/extraction error.")
            continue

        if args.preprocess:
            ok = preprocess_chunk(
                chunk_id,
                abc_dir   = args.output,
                prep_dir  = args.prep_dir,
                split     = split,
                max_files = args.max_files,
                n_points  = args.n_points,
            )
            if not ok:
                print(f"[abc] Preprocessing failed for chunk {chunk_id:04d}.")

    print("\n[abc] Done.")
    if args.preprocess:
        print(f"  Preprocessed data: {args.prep_dir}/train/  and  {args.prep_dir}/val/")
        print( "  To train with ABC data:")
        print( "    python scripts/train.py --config configs/train_abc.yaml")


if __name__ == '__main__':
    main()
