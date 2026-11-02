#!/usr/bin/env python3
"""List or download one official Hugging Face dataset archive.

This script deliberately avoids snapshot_download so the full dataset is not
downloaded by accident.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "DivisonOfficer/Pixel-aligned_RGB-NIR_stereo_dataset"
REPO_TYPE = "dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List repository files and exit.")
    parser.add_argument("--filename", help="Exact archive filename to download.")
    parser.add_argument("--out-dir", default="data/raw", help="Download destination directory.")
    args = parser.parse_args()

    if args.list:
        files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
        for name in files:
            print(name)
        return

    if not args.filename:
        raise SystemExit("Pass --list or --filename <exact-file.tar.gz>.")
    if "/" in args.filename or not args.filename.endswith(".tar.gz"):
        raise SystemExit("Use an exact root-level .tar.gz filename, not a directory or pattern.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=args.filename,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )
    p = Path(path)
    print(f"downloaded={p}")
    print(f"size_bytes={p.stat().st_size}")


if __name__ == "__main__":
    main()
