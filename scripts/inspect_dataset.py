#!/usr/bin/env python3
"""Inspect extracted dataset directory and storage.hdf5 without dumping large values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def sample_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item() if hasattr(arr.item(), "__class__") else str(arr)
    flat = arr.reshape(-1)
    return flat[: min(6, flat.size)].tolist()


def describe_h5(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(path), "groups": {}, "attrs": {}}
    with h5py.File(path, "r") as f:
        report["top_level_keys"] = list(f.keys())
        for key, value in f.attrs.items():
            report["attrs"][key] = sample_value(value)

        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset):
                entry = {
                    "kind": "dataset",
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "attrs": {k: sample_value(v) for k, v in obj.attrs.items()},
                }
            else:
                entry = {
                    "kind": "group",
                    "attrs": {k: sample_value(v) for k, v in obj.attrs.items()},
                }
            report["groups"][name] = entry

        f.visititems(visit)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-out", type=Path, default=Path("results/metrics/dataset_inspection.json"))
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"Missing extracted directory: {root}")

    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    frame_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    h5_files = sorted(root.rglob("*.hdf5"))

    report: dict[str, Any] = {
        "root": str(root),
        "file_count": len(files),
        "first_files": files[:80],
        "frame_dir_count": len(frame_dirs),
        "first_frame_dirs": [p.name for p in frame_dirs[:20]],
        "hdf5_files": [str(p) for p in h5_files],
    }
    if h5_files:
        report["hdf5"] = [describe_h5(p) for p in h5_files]

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:12000])
    print(f"json={args.json_out}")


if __name__ == "__main__":
    main()
