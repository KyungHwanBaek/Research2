#!/usr/bin/env python3
"""Package a COLMAP sparse reconstruction as a 3D Gaussian Splatting scene."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def registered_image_names(images_txt: Path) -> list[str]:
    names: list[str] = []
    with images_txt.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            names.append(parts[-1])
            handle.readline()
    return names


def copy_sparse_model(src: Path, dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ("cameras.bin", "images.bin", "points3D.bin", "rigs.bin", "frames.bin"):
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)
            copied.append(name)
    required = {"cameras.bin", "images.bin", "points3D.bin"}
    missing = sorted(required - set(copied))
    if missing:
        raise SystemExit(f"missing required sparse model files in {src}: {', '.join(missing)}")
    return copied


def copy_registered_images(src_images: Path, dst_images: Path, names: list[str]) -> None:
    dst_images.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for name in names:
        source = src_images / name
        if not source.exists():
            missing.append(name)
            continue
        shutil.copy2(source, dst_images / name)
    if missing:
        raise SystemExit(f"missing registered source images: {', '.join(missing[:10])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--sparse-bin", type=Path, required=True)
    parser.add_argument("--sparse-text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="rgb_colmap_well_lit_outdoor_seq30_fixed")
    args = parser.parse_args()

    names = registered_image_names(args.sparse_text / "images.txt")
    if not names:
        raise SystemExit(f"no registered images found in {args.sparse_text / 'images.txt'}")

    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")

    images_out = args.out / "images"
    sparse_out = args.out / "sparse" / "0"
    copy_registered_images(args.source_images, images_out, names)
    sparse_files = copy_sparse_model(args.sparse_bin, sparse_out)

    manifest = {
        "name": args.name,
        "scene_type": "3dgs_training_input",
        "source_images": str(args.source_images),
        "source_sparse_bin": str(args.sparse_bin),
        "source_sparse_text": str(args.sparse_text),
        "registered_image_count": len(names),
        "registered_images": names,
        "sparse_files": sparse_files,
        "layout": {
            "images": "registered RGB-left images used by COLMAP",
            "sparse/0": "COLMAP binary sparse model",
        },
        "note": (
            "This folder is ready as a standard COLMAP-backed 3D Gaussian "
            "Splatting scene input. Training still requires a compatible 3DGS "
            "implementation and usually an NVIDIA CUDA GPU."
        ),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(
        "\n".join(
            [
                f"# {args.name}",
                "",
                "This folder packages the current best RGB-left COLMAP result as a 3DGS training scene.",
                "",
                "- `images/`: registered images only",
                "- `sparse/0/`: COLMAP binary sparse model",
                "- `manifest.json`: source paths and registered image list",
                "",
                "Typical graphdeco 3DGS command on a CUDA machine:",
                "",
                "```bash",
                f"python train.py -s {args.out}",
                "```",
                "",
                "This Mac does not have CUDA, so local 3DGS training is not expected to run with the official implementation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"out={args.out}")
    print(f"registered_images={len(names)}")
    print(f"first={names[0]} last={names[-1]}")


if __name__ == "__main__":
    main()
