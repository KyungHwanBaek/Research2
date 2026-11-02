#!/usr/bin/env python3
"""Summarize a COLMAP text or binary model into JSON.

If binary model files are passed, this script asks COLMAP to convert to text in a
temporary sibling directory first.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def ensure_text_model(model_dir: Path) -> Path:
    if (model_dir / "images.txt").exists():
        return model_dir
    out = model_dir.parent / f"{model_dir.name}_text_for_summary"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(out),
            "--output_type",
            "TXT",
        ],
        check=True,
    )
    return out


def parse_cameras(path: Path) -> list[dict]:
    cameras = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cameras.append(
            {
                "camera_id": int(parts[0]),
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(x) for x in parts[4:]],
            }
        )
    return cameras


def parse_images(path: Path) -> list[dict]:
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    images = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        images.append(
            {
                "image_id": int(parts[0]),
                "qvec": [float(x) for x in parts[1:5]],
                "tvec": [float(x) for x in parts[5:8]],
                "camera_id": int(parts[8]),
                "name": parts[9],
            }
        )
    return images


def parse_points(path: Path) -> tuple[int, dict]:
    errors = []
    track_lengths = []
    count = 0
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        count += 1
        errors.append(float(parts[7]))
        track_lengths.append(max(0, (len(parts) - 8) // 2))
    stats = {}
    if errors:
        arr = np.asarray(errors, dtype=float)
        tracks = np.asarray(track_lengths, dtype=float)
        stats = {
            "reprojection_error_mean": float(arr.mean()),
            "reprojection_error_median": float(np.median(arr)),
            "reprojection_error_p95": float(np.percentile(arr, 95)),
            "track_length_mean": float(tracks.mean()),
            "track_length_median": float(np.median(tracks)),
        }
    return count, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("json_out", type=Path)
    parser.add_argument("--input-image-count", type=int)
    args = parser.parse_args()

    text_dir = ensure_text_model(args.model_dir)
    cameras = parse_cameras(text_dir / "cameras.txt")
    images = parse_images(text_dir / "images.txt")
    point_count, point_stats = parse_points(text_dir / "points3D.txt")

    total = args.input_image_count or len(images)
    report = {
        "model_dir": str(args.model_dir),
        "camera_count": len(cameras),
        "cameras": cameras,
        "registered_image_count": len(images),
        "input_image_count": total,
        "registration_ratio": len(images) / total if total else None,
        "sparse_point_count": point_count,
        **point_stats,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
