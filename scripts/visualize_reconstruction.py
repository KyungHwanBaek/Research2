#!/usr/bin/env python3
"""Create static trajectory and sparse point cloud figures from COLMAP text model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans"})


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * z * x + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * z * x - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def load_camera_centers(images_txt: Path) -> np.ndarray:
    lines = [line for line in images_txt.read_text().splitlines() if line and not line.startswith("#")]
    centers = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        q = np.array([float(x) for x in parts[1:5]])
        t = np.array([float(x) for x in parts[5:8]])
        r = qvec_to_rotmat(q)
        centers.append(-r.T @ t)
    return np.asarray(centers)


def load_points(points_txt: Path, limit: int = 200_000) -> np.ndarray:
    pts = []
    for line in points_txt.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(pts) >= limit:
            break
    return np.asarray(pts)


def plot_xyz(points: np.ndarray, cameras: np.ndarray, output: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2, c="0.35", alpha=0.35)
    if len(cameras):
        ax.plot(cameras[:, 0], cameras[:, 1], cameras[:, 2], color="tab:red", linewidth=1.5)
        ax.scatter(cameras[:, 0], cameras[:, 1], cameras[:, 2], s=8, color="tab:red")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("COLMAP sparse points and camera trajectory")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text_model_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/figures/rgb_colmap_sparse_overview.png"))
    args = parser.parse_args()

    cameras = load_camera_centers(args.text_model_dir / "images.txt")
    points = load_points(args.text_model_dir / "points3D.txt")
    plot_xyz(points, cameras, args.out)
    print(f"figure={args.out}")


if __name__ == "__main__":
    main()
