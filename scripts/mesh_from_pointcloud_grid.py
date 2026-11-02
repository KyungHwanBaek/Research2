#!/usr/bin/env python3
"""Create a coarse triangular surface approximation from an ASCII point-cloud PLY."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def load_ascii_ply(path: Path, max_points: int | None) -> np.ndarray:
    header = True
    points: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if header:
                if line.strip() == "end_header":
                    header = False
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if max_points is not None and len(points) >= max_points:
                break
    if not points:
        raise SystemExit(f"no points in {path}")
    return np.asarray(points, dtype=np.float64)


def remove_outliers(points: np.ndarray, keep_percentile: float) -> np.ndarray:
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    threshold = np.percentile(dist, keep_percentile)
    kept = points[dist <= threshold]
    return kept if len(kept) >= 16 else points


def pca_project(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def build_grid_mesh(points: np.ndarray, grid_size: int, min_cell_points: int) -> tuple[np.ndarray, np.ndarray]:
    uv = pca_project(points)
    mins = uv.min(axis=0)
    spans = np.maximum(uv.max(axis=0) - mins, 1e-9)
    ij = np.floor((uv - mins) / spans * (grid_size - 1)).astype(int)
    ij = np.clip(ij, 0, grid_size - 1)

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(map(tuple, ij)):
        buckets.setdefault(key, []).append(idx)

    vertex_index: dict[tuple[int, int], int] = {}
    vertices: list[np.ndarray] = []
    for key, indices in buckets.items():
        if len(indices) < min_cell_points:
            continue
        vertex_index[key] = len(vertices)
        vertices.append(np.median(points[indices], axis=0))

    faces: list[tuple[int, int, int]] = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            a = vertex_index.get((i, j))
            b = vertex_index.get((i + 1, j))
            c = vertex_index.get((i, j + 1))
            d = vertex_index.get((i + 1, j + 1))
            if a is not None and b is not None and c is not None:
                faces.append((a, b, c))
            if b is not None and d is not None and c is not None:
                faces.append((b, d, c))

    if len(vertices) < 3 or not faces:
        raise SystemExit("not enough occupied grid cells to build a surface")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def write_binary_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for vertex in vertices:
            handle.write(struct.pack("<fff", float(vertex[0]), float(vertex[1]), float(vertex[2])))
        for face in faces:
            handle.write(struct.pack("<Biii", 3, int(face[0]), int(face[1]), int(face[2])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--min-cell-points", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--keep-percentile", type=float, default=99.0)
    args = parser.parse_args()

    points = load_ascii_ply(args.input_ply, args.max_points)
    points = remove_outliers(points, args.keep_percentile)
    vertices, faces = build_grid_mesh(points, args.grid_size, args.min_cell_points)
    write_binary_ply(args.out, vertices, faces)
    print(f"mesh={args.out}")
    print(f"source_points={len(points)} vertices={len(vertices)} faces={len(faces)}")


if __name__ == "__main__":
    main()
