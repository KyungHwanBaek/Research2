#!/usr/bin/env python3
"""Create an approximate stitched COLMAP text model from disconnected components.

This is a visualization/diagnostic utility, not bundle-adjusted SfM. It aligns
components by shared registered images when possible, then falls back to
time-ordered endpoint/direction stitching for gaps.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * z * x + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * z * x - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=float,
    )


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.8:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + 1e-12)
        return -np.eye(3) + 2 * np.outer(v, v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float)
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = xd.T @ xs / len(src)
    u, singular, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(u @ vt))
    sfix = np.diag([1.0, 1.0, sign])
    rot = u @ sfix @ vt
    var = float((xs * xs).sum() / len(src))
    scale = float(np.trace(np.diag(singular) @ sfix) / max(var, 1e-12))
    trans = mu_d - scale * rot @ mu_s
    return scale, rot, trans


def parse_images(images_txt: Path) -> list[dict]:
    lines = [line.strip() for line in images_txt.read_text().splitlines() if line.strip() and not line.startswith("#")]
    images = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        q = np.array([float(x) for x in parts[1:5]])
        t = np.array([float(x) for x in parts[5:8]])
        center = -qvec_to_rotmat(q).T @ t
        images.append({"name": parts[9], "center": center})
    return sorted(images, key=lambda item: int(Path(item["name"]).stem))


def parse_points(points_txt: Path, limit: int | None = None) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    points = []
    for line in points_txt.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb = (int(parts[4]), int(parts[5]), int(parts[6]))
        points.append((xyz, rgb))
        if limit is not None and len(points) >= limit:
            break
    return points


def median_step(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return 1.0
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    steps = steps[steps > 1e-9]
    return float(np.median(steps)) if len(steps) else 1.0


def direction(centers: np.ndarray, tail: bool) -> np.ndarray:
    if len(centers) < 2:
        return np.array([1.0, 0.0, 0.0])
    k = min(8, len(centers) - 1)
    if tail:
        vec = centers[-1] - centers[-1 - k]
    else:
        vec = centers[k] - centers[0]
    if np.linalg.norm(vec) < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return vec


def transform_points(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (scale * (rot @ points.T)).T + trans


def write_text_model(out_dir: Path, cameras: list[dict], points: list[tuple[np.ndarray, tuple[int, int, int]]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cameras.txt").write_text("# Synthetic stitched model\n1 PINHOLE 1 1 1 1 0.5 0.5\n", encoding="utf-8")
    image_lines = ["# Synthetic stitched model"]
    for idx, cam in enumerate(cameras, 1):
        c = cam["center"]
        image_lines.append(f"{idx} 1 0 0 0 {-c[0]:.9f} {-c[1]:.9f} {-c[2]:.9f} 1 {cam['name']}")
        image_lines.append("0 0 -1")
    (out_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    point_lines = ["# Synthetic stitched model"]
    for idx, (xyz, rgb) in enumerate(points, 1):
        point_lines.append(
            f"{idx} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} {rgb[0]} {rgb[1]} {rgb[2]} 0.0"
        )
    (out_dir / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")


def write_ply(out: Path, points: list[tuple[np.ndarray, tuple[int, int, int]]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, rgb in points:
            handle.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {rgb[0]} {rgb[1]} {rgb[2]}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--components", nargs="+", required=True)
    parser.add_argument("--out-text", type=Path, required=True)
    parser.add_argument("--out-ply", type=Path, required=True)
    parser.add_argument("--max-points-per-component", type=int, default=12000)
    args = parser.parse_args()

    placed_by_name: dict[str, np.ndarray] = {}
    global_cameras: list[dict] = []
    global_points: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    prev_centers: np.ndarray | None = None

    for comp in args.components:
        text_dir = args.sparse_root / f"{comp}_text_for_summary"
        images = parse_images(text_dir / "images.txt")
        local_centers = np.array([img["center"] for img in images])
        local_by_name = {img["name"]: img["center"] for img in images}
        shared = [name for name in local_by_name if name in placed_by_name]

        if not global_cameras:
            scale, rot, trans = 1.0, np.eye(3), np.zeros(3)
            method = "anchor"
        elif len(shared) >= 3:
            src = np.array([local_by_name[name] for name in shared])
            dst = np.array([placed_by_name[name] for name in shared])
            scale, rot, trans = umeyama(src, dst)
            method = f"shared:{len(shared)}"
        elif len(shared) >= 2:
            src = np.array([local_by_name[name] for name in shared])
            dst = np.array([placed_by_name[name] for name in shared])
            scale = np.linalg.norm(dst[-1] - dst[0]) / max(np.linalg.norm(src[-1] - src[0]), 1e-12)
            rot = rotation_between(src[-1] - src[0], dst[-1] - dst[0])
            trans = dst[0] - scale * rot @ src[0]
            method = f"shared-line:{len(shared)}"
        else:
            assert prev_centers is not None
            target_dir = direction(prev_centers, tail=True)
            source_dir = direction(local_centers, tail=False)
            rot = rotation_between(source_dir, target_dir)
            scale = median_step(prev_centers[-min(30, len(prev_centers)) :]) / max(
                median_step(local_centers[: min(30, len(local_centers))]), 1e-12
            )
            scale = float(np.clip(scale, 0.25, 4.0))
            gap = max(median_step(prev_centers[-min(30, len(prev_centers)) :]), 1e-3) * 3.0
            anchor = prev_centers[-1] + target_dir / (np.linalg.norm(target_dir) + 1e-12) * gap
            trans = anchor - scale * rot @ local_centers[0]
            method = "endpoint"

        transformed_centers = transform_points(local_centers, scale, rot, trans)
        added_cameras = 0
        for img, center in zip(images, transformed_centers):
            if img["name"] in placed_by_name:
                continue
            placed_by_name[img["name"]] = center
            global_cameras.append({"name": img["name"], "center": center})
            added_cameras += 1

        points = parse_points(text_dir / "points3D.txt", args.max_points_per_component)
        if points:
            xyz = np.array([p[0] for p in points])
            transformed = transform_points(xyz, scale, rot, trans)
            global_points.extend([(pt, rgb) for pt, (_, rgb) in zip(transformed, points)])

        prev_centers = np.array([placed_by_name[cam["name"]] for cam in global_cameras[-added_cameras:]]) if added_cameras else transformed_centers
        print(f"component {comp}: method={method}, added_cameras={added_cameras}, added_points={len(points)}")

    global_cameras.sort(key=lambda item: int(Path(item["name"]).stem))
    write_text_model(args.out_text, global_cameras, global_points)
    write_ply(args.out_ply, global_points)
    print(f"stitched_cameras={len(global_cameras)}")
    print(f"stitched_points={len(global_points)}")
    print(f"text_model={args.out_text}")
    print(f"ply={args.out_ply}")


if __name__ == "__main__":
    main()
