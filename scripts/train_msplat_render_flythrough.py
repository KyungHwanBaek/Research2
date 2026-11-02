#!/usr/bin/env python3
"""Train msplat and render interpolated novel-view fly-through frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_rgb(array: np.ndarray, path: Path) -> None:
    image = np.clip(array, 0.0, 1.0)
    image = (image * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def resize_for_gif(path: Path, width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.width <= width:
        return image
    scale = width / image.width
    return image.resize((width, max(1, int(image.height * scale))), Image.Resampling.LANCZOS)


def make_gif(frame_paths: list[Path], output: Path, width: int, duration_ms: int) -> None:
    frames = [resize_for_gif(path, width) for path in frame_paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def make_contact_sheet(paths: list[Path], output: Path, thumb_w: int = 280) -> None:
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        scale = thumb_w / image.width
        thumb = image.resize((thumb_w, max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb.width, thumb.height + 30), "white")
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb.height + 7), path.name, fill=(30, 30, 30))
        thumbs.append(canvas)
    if not thumbs:
        return
    cols = min(4, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * cell_w, (i // cols) * cell_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def rotmat_to_quat(matrix: np.ndarray) -> np.ndarray:
    m = matrix
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    idx = int(np.argmax(np.diag(m)))
    if idx == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if idx == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float32,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    return np.cos(theta) * q0 + sin_theta * (q1 - q0 * dot) / sin_theta_0


def interpolate_pose(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = (1.0 - t) * a[:3, 3] + t * b[:3, 3]
    qa = rotmat_to_quat(a[:3, :3])
    qb = rotmat_to_quat(b[:3, :3])
    pose[:3, :3] = quat_to_rotmat(slerp(qa, qb, t))
    return pose


def interpolated_path(poses: list[np.ndarray], frames: int) -> list[np.ndarray]:
    if len(poses) < 2:
        return poses
    result: list[np.ndarray] = []
    segments = len(poses) - 1
    for i in range(frames):
        u = i / max(frames - 1, 1)
        s = min(segments - 1, int(u * segments))
        local = u * segments - s
        result.append(interpolate_pose(poses[s], poses[s + 1], local))
    return result


def camera_path(dataset: object, key_indices: list[int]) -> list[np.ndarray]:
    poses = []
    for index in key_indices:
        if index < 0 or index >= dataset.num_train:
            raise SystemExit(f"camera index {index} outside 0..{dataset.num_train - 1}")
        poses.append(np.asarray(dataset.camera_pose(index), dtype=np.float32))
    return poses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ply-output", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--gif-output", required=True)
    parser.add_argument("--contact-sheet", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--num-iters", type=int, default=8000)
    parser.add_argument("--downscale-factor", type=float, default=2.0)
    parser.add_argument("--key-cameras", default="0,6,13,19,25")
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--gif-width", type=int, default=540)
    parser.add_argument("--gif-duration-ms", type=int, default=90)
    args = parser.parse_args()

    from msplat import Dataset, GaussianTrainer, TrainingConfig, cleanup, sync

    config = TrainingConfig(
        iterations=args.num_iters,
        sh_degree=3,
        sh_degree_interval=1000,
        ssim_weight=0.2,
        num_downscales=0,
        resolution_schedule=3000,
        refine_every=100,
        warmup_length=500,
        reset_alpha_every=30,
        densify_grad_thresh=0.0002,
        densify_size_thresh=0.01,
        stop_screen_size_at=4000,
        split_screen_size=0.05,
        keep_crs=False,
        downscale_factor=args.downscale_factor,
        output=args.ply_output,
        save_every=-1,
    )
    dataset = Dataset(args.input, downscale_factor=args.downscale_factor)
    trainer = GaussianTrainer(dataset, config)
    history: list[dict[str, float | int]] = []

    def on_step(stats: object) -> None:
        row = {
            "iteration": int(stats.iteration),
            "splat_count": int(stats.splat_count),
            "ms_per_step": float(stats.ms_per_step),
        }
        history.append(row)
        print(
            f"step={row['iteration']:>6}  "
            f"splats={row['splat_count']:>8,}  "
            f"ms={row['ms_per_step']:.1f}"
        )

    print(f"Loaded {dataset.num_train} train cameras")
    trainer.train(on_step, callback_every=200)
    sync()
    Path(args.ply_output).parent.mkdir(parents=True, exist_ok=True)
    trainer.export_ply(args.ply_output)

    key_indices = [int(x) for x in args.key_cameras.split(",") if x.strip()]
    key_poses = camera_path(dataset, key_indices)
    path_poses = interpolated_path(key_poses, args.num_frames)
    frames_dir = Path(args.frames_dir)
    frame_paths: list[Path] = []
    for i, pose in enumerate(path_poses):
        rendered = trainer.render_from_pose(pose, ref_cam_idx=0)
        path = frames_dir / f"novel_{i:03d}.png"
        save_rgb(rendered, path)
        frame_paths.append(path)
        if i % 8 == 0:
            print(f"frame={path}")

    make_gif(frame_paths, Path(args.gif_output), width=args.gif_width, duration_ms=args.gif_duration_ms)
    contact_indices = np.linspace(0, len(frame_paths) - 1, min(12, len(frame_paths)), dtype=int)
    make_contact_sheet([frame_paths[int(i)] for i in contact_indices], Path(args.contact_sheet))

    metrics = {
        "input": args.input,
        "num_iters": args.num_iters,
        "downscale_factor": args.downscale_factor,
        "train_cameras": int(dataset.num_train),
        "key_cameras": key_indices,
        "num_novel_frames": len(frame_paths),
        "final_gaussians": int(trainer.splat_count),
        "ply_output": args.ply_output,
        "frames_dir": args.frames_dir,
        "gif_output": args.gif_output,
        "contact_sheet": args.contact_sheet,
        "history": history,
        "note": "Novel views are interpolated camera poses rendered from the trained 3DGS scene.",
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cleanup()


if __name__ == "__main__":
    main()
