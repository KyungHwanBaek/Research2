#!/usr/bin/env python3
"""Train msplat on a scene and render selected training-camera views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_render(array: np.ndarray, path: Path) -> None:
    image = np.clip(array, 0.0, 1.0)
    image = (image * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def make_contact_sheet(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return
    thumb_w = 360
    thumbs = []
    for path, image in zip(paths, images, strict=True):
        scale = thumb_w / image.width
        thumb = image.resize((thumb_w, max(1, int(image.height * scale))))
        canvas = Image.new("RGB", (thumb.width, thumb.height + 34), "white")
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb.height + 8), path.name, fill=(30, 30, 30))
        thumbs.append(canvas)

    cols = min(3, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(thumb, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ply-output", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--contact-sheet", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--num-iters", type=int, default=3000)
    parser.add_argument("--downscale-factor", type=float, default=4.0)
    parser.add_argument("--camera-indices", default="0,13,25")
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
    trainer.train(on_step, callback_every=100)
    sync()

    ply_output = Path(args.ply_output)
    ply_output.parent.mkdir(parents=True, exist_ok=True)
    trainer.export_ply(str(ply_output))

    render_dir = Path(args.render_dir)
    indices = [int(x) for x in args.camera_indices.split(",") if x.strip()]
    render_paths: list[Path] = []
    for index in indices:
        if index < 0 or index >= dataset.num_train:
            raise SystemExit(f"camera index {index} outside 0..{dataset.num_train - 1}")
        rendered = trainer.render(index)
        path = render_dir / f"render_cam_{index:03d}.png"
        save_render(rendered, path)
        render_paths.append(path)
        print(f"render={path}")

    make_contact_sheet(render_paths, Path(args.contact_sheet))

    metrics = {
        "input": args.input,
        "num_iters": args.num_iters,
        "downscale_factor": args.downscale_factor,
        "train_cameras": int(dataset.num_train),
        "final_gaussians": int(trainer.splat_count),
        "ply_output": str(ply_output),
        "render_paths": [str(path) for path in render_paths],
        "contact_sheet": args.contact_sheet,
        "history": history,
        "note": "Rendered RGB views from the trained 3DGS scene at selected training camera indices.",
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cleanup()


if __name__ == "__main__":
    main()
