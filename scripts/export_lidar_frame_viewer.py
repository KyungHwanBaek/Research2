#!/usr/bin/env python3
"""Export one raw dataset LiDAR frame as PLY and a self-contained HTML viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      --ink: #1b1f24;
      --muted: #7b828a;
      --line: #d9dde2;
      --panel: #f1f2f4;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; width: 100%; height: 100%; color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow: hidden; }
    body { display: flex; flex-direction: column; background: #fff; }
    header { flex: 0 0 auto; padding: 10px 16px 12px; border-bottom: 1px solid #eceff2; background: #fff; }
    h1 { margin: 0 0 10px; font-size: clamp(22px, 3vw, 36px); line-height: 1.05; letter-spacing: 0; font-weight: 760; }
    .bar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    button { border: 1px solid var(--line); background: #fff; color: var(--ink); min-height: 44px; padding: 0 16px; border-radius: 14px; font-size: 18px; cursor: pointer; }
    button.active { background: #171b20; color: #fff; border-color: #171b20; }
    label { display: inline-flex; align-items: center; gap: 10px; font-size: 18px; font-weight: 650; white-space: nowrap; }
    input[type="range"] { width: min(42vw, 520px); accent-color: #111820; }
    main { position: relative; flex: 1 1 auto; min-height: 0; background: var(--panel); }
    canvas { display: block; width: 100%; height: 100%; cursor: grab; }
    canvas:active { cursor: grabbing; }
    footer { flex: 0 0 auto; min-height: 42px; display: flex; align-items: center; gap: 28px; padding: 8px 16px; color: #8a9096; font-size: 18px; font-weight: 650; background: #fff; border-top: 1px solid #eceff2; flex-wrap: wrap; }
    @media (max-width: 720px) {
      header { padding: 10px; }
      h1 { font-size: 22px; }
      button { min-height: 38px; padding: 0 12px; border-radius: 12px; font-size: 15px; }
      label { font-size: 15px; }
      input[type="range"] { width: 170px; }
      footer { gap: 14px; font-size: 14px; min-height: 36px; padding: 7px 10px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <div class="bar">
      <button id="perspective" class="active">Perspective</button>
      <button id="top">Top</button>
      <button id="side">Side</button>
      <label>Point size <input id="pointSize" type="range" min="0.5" max="5" step="0.1" value="1.1" /></label>
    </div>
  </header>
  <main><canvas id="view"></canvas></main>
  <footer>
    <span>Drag: rotate</span>
    <span>Scroll: zoom</span>
    <span>Single raw LiDAR frame, not SLAM-accumulated</span>
    <span>__POINT_COUNT__ sampled points of __VALID_COUNT__ valid</span>
  </footer>
  <script id="model-data" type="application/json">__DATA_JSON__</script>
  <script>
    const data = JSON.parse(document.getElementById("model-data").textContent);
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const buttons = ["perspective", "top", "side"].map((id) => document.getElementById(id));
    const sizeInput = document.getElementById("pointSize");
    let mode = "perspective";
    let yaw = -0.72;
    let pitch = 0.72;
    let zoom = 80;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    const raw = data.points.map((p) => ({ x: p[0], y: p[1], z: p[2], color: `rgb(${p[3]},${p[4]},${p[5]})` }));
    const center = raw.reduce((a, p) => ({ x: a.x + p.x, y: a.y + p.y, z: a.z + p.z }), { x: 0, y: 0, z: 0 });
    center.x /= raw.length || 1; center.y /= raw.length || 1; center.z /= raw.length || 1;
    const ext = raw.reduce((m, p) => Math.max(m, Math.abs(p.x - center.x), Math.abs(p.y - center.y), Math.abs(p.z - center.z)), 1);

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function setMode(next) {
      mode = next;
      buttons.forEach((btn) => btn.classList.toggle("active", btn.id === next));
      if (next === "top") { yaw = 0; pitch = Math.PI / 2; zoom = 100; }
      if (next === "side") { yaw = -Math.PI / 2; pitch = 0.05; zoom = 100; }
      if (next === "perspective") { yaw = -0.72; pitch = 0.72; zoom = 80; }
      draw();
    }

    function project(p) {
      const x = (p.x - center.x) / ext;
      const y = (p.y - center.y) / ext;
      const z = (p.z - center.z) / ext;
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * x + sy * z;
      const z1 = -sy * x + cy * z;
      const y1 = cp * y - sp * z1;
      const z2 = sp * y + cp * z1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const persp = mode === "perspective" ? 1 / (1.9 - z2 * 0.55) : 1;
      return { sx: w / 2 + x1 * zoom * 4.8 * persp, sy: h / 2 - y1 * zoom * 4.8 * persp, depth: z2, scale: persp };
    }

    function grid() {
      ctx.strokeStyle = "#d8dce1";
      ctx.lineWidth = 1;
      for (let i = -10; i <= 10; i++) {
        const a = project({ x: center.x + (i / 10) * ext * 2.4, y: center.y - ext * 1.05, z: center.z - ext * 1.2 });
        const b = project({ x: center.x + (i / 10) * ext * 2.4, y: center.y - ext * 1.05, z: center.z + ext * 1.2 });
        const c = project({ x: center.x - ext * 1.2, y: center.y - ext * 1.05, z: center.z + (i / 10) * ext * 2.4 });
        const d = project({ x: center.x + ext * 1.2, y: center.y - ext * 1.05, z: center.z + (i / 10) * ext * 2.4 });
        ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(c.sx, c.sy); ctx.lineTo(d.sx, d.sy); ctx.stroke();
      }
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      grid();
      const projected = raw.map((p) => ({ ...project(p), color: p.color })).sort((a, b) => a.depth - b.depth);
      const r = parseFloat(sizeInput.value);
      for (const p of projected) {
        ctx.fillStyle = p.color;
        ctx.globalAlpha = 0.76;
        ctx.fillRect(p.sx, p.sy, r * p.scale, r * p.scale);
      }
      ctx.globalAlpha = 1;
    }

    buttons.forEach((btn) => btn.addEventListener("click", () => setMode(btn.id)));
    sizeInput.addEventListener("input", draw);
    canvas.addEventListener("pointerdown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener("pointerup", () => { dragging = false; });
    canvas.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      yaw += (e.clientX - lastX) * 0.008;
      pitch = Math.max(-1.45, Math.min(1.45, pitch + (e.clientY - lastY) * 0.008));
      lastX = e.clientX; lastY = e.clientY;
      draw();
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      zoom = Math.max(20, Math.min(220, zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
      draw();
    }, { passive: false });
    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""


def load_lidar_frame(h5_path: Path, frame_index: int) -> tuple[str, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        frame_names = sorted(handle["frame"].keys())
        if frame_index < 0 or frame_index >= len(frame_names):
            raise SystemExit(f"frame index {frame_index} outside 0..{len(frame_names) - 1}")
        frame_name = frame_names[frame_index]
        points = handle["frame"][frame_name]["lidar"]["points"][()]
    points = points.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-6)
    return frame_name, points[valid]


def colorize(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(points, axis=1)
    lo, hi = np.percentile(distances, [2, 98])
    t = np.clip((distances - lo) / max(hi - lo, 1e-6), 0, 1)
    r = (40 + 210 * t).astype(np.uint8)
    g = (190 - 110 * t).astype(np.uint8)
    b = (230 - 185 * t).astype(np.uint8)
    return np.column_stack([r, g, b])


def deterministic_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_viewer(path: Path, title: str, points: np.ndarray, colors: np.ndarray, valid_count: int) -> None:
    payload_points = [
        [
            round(float(point[0]), 6),
            round(float(point[1]), 6),
            round(float(point[2]), 6),
            int(color[0]),
            int(color[1]),
            int(color[2]),
        ]
        for point, color in zip(points, colors, strict=True)
    ]
    html = (
        HTML_TEMPLATE.replace("__TITLE__", title)
        .replace("__POINT_COUNT__", f"{len(points):,}")
        .replace("__VALID_COUNT__", f"{valid_count:,}")
        .replace("__DATA_JSON__", json.dumps({"points": payload_points}, separators=(",", ":")))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_metrics(path: Path, h5_path: Path, frame_index: int, frame_name: str, points: np.ndarray, sampled: np.ndarray) -> None:
    distances = np.linalg.norm(points, axis=1)
    metrics = {
        "source_hdf5": str(h5_path),
        "frame_index": frame_index,
        "frame_name": frame_name,
        "valid_point_count": int(len(points)),
        "sampled_point_count": int(len(sampled)),
        "xyz_min": [float(x) for x in points.min(axis=0)],
        "xyz_max": [float(x) for x in points.max(axis=0)],
        "distance_mean": float(distances.mean()),
        "distance_median": float(np.median(distances)),
        "distance_p95": float(np.percentile(distances, 95)),
        "note": "Single raw LiDAR frame in dataset coordinates; this is not a SLAM-accumulated map.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("--frame-index", type=int, default=26)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--title", default="Raw LiDAR single-frame point cloud")
    parser.add_argument("--viewer-out", type=Path, required=True)
    parser.add_argument("--ply-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    args = parser.parse_args()

    frame_name, points = load_lidar_frame(args.h5_path, args.frame_index)
    sampled = deterministic_sample(points, args.max_points)
    colors = colorize(sampled)
    title = f"{args.title} · frame {args.frame_index:03d} ({frame_name})"
    write_ply(args.ply_out, sampled, colors)
    write_viewer(args.viewer_out, title, sampled, colors, len(points))
    write_metrics(args.metrics_out, args.h5_path, args.frame_index, frame_name, points, sampled)
    print(f"frame={frame_name}")
    print(f"valid_points={len(points)} sampled_points={len(sampled)}")
    print(f"viewer={args.viewer_out}")
    print(f"ply={args.ply_out}")
    print(f"metrics={args.metrics_out}")


if __name__ == "__main__":
    main()
