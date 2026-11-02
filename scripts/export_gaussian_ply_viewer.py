#!/usr/bin/env python3
"""Create a lightweight HTML preview from a binary 3DGS PLY file."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    * { box-sizing: border-box; }
    html, body { margin: 0; width: 100%; height: 100%; color: #1b1f24; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow: hidden; }
    body { display: flex; flex-direction: column; background: #fff; }
    header { flex: 0 0 auto; padding: 10px 16px 12px; border-bottom: 1px solid #eceff2; background: #fff; }
    h1 { margin: 0 0 10px; font-size: clamp(22px, 3vw, 36px); line-height: 1.05; letter-spacing: 0; font-weight: 760; }
    .bar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    button { border: 1px solid #d9dde2; background: #fff; color: #1b1f24; min-height: 44px; padding: 0 16px; border-radius: 14px; font-size: 18px; cursor: pointer; }
    button.active { background: #171b20; color: #fff; border-color: #171b20; }
    label { display: inline-flex; align-items: center; gap: 10px; font-size: 18px; font-weight: 650; white-space: nowrap; }
    input[type="range"] { width: min(42vw, 520px); accent-color: #111820; }
    main { position: relative; flex: 1 1 auto; min-height: 0; background: #f1f2f4; }
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
      <label>Point size <input id="pointSize" type="range" min="0.5" max="5" step="0.1" value="1.0" /></label>
    </div>
  </header>
  <main><canvas id="view"></canvas></main>
  <footer>
    <span>Drag: rotate</span>
    <span>Scroll: zoom</span>
    <span>Preview shows Gaussian centers, not full splat rendering</span>
    <span>__POINT_COUNT__ sampled centers of __TOTAL_COUNT__ Gaussians</span>
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
    let zoom = 78;
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
      if (next === "top") { yaw = 0; pitch = Math.PI / 2; zoom = 95; }
      if (next === "side") { yaw = -Math.PI / 2; pitch = 0.05; zoom = 95; }
      if (next === "perspective") { yaw = -0.72; pitch = 0.72; zoom = 78; }
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
      return { sx: w / 2 + x1 * zoom * 4.6 * persp, sy: h / 2 - y1 * zoom * 4.6 * persp, depth: z2, scale: persp };
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
        ctx.globalAlpha = 0.72;
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


def parse_header(path: Path) -> tuple[int, int, list[str]]:
    offset = 0
    vertex_count = 0
    properties: list[str] = []
    in_vertex = False
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise SystemExit("unexpected EOF before PLY header ended")
            offset += len(line)
            text = line.decode("ascii").strip()
            if text == "format binary_little_endian 1.0":
                pass
            elif text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
                in_vertex = True
            elif text.startswith("element "):
                in_vertex = False
            elif in_vertex and text.startswith("property float "):
                properties.append(text.split()[-1])
            elif text == "end_header":
                break
    if not vertex_count or not properties:
        raise SystemExit("could not parse binary Gaussian PLY header")
    return offset, vertex_count, properties


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_gaussian_centers(path: Path, max_points: int) -> tuple[list[list[float]], int]:
    offset, vertex_count, properties = parse_header(path)
    floats_per_vertex = len(properties)
    record = struct.Struct("<" + "f" * floats_per_vertex)
    indices = np.arange(vertex_count)
    if vertex_count > max_points:
        indices = np.linspace(0, vertex_count - 1, max_points, dtype=np.int64)
    wanted = set(int(i) for i in indices)
    prop_index = {name: i for i, name in enumerate(properties)}
    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    missing = [name for name in required if name not in prop_index]
    if missing:
        raise SystemExit(f"missing properties in Gaussian PLY: {', '.join(missing)}")

    points: list[list[float]] = []
    sh_c0 = 0.28209479177387814
    with path.open("rb") as handle:
        handle.seek(offset)
        for i in range(vertex_count):
            raw = handle.read(record.size)
            if i not in wanted:
                continue
            values = record.unpack(raw)
            rgb = np.array(
                [
                    values[prop_index["f_dc_0"]],
                    values[prop_index["f_dc_1"]],
                    values[prop_index["f_dc_2"]],
                ]
            )
            rgb = np.clip((rgb * sh_c0 + 0.5) * 255.0, 0, 255).astype(int)
            alpha = float(sigmoid(np.array(values[prop_index["opacity"]])))
            if alpha < 0.01:
                continue
            points.append(
                [
                    round(float(values[prop_index["x"]]), 6),
                    round(float(values[prop_index["y"]]), 6),
                    round(float(values[prop_index["z"]]), 6),
                    int(rgb[0]),
                    int(rgb[1]),
                    int(rgb[2]),
                ]
            )
    return points, vertex_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="3DGS Gaussian centers preview")
    parser.add_argument("--max-points", type=int, default=50000)
    args = parser.parse_args()

    points, total = load_gaussian_centers(args.ply, args.max_points)
    html = (
        HTML_TEMPLATE.replace("__TITLE__", args.title)
        .replace("__POINT_COUNT__", f"{len(points):,}")
        .replace("__TOTAL_COUNT__", f"{total:,}")
        .replace("__DATA_JSON__", json.dumps({"points": points}, separators=(",", ":")))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"viewer={args.out}")
    print(f"sampled_centers={len(points)} total_gaussians={total}")


if __name__ == "__main__":
    main()
