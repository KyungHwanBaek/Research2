#!/usr/bin/env python3
"""Export a self-contained browser viewer for a COLMAP text model."""

from __future__ import annotations

import argparse
import json
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


def load_camera_centers(images_txt: Path) -> list[list[float]]:
    centers = []
    with images_txt.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            q = np.array([float(x) for x in parts[1:5]])
            t = np.array([float(x) for x in parts[5:8]])
            r = qvec_to_rotmat(q)
            center = -r.T @ t
            centers.append([round(float(x), 6) for x in center])
            handle.readline()
    return centers


def load_points(points_txt: Path, max_points: int) -> list[list[float]]:
    points = []
    with points_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            xyz_rgb = [float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])]
            points.append(
                [
                    round(float(xyz_rgb[0]), 6),
                    round(float(xyz_rgb[1]), 6),
                    round(float(xyz_rgb[2]), 6),
                    xyz_rgb[3],
                    xyz_rgb[4],
                    xyz_rgb[5],
                ]
            )
            if len(points) >= max_points:
                break
    return points


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {{
      --ink: #1b1f24;
      --muted: #7b828a;
      --line: #d9dde2;
      --panel: #f1f2f4;
      --accent: #ff3158;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; width: 100%; height: 100%; color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; overflow: hidden; }}
    body {{ display: flex; flex-direction: column; background: #fff; }}
    header {{ flex: 0 0 auto; padding: 10px 16px 12px; border-bottom: 1px solid #eceff2; background: #fff; }}
    h1 {{ margin: 0 0 10px; font-size: clamp(22px, 3vw, 36px); line-height: 1.05; letter-spacing: 0; font-weight: 760; }}
    .bar {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
    button {{ border: 1px solid var(--line); background: #fff; color: var(--ink); min-height: 44px; padding: 0 16px; border-radius: 14px; font-size: 18px; cursor: pointer; }}
    button.active {{ background: #171b20; color: #fff; border-color: #171b20; }}
    label {{ display: inline-flex; align-items: center; gap: 10px; font-size: 18px; font-weight: 650; white-space: nowrap; }}
    input[type="range"] {{ width: min(42vw, 520px); accent-color: #111820; }}
    main {{ position: relative; flex: 1 1 auto; min-height: 0; background: var(--panel); }}
    canvas {{ display: block; width: 100%; height: 100%; cursor: grab; }}
    canvas:active {{ cursor: grabbing; }}
    footer {{ flex: 0 0 auto; min-height: 42px; display: flex; align-items: center; gap: 28px; padding: 8px 16px; color: #8a9096; font-size: 18px; font-weight: 650; background: #fff; border-top: 1px solid #eceff2; flex-wrap: wrap; }}
    @media (max-width: 720px) {{
      header {{ padding: 10px; }}
      h1 {{ font-size: 22px; }}
      button {{ min-height: 38px; padding: 0 12px; border-radius: 12px; font-size: 15px; }}
      label {{ font-size: 15px; }}
      input[type="range"] {{ width: 170px; }}
      footer {{ gap: 14px; font-size: 14px; min-height: 36px; padding: 7px 10px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <div class="bar">
      <button id="perspective" class="active">Perspective</button>
      <button id="top">Top</button>
      <button id="side">Side</button>
      <label>Point size <input id="pointSize" type="range" min="0.5" max="5" step="0.1" value="1.4" /></label>
    </div>
  </header>
  <main><canvas id="view"></canvas></main>
  <footer>
    <span>Drag: rotate</span>
    <span>Scroll: zoom</span>
    <span>Red line: estimated COLMAP camera trajectory</span>
    <span>__POINT_COUNT__ points</span>
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

    const raw = data.points.map((p) => ({ x: p[0], y: p[1], z: p[2], color: `rgb(${{p[3]}},${{p[4]}},${{p[5]}})` }));
    const cams = data.cameras.map((p) => ({ x: p[0], y: p[1], z: p[2] }));
    const all = raw.concat(cams);
    const center = all.reduce((a, p) => ({{ x: a.x + p.x, y: a.y + p.y, z: a.z + p.z }}), {{ x: 0, y: 0, z: 0 }});
    center.x /= all.length || 1; center.y /= all.length || 1; center.z /= all.length || 1;
    const ext = all.reduce((m, p) => Math.max(m, Math.abs(p.x - center.x), Math.abs(p.y - center.y), Math.abs(p.z - center.z)), 1);

    function resize() {{
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }}

    function setMode(next) {{
      mode = next;
      buttons.forEach((btn) => btn.classList.toggle("active", btn.id === next));
      if (next === "top") {{ yaw = 0; pitch = Math.PI / 2; zoom = 95; }}
      if (next === "side") {{ yaw = -Math.PI / 2; pitch = 0.05; zoom = 95; }}
      if (next === "perspective") {{ yaw = -0.72; pitch = 0.72; zoom = 78; }}
      draw();
    }}

    function project(p) {{
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
      return {{ sx: w / 2 + x1 * zoom * 4.6 * persp, sy: h / 2 - y1 * zoom * 4.6 * persp, depth: z2, scale: persp }};
    }}

    function grid() {{
      const w = canvas.clientWidth, h = canvas.clientHeight;
      ctx.strokeStyle = "#d8dce1";
      ctx.lineWidth = 1;
      for (let i = -10; i <= 10; i++) {{
        const a = project({{ x: center.x + (i / 10) * ext * 2.4, y: center.y - ext * 1.05, z: center.z - ext * 1.2 }});
        const b = project({{ x: center.x + (i / 10) * ext * 2.4, y: center.y - ext * 1.05, z: center.z + ext * 1.2 }});
        const c = project({{ x: center.x - ext * 1.2, y: center.y - ext * 1.05, z: center.z + (i / 10) * ext * 2.4 }});
        const d = project({{ x: center.x + ext * 1.2, y: center.y - ext * 1.05, z: center.z + (i / 10) * ext * 2.4 }});
        ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(c.sx, c.sy); ctx.lineTo(d.sx, d.sy); ctx.stroke();
      }}
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      grid();
      const projected = raw.map((p) => ({{ ...project(p), color: p.color }})).sort((a, b) => a.depth - b.depth);
      const r = parseFloat(sizeInput.value);
      for (const p of projected) {{
        ctx.fillStyle = p.color;
        ctx.globalAlpha = 0.72;
        ctx.fillRect(p.sx, p.sy, r * p.scale, r * p.scale);
      }}
      ctx.globalAlpha = 1;
      if (cams.length) {{
        ctx.strokeStyle = "#ff3158";
        ctx.lineWidth = 2;
        ctx.beginPath();
        cams.map(project).forEach((p, i) => i ? ctx.lineTo(p.sx, p.sy) : ctx.moveTo(p.sx, p.sy));
        ctx.stroke();
        const first = project(cams[0]), last = project(cams[cams.length - 1]);
        ctx.fillStyle = "#24d870"; ctx.beginPath(); ctx.arc(first.sx, first.sy, 6, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#d72cff"; ctx.beginPath(); ctx.arc(last.sx, last.sy, 6, 0, Math.PI * 2); ctx.fill();
      }}
    }}

    buttons.forEach((btn) => btn.addEventListener("click", () => setMode(btn.id)));
    sizeInput.addEventListener("input", draw);
    canvas.addEventListener("pointerdown", (e) => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); }});
    canvas.addEventListener("pointerup", () => {{ dragging = false; }});
    canvas.addEventListener("pointermove", (e) => {{
      if (!dragging) return;
      yaw += (e.clientX - lastX) * 0.008;
      pitch = Math.max(-1.45, Math.min(1.45, pitch + (e.clientY - lastY) * 0.008));
      lastX = e.clientX; lastY = e.clientY;
      draw();
    }});
    canvas.addEventListener("wheel", (e) => {{
      e.preventDefault();
      zoom = Math.max(20, Math.min(220, zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
      draw();
    }}, {{ passive: false }});
    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text_model_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="RGB COLMAP sparse point cloud")
    parser.add_argument("--max-points", type=int, default=10000)
    args = parser.parse_args()

    points = load_points(args.text_model_dir / "points3D.txt", args.max_points)
    cameras = load_camera_centers(args.text_model_dir / "images.txt")
    payload = {"points": points, "cameras": cameras}
    template = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
    html = (
        template.replace("__TITLE__", args.title)
        .replace("__POINT_COUNT__", f"{len(points):,}")
        .replace("__DATA_JSON__", json.dumps(payload, separators=(",", ":")))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"viewer={args.out}")
    print(f"points={len(points)} cameras={len(cameras)}")


if __name__ == "__main__":
    main()
