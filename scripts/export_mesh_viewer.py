#!/usr/bin/env python3
"""Export a simple browser viewer for binary little-endian triangular PLY meshes."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


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
    label { display: inline-flex; align-items: center; gap: 8px; font-size: 18px; font-weight: 650; white-space: nowrap; }
    input { accent-color: #111820; }
    main { position: relative; flex: 1 1 auto; min-height: 0; background: #f1f2f4; }
    canvas { display: block; width: 100%; height: 100%; cursor: grab; }
    canvas:active { cursor: grabbing; }
    footer { flex: 0 0 auto; min-height: 42px; display: flex; align-items: center; gap: 28px; padding: 8px 16px; color: #8a9096; font-size: 18px; font-weight: 650; background: #fff; border-top: 1px solid #eceff2; flex-wrap: wrap; }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <div class="bar">
      <button id="perspective" class="active">Perspective</button>
      <button id="top">Top</button>
      <button id="side">Side</button>
      <label><input id="wire" type="checkbox" checked /> Wireframe</label>
      <label><input id="solid" type="checkbox" checked /> Solid</label>
    </div>
  </header>
  <main><canvas id="view"></canvas></main>
  <footer>
    <span>Drag: rotate</span>
    <span>Scroll: zoom</span>
    <span>__VERTEX_COUNT__ vertices</span>
    <span>__FACE_COUNT__ faces</span>
  </footer>
  <script id="mesh-data" type="application/json">__DATA_JSON__</script>
  <script>
    const data = JSON.parse(document.getElementById("mesh-data").textContent);
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const buttons = ["perspective", "top", "side"].map((id) => document.getElementById(id));
    const wire = document.getElementById("wire");
    const solid = document.getElementById("solid");
    let mode = "perspective";
    let yaw = -0.72;
    let pitch = 0.72;
    let zoom = 78;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    const verts = data.vertices.map((p) => ({ x: p[0], y: p[1], z: p[2] }));
    const faces = data.faces;
    const center = verts.reduce((a, p) => ({ x: a.x + p.x, y: a.y + p.y, z: a.z + p.z }), { x: 0, y: 0, z: 0 });
    center.x /= verts.length || 1; center.y /= verts.length || 1; center.z /= verts.length || 1;
    const ext = verts.reduce((m, p) => Math.max(m, Math.abs(p.x - center.x), Math.abs(p.y - center.y), Math.abs(p.z - center.z)), 1);

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
      const persp = mode === "perspective" ? 1 / (1.9 - z2 * 0.55) : 1;
      return { sx: canvas.clientWidth / 2 + x1 * zoom * 4.6 * persp, sy: canvas.clientHeight / 2 - y1 * zoom * 4.6 * persp, depth: z2 };
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      const pv = verts.map(project);
      const drawable = faces.map((f) => ({
        f,
        depth: (pv[f[0]].depth + pv[f[1]].depth + pv[f[2]].depth) / 3
      })).sort((a, b) => a.depth - b.depth);
      for (const item of drawable) {
        const a = pv[item.f[0]], b = pv[item.f[1]], c = pv[item.f[2]];
        ctx.beginPath();
        ctx.moveTo(a.sx, a.sy);
        ctx.lineTo(b.sx, b.sy);
        ctx.lineTo(c.sx, c.sy);
        ctx.closePath();
        if (solid.checked) {
          const shade = Math.max(105, Math.min(220, 150 + item.depth * 42));
          ctx.fillStyle = `rgba(${shade}, ${Math.max(110, shade - 12)}, ${Math.max(115, shade - 6)}, 0.46)`;
          ctx.fill();
        }
        if (wire.checked) {
          ctx.strokeStyle = "rgba(30, 35, 42, 0.28)";
          ctx.lineWidth = 0.75;
          ctx.stroke();
        }
      }
    }

    buttons.forEach((btn) => btn.addEventListener("click", () => setMode(btn.id)));
    wire.addEventListener("change", draw);
    solid.addEventListener("change", draw);
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


def parse_header(path: Path) -> tuple[int, int, int]:
    offset = 0
    vertex_count = 0
    face_count = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise SystemExit("unexpected EOF in PLY header")
            offset += len(line)
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            elif text.startswith("element face "):
                face_count = int(text.split()[-1])
            elif text == "end_header":
                break
    if not vertex_count or not face_count:
        raise SystemExit("expected binary PLY with vertex and face elements")
    return offset, vertex_count, face_count


def load_mesh(path: Path, max_faces: int) -> tuple[list[list[float]], list[list[int]], int]:
    offset, vertex_count, face_count = parse_header(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        for _ in range(vertex_count):
            x, y, z = struct.unpack("<fff", handle.read(12))
            vertices.append([round(float(x), 6), round(float(y), 6), round(float(z), 6)])
        for i in range(face_count):
            size = struct.unpack("<B", handle.read(1))[0]
            indices = list(struct.unpack("<" + "i" * size, handle.read(4 * size)))
            if size == 3 and len(faces) < max_faces:
                faces.append(indices)
            elif size > 3 and len(faces) + size - 2 <= max_faces:
                for j in range(1, size - 1):
                    faces.append([indices[0], indices[j], indices[j + 1]])
    return vertices, faces, face_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Sparse Delaunay mesh")
    parser.add_argument("--max-faces", type=int, default=20000)
    args = parser.parse_args()

    vertices, faces, total_faces = load_mesh(args.ply, args.max_faces)
    payload = {"vertices": vertices, "faces": faces}
    html = (
        HTML_TEMPLATE.replace("__TITLE__", args.title)
        .replace("__VERTEX_COUNT__", f"{len(vertices):,}")
        .replace("__FACE_COUNT__", f"{total_faces:,}")
        .replace("__DATA_JSON__", json.dumps(payload, separators=(",", ":")))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"viewer={args.out}")
    print(f"vertices={len(vertices)} faces={len(faces)} total_faces={total_faces}")


if __name__ == "__main__":
    main()
