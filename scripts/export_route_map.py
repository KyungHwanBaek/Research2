#!/usr/bin/env python3
"""Export a route-first HTML page for COLMAP camera trajectories."""

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


def load_centers(images_txt: Path) -> list[dict]:
    lines = [line.strip() for line in images_txt.read_text().splitlines() if line.strip() and not line.startswith("#")]
    rows = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        q = np.array([float(x) for x in parts[1:5]])
        t = np.array([float(x) for x in parts[5:8]])
        center = -qvec_to_rotmat(q).T @ t
        rows.append({"frame": int(Path(parts[9]).stem), "xyz": center})
    return sorted(rows, key=lambda row: row["frame"])


def project_pca(rows: list[dict]) -> list[dict]:
    xyz = np.array([row["xyz"] for row in rows], dtype=float)
    xyz -= xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz, full_matrices=False)
    xy = xyz @ vt[:2].T
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-9)
    xy = (xy - xy.min(axis=0)) / span
    return [{"frame": row["frame"], "x": round(float(p[0]), 6), "y": round(float(1 - p[1]), 6)} for row, p in zip(rows, xy)]


def downsample(points: list[dict], target: int = 900) -> list[dict]:
    if len(points) <= target:
        return points
    idx = np.linspace(0, len(points) - 1, target).round().astype(int)
    return [points[i] for i in idx]


HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>09-05 Route View</title>
  <style>
    :root { --ink:#151922; --muted:#5f6876; --line:#d9dee7; --bg:#f6f7f9; --panel:#fff; --accent:#1264d8; --red:#f04452; --blue:#3a82f7; --green:#1dbb73; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }
    header { background:#fff; border-bottom:1px solid var(--line); padding:24px 28px 18px; }
    h1 { margin:0 0 8px; font-size:30px; letter-spacing:0; }
    h2 { margin:0 0 12px; font-size:20px; letter-spacing:0; }
    p { margin:0; color:var(--muted); line-height:1.55; max-width:1080px; }
    main { max-width:1220px; margin:0 auto; padding:22px; }
    section { background:#fff; border:1px solid var(--line); border-radius:8px; padding:20px; margin-bottom:16px; }
    .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }
    .panel { border:1px solid var(--line); border-radius:8px; background:#fff; padding:14px; }
    canvas { display:block; width:100%; height:420px; border-radius:8px; background:#17212c; }
    .light canvas { background:#eef0f4; }
    .legend { display:flex; gap:12px; flex-wrap:wrap; margin-top:10px; font-size:13px; color:var(--muted); }
    .swatch { display:inline-flex; align-items:center; gap:6px; }
    .dot { width:12px; height:12px; border-radius:50%; display:inline-block; }
    .warn { border-left:4px solid #d28a12; background:#fff7e8; }
    a { color:var(--accent); font-weight:700; text-decoration:none; }
    a:hover { text-decoration:underline; }
    table { width:100%; border-collapse:collapse; font-size:14px; margin-top:10px; }
    th,td { border-bottom:1px solid var(--line); padding:8px 9px; text-align:left; vertical-align:top; }
    th { background:#f0f3f7; }
    @media (max-width:720px){ main{padding:14px} header{padding:20px 18px 14px} h1{font-size:24px} canvas{height:340px} }
  </style>
</head>
<body>
  <header>
    <h1>09-05-17-34-36 · 위에서 본 route 비교</h1>
    <p>친구의 LiDAR 결과는 metric SLAM trajectory이고, 아래 RGB/NIR COLMAP route는 image-only sparse SfM 결과다. 둘이 같은 모양으로 딱 맞지 않는 것이 현재 핵심 결과다.</p>
  </header>
  <main>
    <section class="warn">
      <h2>먼저 이렇게 보면 된다</h2>
      <p>친구 그림처럼 길 모양을 보고 싶으면 이 페이지의 아래 그림을 먼저 보면 된다. 단, COLMAP은 LiDAR처럼 바닥 기준 metric top-view를 직접 복원하지 않는다. 특히 component가 끊기면 각 component의 좌표계/scale이 따로 놀기 때문에, 내가 만든 stitched route는 보고용 근사치일 뿐이다.</p>
    </section>

    <section>
      <div class="cards">
        <div class="panel">
          <h2>친구 LiDAR route 모양</h2>
          <canvas id="lidarRef" width="720" height="520"></canvas>
          <div class="legend">
            <span class="swatch"><span class="dot" style="background:#22c96b"></span>START</span>
            <span class="swatch"><span class="dot" style="background:#ff4963"></span>current/end marker</span>
          </div>
        </div>
        <div class="panel">
          <h2>내 RGB COLMAP stitched route</h2>
          <canvas id="rgbRoute" width="720" height="520"></canvas>
          <div class="legend">
            <span>1,788 cameras · 58,411 points</span>
            <a href="viewers/rgb_colmap_0905_all_s1_360_stitched_viewer.html">3D viewer</a>
          </div>
        </div>
        <div class="panel">
          <h2>내 NIR COLMAP stitched route</h2>
          <canvas id="nirRoute" width="720" height="520"></canvas>
          <div class="legend">
            <span>1,782 cameras · 51,040 points</span>
            <a href="viewers/nir_colmap_0905_all_s1_360_stitched_viewer.html">3D viewer</a>
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2>왜 친구 것처럼 안 보이나?</h2>
      <table>
        <thead><tr><th>구분</th><th>친구 LiDAR/KISS-ICP</th><th>내 COLMAP RGB/NIR</th></tr></thead>
        <tbody>
          <tr><td>입력</td><td>LiDAR scan sequence</td><td>RGB/NIR image sequence</td></tr>
          <tr><td>경로</td><td>scan-to-map ICP로 metric trajectory 누적</td><td>이미지 feature matching으로 camera pose 추정</td></tr>
          <tr><td>끊김</td><td>약한 frame이 있어도 LiDAR geometry로 이어질 수 있음</td><td>visual overlap이 약하면 component가 분리됨</td></tr>
          <tr><td>결과 형태</td><td>위에서 본 길 + surface/mesh가 자연스러움</td><td>sparse point cloud와 local camera trajectory 중심</td></tr>
        </tbody>
      </table>
    </section>
  </main>

  <script>
    const rgb = __RGB_ROUTE__;
    const nir = __NIR_ROUTE__;

    function fit(points, w, h, pad) {
      const xs = points.map(p => p.x), ys = points.map(p => p.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const sx = (w - pad * 2) / Math.max(maxX - minX, 1e-6);
      const sy = (h - pad * 2) / Math.max(maxY - minY, 1e-6);
      const s = Math.min(sx, sy);
      return p => ({ x: pad + (p.x - minX) * s + (w - pad * 2 - (maxX - minX) * s) / 2, y: pad + (p.y - minY) * s + (h - pad * 2 - (maxY - minY) * s) / 2 });
    }

    function drawRoute(id, points, color) {
      const c = document.getElementById(id), ctx = c.getContext("2d"), w = c.width, h = c.height;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      for (let x = 40; x < w; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      for (let y = 40; y < h; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
      const map = fit(points, w, h, 38);
      ctx.lineWidth = 4;
      ctx.strokeStyle = color;
      ctx.beginPath();
      points.forEach((p, i) => { const q = map(p); if (i) ctx.lineTo(q.x, q.y); else ctx.moveTo(q.x, q.y); });
      ctx.stroke();
      const first = map(points[0]), last = map(points[points.length - 1]);
      ctx.fillStyle = "#22c96b"; ctx.beginPath(); ctx.arc(first.x, first.y, 10, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#ff4963"; ctx.beginPath(); ctx.arc(last.x, last.y, 10, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = "700 18px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"; ctx.fillText("START", first.x + 12, first.y + 6);
    }

    function drawLidarRef() {
      const c = document.getElementById("lidarRef"), ctx = c.getContext("2d"), w = c.width, h = c.height;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      for (let x = 40; x < w; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      for (let y = 40; y < h; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
      const pts = [
        {x:0.08,y:0.45},{x:0.28,y:0.38},{x:0.46,y:0.32},{x:0.50,y:0.85},
        {x:0.88,y:0.73},{x:0.76,y:0.08},{x:0.86,y:0.05},{x:0.84,y:0.13}
      ];
      const map = fit(pts, w, h, 48);
      ctx.strokeStyle = "#7f90a7"; ctx.lineWidth = 6; ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.beginPath(); pts.forEach((p,i)=>{const q=map(p); if(i) ctx.lineTo(q.x,q.y); else ctx.moveTo(q.x,q.y);}); ctx.stroke();
      const first = map(pts[0]), current = map({x:0.10,y:0.48});
      ctx.fillStyle = "#22c96b"; ctx.beginPath(); ctx.arc(first.x, first.y, 10, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#ff4963"; ctx.beginPath(); ctx.arc(current.x, current.y, 13, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = "700 18px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"; ctx.fillText("START", first.x + 12, first.y + 6);
    }

    drawLidarRef();
    drawRoute("rgbRoute", rgb, "#3a82f7");
    drawRoute("nirRoute", nir, "#b15cff");
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-text", type=Path, required=True)
    parser.add_argument("--nir-text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rgb = downsample(project_pca(load_centers(args.rgb_text / "images.txt")))
    nir = downsample(project_pca(load_centers(args.nir_text / "images.txt")))
    html = HTML.replace("__RGB_ROUTE__", json.dumps(rgb, separators=(",", ":")))
    html = html.replace("__NIR_ROUTE__", json.dumps(nir, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"route_page={args.out}")
    print(f"rgb_points={len(rgb)} nir_points={len(nir)}")


if __name__ == "__main__":
    main()
