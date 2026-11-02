# LiDAR-free reconstruction improvement

Completed: 2026-10-19 KST. Sequence: `sampled_540p/09-05-17-34-36`.

Open [the result gallery](results/stereo_improved_0905/index.html), [the main 3D point viewer](results/stereo_improved_0905/segment0_points.html), or [the learned novel-view fly-through](results/stereo_improved_0905/3dgs_flythrough.gif).

## What changed

The previous full-route RGB-left COLMAP run produced 58,409 sparse points in 10 disconnected models. The new pipeline uses calibrated left/right RGB and NIR images, stereo visual odometry, selected local SfM constraints, multiview depth consistency, voxel fusion and TSDF surface extraction. No LiDAR scans, LiDAR poses, or camera-to-LiDAR transforms are used.

The 1,823-frame source route is sampled every three frames for 609 tracked stereo camera poses. Fusion uses 306 of those poses, every second tracked pose. The raw image sequence and all previous reconstruction artifacts are preserved.

## Measured stereo improvement

The supplied stereo extrinsics did not align the delivered undistorted images accurately. Rotation and translation direction were refined from RGB matches in nine frames: 60, 180, 420, 600, 900, 1080, 1350, 1500, 1740. Intrinsics and baseline magnitude were retained. Frames 30, 300, 750, 1200, 1600 were reserved for this calibration comparison.

| Independent-frame diagnostic | Original calibration | Refined calibration |
|---|---:|---:|
| Mean of median vertical stereo errors | 5.126 px | 0.209 px |
| Mean confident RGB stereo pixel fraction | 9.89% | 44.77% |

These are stereo consistency/coverage measurements, not independently measured absolute 3D accuracy. Retained depth must pass left/right disparity consistency, texture and speckle checks. RGB and NIR depths are rejected where they disagree substantially. Fused points must additionally agree with depth in another camera view within a 5.5% relative tolerance.

## Geometry output

| Segment | Source frame range | Fused views | Colored points | Surface triangles |
|---|---|---:|---:|---:|
| 1 | 0–1548 | 259 | 4,885,873 | 1,063,140 |
| 2 | 1551–1641 | 16 | 214,067 | 93,587 |
| 3 | 1644–1822 | 31 | 414,790 | 149,623 |
| Total | | 306 | 5,514,730 | 1,306,350 |

Point fusion uses a 4.5 cm voxel grid and a 28 m depth limit. TSDF uses 6 cm voxels and a 24 cm truncation band. These settings describe resolution under the calibration-unit assumption; they are not accuracy guarantees. Mesh islands with fewer than 150 triangles are removed; the WebGL previews are simplified to at most 180,000 triangles per segment. Downloadable PLY meshes retain the full supported surfaces.

## Ground prior and remaining uncertainty

The selected geometry applies a soft, approximately level-ground prior in each segment, using floor planes measured from stereo depth. This reduces accumulated roll/pitch and height drift. The recovered camera-to-ground distance is approximately 1.2 m. The assumption can suppress real ramp/elevation changes, so this output should not be used to measure terrain height without independent validation.

The unconstrained poses are preserved in `experiments/stereo_improved_0905/optimized_poses_unconstrained.npz`; the full unconstrained point clouds/meshes are preserved in `experiments/stereo_improved_0905/fused_unconstrained/`. Floor-level residuals are assumption-consistency diagnostics, not ground-truth errors; see `experiments/stereo_improved_0905/ground_prior_comparison.json`.

There are still three independently located segments. Image gaps of about one second during fast turns near frames 1551 and 1644 prevent verified temporal connections. A global visual-word retrieval search tested 180 candidate pairs and found no cross-segment correspondence meeting the geometric and reverse-consistency checks. Some original COLMAP models had extremely inconsistent stereo scale and were excluded as constraints. No synthetic connections were added.

Calibration translation is interpreted as millimetres, retaining a 0.15861 m baseline. Absolute units have not been independently measured. Horizontal drift and absolute shape error remain unmeasured because there is no independent ground truth. More points and triangles establish denser coverage, not a corresponding multiplication of accuracy.

## 3DGS comparison

A local segment, frames 140–440 sampled every five frames, uses 61 rectified RGB-left images at 720×540: 53 training views and eight excluded views for rendering evaluation. Both initializations use the same images, poses, split and training settings. Poses were estimated with all source frames. Stereo initialization excludes depth from evaluation views; the sparse SfM initialization uses the ordinary full-model point cloud.

| Experiment | Excluded-view PSNR ↑ | SSIM ↑ | L1 ↓ |
|---|---:|---:|---:|
| Sparse initialization / 12k | 26.448 | 0.8302 | 0.03548 |
| Stereo initialization / 12k | 23.414 | 0.7900 | 0.03876 |
| Selected sparse initialization / 30k | 26.935 | 0.8385 | 0.03292 |

Stereo point initialization reduced rendering quality in this experiment and was not selected. The selected rendering model keeps the locally bundle-adjusted sparse initialization and trains for 30,000 iterations. It contains 422,782 Gaussians and supplies 72 interpolated camera renders. This rendering model is separate from the full-route stereo point clouds and meshes.

The local rendering gain from 12k to 30k is modest; the much larger change is dense geometry coverage. The current Metal renderer warned of per-tile overflow for some dense tiles, which can discard contributions. The exported render metrics and GIF reflect that renderer behavior. No claim is made that the selected model is globally optimal.

## Verification

- Pixel → calibrated 3D → pixel round-trip and all camera rotation matrices checked.
- Pose optimization converged and reduced its recorded objective.
- Every final point cloud contains finite positions and the expected PLY vertex count.
- Every mesh is nonempty, has valid triangle indices and matches recorded counts.
- Viewer JavaScript passes `node --check`; embedded vertex/color/index buffer lengths and indices checked.
- Saved excluded-view PNGs independently reproduce renderer PSNR within 0.03 dB, verifying the image split and ordering.
- Open3D 0.19 passed a known-plane TSDF integration test before generating meshes.
- Browser visual/interactivity testing could not run: the Browser runtime returned no available browsers. Generated PNG/JPEG/GIF previews were inspected locally.

Machine-readable results: [verification.json](results/stereo_improved_0905/verification.json), [geometry_metrics.json](results/stereo_improved_0905/geometry_metrics.json), and each rendering run's `metrics.json`.

## Reproduce

Use the existing `.venv` for stereo/tracking/optimization. The separate `.venv-mesh312` is Python 3.12 with `requirements-stereo-mesh.txt` installed. The Open3D 0.20 macOS ARM wheel returned empty TSDF geometry even for a constant-depth plane; 0.19 worked. See the [upstream issue](https://github.com/isl-org/Open3D/issues/7592).

```sh
.venv/bin/python scripts/improve_stereo_reconstruction.py --refine-calibration
.venv/bin/python scripts/improve_stereo_reconstruction.py
.venv/bin/python scripts/improve_stereo_reconstruction.py --track --stride 3
.venv/bin/python scripts/improve_stereo_reconstruction.py --loop-search
.venv/bin/python scripts/fuse_stereo_reconstruction.py --optimize
.venv/bin/python scripts/fuse_stereo_reconstruction.py --optimize --ground-prior
.venv/bin/python scripts/fuse_stereo_reconstruction.py --fuse
.venv-mesh312/bin/python scripts/mesh_stereo_surface.py
.venv/bin/python scripts/prepare_stereo_3dgs.py
.venv/bin/python scripts/train_stereo_3dgs.py --method sparse
.venv/bin/python scripts/train_stereo_3dgs.py --method stereo
.venv/bin/python scripts/train_stereo_3dgs.py --method sparse --iterations 30000 --resume results/renders/3dgs_0905_sparse_12000/checkpoint
.venv-mesh312/bin/python scripts/export_stereo_results.py
.venv/bin/python scripts/verify_stereo_results.py
.venv/bin/python scripts/write_stereo_report.py
```

Metal training requires normal macOS GPU access outside the managed sandbox on this machine. Depth/features are cached under a hash of the stereo extrinsics, preventing reuse after calibration changes.
