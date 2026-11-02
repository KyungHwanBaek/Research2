# Joint stereo reconstruction and public gallery

Updated 2026-10-19 KST. This follows `STEREO_IMPROVEMENT_REPORT.md` and preserves all prior results.

Public gallery: https://stereo-reconstruction-lab-kyunghwan.sillycod2.chatgpt.site — native deployment succeeded and the audience is confirmed `public`. Deployment receipt: `experiments/stereo_joint_ba_0905/public_deployment.json`.

## Actual reconstruction changes

- Built native calibrated two-camera rigs with PyCOLMAP 4.2.1. Each synchronized left/right pair has a fixed 0.15861 m baseline and fixed camera intrinsics.
- Matched actual left/right RGB SIFT coordinates and created temporal feature tracks. These are measured image coordinates; right-image observations are not fabricated from dense disparity.
- Jointly optimized camera poses and sparse 3D points using robust Ceres bundle adjustment on all 609 sampled frames across the three tracking segments.
- Checked poses against dense stereo pixels, which are separate from the sparse feature-coordinate optimization. Selected conservative SO(3)/translation blends using these diagnostics.
- Fused the selected poses into new RGB/NIR dense point clouds and regenerated TSDF surface meshes in the verified Open3D 0.19 environment.

## Measured geometry diagnostics

| Segment | Native BA mean feature reprojection before → after | Selected BA blend | Dense relative depth median before → selected | Dense agreement before → selected |
|---|---|---:|---|---|
| 1 | 2.319 → 0.662 px | 0.75 | 3.010% → 2.872% | 76.06% → 79.58% |
| 2 | 2.317 → 0.236 px | 1.00 | 1.966% → 1.833% | 72.31% → 74.66% |
| 3 | 2.022 → 0.443 px | 0.50 | 2.726% → 2.493% | 79.03% → 80.83% |

Dense agreement uses the same 5.5% relative-depth tolerance as the earlier pipeline. The feature reprojection column describes the full native BA experiment. Published geometry uses the selected trajectory blends, so the native minimum is not presented as the reprojection error of the blended output. Dense diagnostics were used to choose the blending weights; they are validation measurements, not an untouched final test or ground-truth accuracy measurement.

The largest segment did not improve both dense diagnostics when using 100% BA poses. A 75% blend improves both; the full candidate and all blend scores are retained in `experiments/stereo_joint_ba_0905/comparison.json`.

## New reconstruction artifacts

| Segment | Colored points | Supported mesh triangles |
|---|---:|---:|
| 1 | 5,411,117 | 1,023,115 |
| 2 | 224,437 | 97,587 |
| 3 | 428,991 | 134,762 |
| Total | 6,064,545 | 1,255,464 |

Full-resolution geometry is stored in `experiments/stereo_joint_ba_0905/fused/`; selected camera poses are in `experiments/stereo_joint_ba_0905/selected_poses.npz`.

## Rendering outcome

A separate local joint-BA experiment used the exact same 61 left images and 53/8 training/evaluation split as the prior render comparison. Native local BA reduced mean feature reprojection error from 1.693 to 0.792 px and dense median error from 6.261% to 5.645%.

However, its 30k-iteration 3DGS rendering scored PSNR 24.517 / SSIM 0.7861, below the prior sparse-initialization model's PSNR 26.935 / SSIM 0.8385. The public gallery therefore retains the better prior rendering model. Geometry diagnostics improved; rendering quality did not improve in this experiment. Both runs remain available, and the public asset builder selects the highest measured PSNR rather than assuming the newest run is better.

## Public gallery

Site source: `sites/reconstruction-lab-share/`. The static gallery provides:

- Three labeled temporal segments with source photos and capture times.
- WebGL point-cloud and surface modes, rotate/pan/zoom, and touch gestures.
- A lightweight H.264 novel-view video and original/render comparison sliders.
- Per-segment share links and downloadable sampled PLY point clouds.
- Explicit limitations and attribution to the official RGB-NIR stereo dataset.

The website displays up to 260,000 points and 100,000 triangles per segment to keep downloads small; full-resolution counts are labeled separately. It does not depend on the user's Mac or a localhost server. The Site audience is explicitly set to `public` at the user's request. Exact deployment/version information is recorded separately after native deployment verification.

## Verification and limits

All three native BA solves converged. Pose rotations, finite coordinates, binary buffer lengths, face indices, static asset links, JavaScript syntax, video codec/dimensions, and asset-size limits were checked by `scripts/verify_public_reconstruction_site.py`.

Browser visual/interactivity QA was unavailable: the Browser runtime listed no browsers. Publication remains independently verified through the native Sites deployment result. No claim is made that browser interaction was manually tested.

The route remains in three independent coordinate systems. Approximately level-ground assumptions from the previous initialization remain relevant, even after joint optimization and blending. Baseline units follow the calibration interpretation; absolute distances, terrain elevations, and global trajectory error remain unverified without independent measurements. The Metal renderer reported per-tile overflow on the new candidate, and its measured rendering score includes that behavior.

## Rerun

```sh
.venv/bin/python scripts/joint_stereo_bundle_adjustment.py
.venv/bin/python scripts/select_joint_stereo_poses.py
.venv/bin/python scripts/fuse_stereo_reconstruction.py --fuse --poses experiments/stereo_joint_ba_0905/selected_poses.npz --fusion-directory experiments/stereo_joint_ba_0905/fused
.venv-mesh312/bin/python scripts/mesh_stereo_surface.py --poses experiments/stereo_joint_ba_0905/selected_poses.npz --output experiments/stereo_joint_ba_0905/fused
.venv/bin/python scripts/joint_stereo_bundle_adjustment.py --local-render
.venv/bin/python scripts/train_stereo_3dgs.py --method joint --iterations 30000
.venv/bin/python scripts/prepare_public_reconstruction_site.py --geometry experiments/stereo_joint_ba_0905/fused
.venv/bin/python scripts/verify_public_reconstruction_site.py
```

The website uses the Sites source workflow and `.openai/hosting.json`; credentials are not stored in source. Reuse the existing project ID for future publication.
