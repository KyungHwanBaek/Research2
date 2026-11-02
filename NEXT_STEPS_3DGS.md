# Next Step: From COLMAP To 3D Gaussian Splatting

## Current Branch

The project is currently on branch 2 from the meeting discussion:

`COLMAP-based 3D geometry -> optional 3D Gaussian Splatting`

This is different from the friend's branch:

`LiDAR scans -> KISS-ICP / LiDAR SLAM -> LiDAR point-cloud map`

## What Has Been Achieved

- RGB-left-only COLMAP SfM baseline is working.
- Best sequence: `01-26-15-41-24`
- Registered image range: `000020.png` to `000045.png`
- Registered images: 26 / 48
- Sparse COLMAP points: 3,104
- Camera trajectory is estimated.
- Result viewer: `results/viewers/rgb_colmap_well_lit_outdoor_seq30_fixed_viewer.html`

## What The Next Stage Means

3DGS does not replace COLMAP in the standard pipeline. It usually uses COLMAP output as initialization:

1. COLMAP estimates camera poses.
2. COLMAP produces an initial sparse point cloud.
3. 3DGS initializes Gaussians from those sparse points.
4. 3DGS optimizes colored Gaussian blobs so that rendered views match the input images.

So the next stage is not "make another point cloud." It is:

`Use the COLMAP cameras + sparse points as 3DGS training input.`

## What I Prepared And Ran

I packaged the current best COLMAP result into a standard 3DGS scene folder:

`exports/3dgs_rgb_colmap_well_lit_outdoor_seq30_fixed`

Layout:

- `images/`: 26 registered RGB-left images
- `sparse/0/`: COLMAP binary model
- `manifest.json`: source paths and registered image list
- `README.md`: minimal training command notes

This is the practical handoff point to a 3DGS trainer.

Then I tested Mac-native 3DGS training with `msplat`, a Metal-based Apple Silicon implementation.

- Smoke test: `results/pointclouds/3dgs_msplat_smoke_100.ply`
- Main local test: `results/pointclouds/3dgs_msplat_3000_down4.ply`
- Preview viewer: `results/viewers/3dgs_msplat_3000_down4_centers_viewer.html`
- Rendered PNG views:
  - `results/renders/3dgs_msplat_3000_down4/render_cam_000.png`
  - `results/renders/3dgs_msplat_3000_down4/render_cam_013.png`
  - `results/renders/3dgs_msplat_3000_down4/render_cam_025.png`
- Render contact sheet: `results/figures/3dgs_msplat_3000_down4_render_contact_sheet.jpg`
- Novel-view fly-through:
  - `results/renders/3dgs_msplat_8000_down2_flythrough/flythrough.gif`
  - `results/renders/3dgs_msplat_8000_down2_flythrough/index.html`
  - `results/figures/3dgs_msplat_8000_down2_flythrough_contact_sheet.jpg`
  - 8,000 iterations, downscale factor 2, 351,281 final Gaussians
- Training input: 26 registered COLMAP views
- Initial COLMAP points: 3,104
- Final Gaussians after 3,000 iterations: 78,071

Important nuance: official graphdeco 3DGS is CUDA-oriented, but Mac-native alternatives exist. On this Mac, `msplat` works when run outside the managed sandbox because the Metal library needs normal macOS framework/resource access.

## Local Training Command Used

```bash
.venv/bin/msplat-train \
  --input exports/3dgs_rgb_colmap_well_lit_outdoor_seq30_fixed \
  --output results/pointclouds/3dgs_msplat_3000_down4.ply \
  --num-iters 3000 \
  --downscale-factor 4 \
  --num-downscales 0 \
  --sh-degree 3 \
  --save-every -1
```

## Local Rendering Command Used

```bash
.venv/bin/python scripts/train_msplat_and_render.py \
  --input exports/3dgs_rgb_colmap_well_lit_outdoor_seq30_fixed \
  --ply-output results/pointclouds/3dgs_msplat_3000_down4_rendered_run.ply \
  --render-dir results/renders/3dgs_msplat_3000_down4 \
  --contact-sheet results/figures/3dgs_msplat_3000_down4_render_contact_sheet.jpg \
  --metrics-out results/metrics/3dgs_msplat_3000_down4_rendered_run.json \
  --num-iters 3000 \
  --downscale-factor 4 \
  --camera-indices 0,13,25
```

Rendering means projecting the trained 3D Gaussian scene into a chosen 2D camera view. The output is an image, not another point cloud.

For the stronger fly-through, I rendered 48 camera poses interpolated between training cameras. This is closer to novel-view synthesis because many frames are not exact original camera views.

## CUDA Training Command, If Needed

On a CUDA machine with the official graphdeco implementation:

```bash
python train.py -s /path/to/exports/3dgs_rgb_colmap_well_lit_outdoor_seq30_fixed
```

Expected limitation:

- The scene has only 26 registered views.
- The trajectory covers only part of the original sequence.
- If 3DGS trains, it will likely reconstruct that local segment, not the entire campus/building area.

## What To Tell The Friend

"I chose the COLMAP branch. I now have calibrated RGB-only SfM working: 26 registered views and 3,104 sparse points. Since 3DGS usually depends on COLMAP poses, I packaged my best COLMAP result into a 3DGS-ready scene and trained a Mac-native Metal 3DGS run with msplat. It densified the scene to 78,071 Gaussians after 3,000 iterations. Your KISS-ICP LiDAR result is a different branch: denser and metric from the sensor side, while mine is RGB/COLMAP-driven and more image/color-oriented. The interesting next question is whether we compare the two branches or fuse them."

## Deeper Research Directions

- Day/night split: run separate COLMAP baselines for bright and dark sequences, then compare feature count, registration ratio, and point count.
- RGB vs NIR: use NIR images for low-light feature matching and compare against RGB.
- RGB+NIR: investigate whether NIR can improve matching in dark/night scenes.
- COLMAP vs LiDAR: compare RGB SfM trajectory shape with LiDAR SLAM trajectory, if the friend can export poses.
- Fusion: use LiDAR depth as geometric prior and RGB images for texture/color.
- 3DGS: improve the local msplat run, increase iterations/resolution, and compare rendering quality against sparse COLMAP viewer.
