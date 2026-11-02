# Discussion Notes

Use this as the short version for the friend discussion.

## What I Have

- My main result is RGB-left-only COLMAP SfM, not LiDAR SLAM.
- Best RGB baseline: `01-26-15-41-24`, `rgb_colmap_well_lit_outdoor_seq30_fixed`.
- It registered 26 / 48 images.
- It reconstructed 3,104 sparse 3D points.
- It estimates a red camera trajectory from visual feature matches.
- Viewer: `results/viewers/rgb_colmap_well_lit_outdoor_seq30_fixed_viewer.html`
- Stronger low-resolution variant: 29 / 48 images, 3,518 sparse points, `results/viewers/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed_component0_viewer.html`.
- Sparse mesh approximation: `results/viewers/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed_component0_sparse_delaunay_mesh_viewer.html`.
- I also packaged this result for the next 3DGS step: `exports/3dgs_rgb_colmap_well_lit_outdoor_seq30_fixed`.
- I ran a Mac-native 3DGS test with `msplat`: 3,000 iterations, 78,071 final Gaussians.
- 3DGS output: `results/pointclouds/3dgs_msplat_3000_down4.ply`
- 3DGS center preview: `results/viewers/3dgs_msplat_3000_down4_centers_viewer.html`
- I also rendered three 2D views from the trained Gaussian scene:
  - `results/renders/3dgs_msplat_3000_down4/render_cam_000.png`
  - `results/renders/3dgs_msplat_3000_down4/render_cam_013.png`
  - `results/renders/3dgs_msplat_3000_down4/render_cam_025.png`
  - Contact sheet: `results/figures/3dgs_msplat_3000_down4_render_contact_sheet.jpg`
- I then ran a stronger novel-view fly-through:
  - 8,000 iterations, downscale factor 2
  - 351,281 final Gaussians
  - 48 interpolated rendered frames
  - Gallery: `results/renders/3dgs_msplat_8000_down2_flythrough/index.html`
  - GIF: `results/renders/3dgs_msplat_8000_down2_flythrough/flythrough.gif`

## What The Friend Has

- The friend's result is likely LiDAR SLAM with KISS-ICP.
- That means LiDAR scans are aligned over time using ICP.
- The output is closer to a metric point-cloud map.
- It is expected to look denser and more stable than my RGB sparse SfM result.

## One New Comparison I Added

- I exported a raw LiDAR frame from my same best sequence.
- Frame index 26 has 126,610 valid LiDAR points.
- I sampled 50,000 points for browser viewing.
- Viewer: `results/viewers/lidar_raw_01-26-15-41-24_frame026_viewer.html`
- This is only a single raw LiDAR frame, not a SLAM-accumulated map.

## Talking Point

"My current baseline is RGB-only SfM: it finds matched visual features across images, estimates camera poses, and reconstructs sparse 3D points. Your KISS-ICP result is LiDAR SLAM: it aligns dense LiDAR scans over time. So our outputs are not directly the same category yet. I added a raw LiDAR frame viewer from the same dataset to compare the sensor density, but the next step would be accumulating those LiDAR frames with KISS-ICP or comparing RGB SfM poses against LiDAR SLAM poses."

## 3DGS Talking Point

"For the COLMAP branch, the next natural step is 3DGS because standard 3DGS uses COLMAP camera poses and sparse points as initialization. I packaged my best COLMAP scene into a 3DGS-ready folder and trained a Mac-native Metal run with msplat. The official graphdeco implementation still needs CUDA, but this Mac can run 3DGS through a Metal implementation."

## Rendering Talking Point

"Rendering means choosing a camera viewpoint and producing a 2D image from the learned 3D Gaussian scene. The Gaussian PLY is the scene representation; the rendered PNG is what the scene looks like from a camera. I rendered three views after training, so I now have both the Gaussian scene file and actual rendered images."

## Better 3DGS Talking Point

"The first renders were training-camera views, so they looked like the original photos. I pushed it further and rendered a fly-through from interpolated camera poses. That gives a better 3DGS-style result: the model is being viewed from positions between the original cameras, not only from exact photographed views."

## Good Questions To Ask

- Which exact sequence did you run through KISS-ICP?
- Did you use raw LiDAR only, or camera-LiDAR calibration too?
- How many frames/scans did your map accumulate?
- Is your output in metric scale?
- Can we run both pipelines on the same sequence and compare trajectory shape?
