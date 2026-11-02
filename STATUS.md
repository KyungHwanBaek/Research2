# Status

Last updated: 2026-10-21 KST.

## Latest: student union RGB/COLMAP + LiDAR reconstruction

- Correct target: `01-26-15-41-24`, 48 frames, courtyard → stairs/ramp → entrance labeled Student Union Bldg.
- Friend's completed map/trajectory is not available. Built a map directly from the dataset's 48 raw LiDAR scans.
- Compared KISS-ICP, COLMAP metric alignment, LiDAR-depth PnP, multiscale ICP, FPFH/RANSAC rescue, pose-graph optimization, and joint COLMAP landmark/camera + LiDAR-plane refinement. Also tested near-range-excluded re-registration; its score was effectively identical to the selected pose graph.
- Selected geometry: 48 camera poses; 232,136 colored points; 1,648,179 total LiDAR points; 2,120,479 Poisson mesh triangles. Near-sensor returns within 3m are excluded from final full-map integration/Poisson meshing after detecting false occluders by ray-cast visual inspection.
- Identical 37-pair cross-scan checks on returns excluded from ICP: original metric COLMAP overlap 71.71% → selected pose graph 77.81%; matched tangent-plane median 5.45cm → 2.69cm. These are consistency diagnostics, not survey ground truth.
- Joint camera/point refinement converged; reserved reprojection median 4.03 → 0.52px, but LiDAR consistency worsened. Kept it as an appearance trial rather than the final structural map.
- Compared 12k-iteration Gaussian models: original COLMAP PSNR 23.281 / SSIM .7181; joint poses + sparse points 22.484 / .6992; joint poses + LiDAR points 15.778 / .5831. Extended COLMAP to 30k: 23.169 / .7179, so retain the 12k model. Different scene/split from prior 09-05 result; do not compare PSNR across them.
- Published and verified Sites deployment `succeeded`, version 2: https://stereo-reconstruction-lab-kyunghwan.sillycod2.chatgpt.site/student-hall.html . Public audience preserved. Native deployment receipt: `experiments/student_hall_lidar/public_deployment.json`.
- Full methods, raw-data provenance, commands, caveats: `STUDENT_HALL_LIDAR_REPORT.md`. Final checks: `experiments/student_hall_lidar/verification.json`.

## Latest: joint stereo BA and publicly shared gallery

- Public deployment verified successful: https://stereo-reconstruction-lab-kyunghwan.sillycod2.chatgpt.site
- Site audience is `public`; visitors do not need a localhost server or the owner's Mac.
- Added native PyCOLMAP/Ceres stereo-rig bundle adjustment on 609 frames and selected conservative trajectory blends using separate dense-depth diagnostics. All three solves converged.
- Main-segment dense median relative error: 3.010% → 2.872%; depth agreement: 76.06% → 79.58%.
- Regenerated geometry: 6,064,545 colored points and 1,255,464 mesh triangles across three independent segments.
- The new local joint-BA 3DGS candidate scored PSNR 24.517, below the prior 26.935 model, so the public gallery retains the better prior rendering.
- Added compact WebGL point/mesh exploration, original-photo filmstrips, video, comparison sliders and per-segment sharing. Static files/geometry buffers/video validated; browser visual QA unavailable.
- Method, limitations and rerun commands: `JOINT_STEREO_PUBLIC_REPORT.md`. Verified deployment receipt: `experiments/stereo_joint_ba_0905/public_deployment.json`.

## Previous: LiDAR-free stereo improvement

- Completed a new RGB/NIR stereo reconstruction of the full `09-05-17-34-36` route, with 609 sampled camera poses in three verified tracking segments.
- Refined stereo rotation/translation direction using nine RGB frames. Independent-frame vertical stereo error decreased from 5.126 px to 0.209 px; confident RGB stereo coverage increased from 9.89% to 44.77%.
- Added validated local SfM constraints, multiview depth consistency, dense colored point fusion, and TSDF surface meshes. The selected geometry includes an approximately level-ground prior to reduce height drift. Unconstrained results are preserved separately.
- Trained and evaluated matched 3DGS initializations on 61 local views (53 training / 8 excluded). Sparse initialization outperformed stereo initialization: PSNR 26.448 vs 23.414 at 12k iterations. Selected sparse initialization trained to 30k: PSNR 26.935, SSIM 0.8385.
- Final gallery: `results/stereo_improved_0905/index.html`.
- Full outcome, artifact counts, limitations, verification and rerun commands: `STEREO_IMPROVEMENT_REPORT.md`.
- No LiDAR data or poses were used. Absolute geometry accuracy and cross-segment alignment remain unverified. The older sections below document historical baselines.

## Done

- Parsed the project request and inspected the empty workspace.
- Created project structure, scripts, `.gitignore`, `README.md`, `STATUS.md`, and `PROJECT_NOTES.md`.
- Captured hardware/software environment:
  - Apple Silicon MacBook Pro, M4, 24 GB RAM.
  - macOS 26.6.2, arm64.
  - No NVIDIA GPU or CUDA.
  - About 210 GiB free disk space at start.
- Installed project Python dependencies in `.venv`.
- Installed COLMAP 4.2.1_1 via Homebrew.
- Read official GitHub README and Hugging Face file list.
- Downloaded only selected official root archives:
  - `01-26-15-36-18.tar.gz`: 16 frames, preserved but too short.
  - `01-24-08-49-46.tar.gz`: 90 frames, used for dark-indoor pilot.
  - `01-26-15-41-24.tar.gz`: 48 frames, official `well_lit_outdoor` sequence used for improved RGB baseline.
  - `01-26-15-34-37.tar.gz`: 26 frames, official `well_lit_outdoor` sequence tried as one additional candidate.
- Inspected extracted tree, `storage.hdf5`, `calibration.npz`, and `cam2lidar.npy`.
- Prepared undistorted RGB-left COLMAP inputs using official `calibration.npz`.
- Extended `scripts/prepare_colmap_images.py` with `--modality rgb|nir`, so the same COLMAP preparation flow can now run on NIR-left as well as RGB-left.
- Ran COLMAP sparse experiments:
  - `rgb_colmap_pilot`: sequential, 3/90 registered, 46 sparse points.
  - `rgb_colmap_tail`: sequential, failed to create model.
  - `rgb_colmap_tail_exhaustive`: exhaustive, 11/42 registered, 779 sparse points.
  - `rgb_colmap_well_lit_outdoor_fast`: sequential on better-lit official sequence, best component has 26/48 registered, 3,071 sparse points.
  - `rgb_colmap_well_lit_outdoor_768_seq30_fixed`: resized 768px input, sequential overlap 30, fixed intrinsics, 26/48 registered, 2,429 sparse points, lower reprojection error.
  - `rgb_colmap_well_lit_outdoor_768_exhaustive_fixed`: resized 768px input, exhaustive matching, fixed intrinsics, 29/48 registered, 3,518 sparse points.
  - `rgb_colmap_well_lit_outdoor_seq30_fixed`: full-resolution input, sequential overlap 30, fixed intrinsics, 26/48 registered, 3,104 sparse points.
  - `rgb_colmap_well_lit_outdoor_153437_seq30_fixed`: additional full-resolution well-lit candidate, sequential overlap 30, fixed intrinsics, 3/26 registered, 111 sparse points.
  - `rgb_colmap_well_lit_outdoor_153437_768_exhaustive_fixed`: same additional candidate at 768px with exhaustive matching, 12/26 registered, 2,573 sparse points.
- Ran a labeled troubleshooting experiment:
  - `rgb_colmap_well_lit_outdoor_768_clahe_seq30_fixed`: CLAHE contrast-enhanced 768px input, sequential overlap 30, fixed intrinsics, 26/48 registered, 3,146 sparse points.
- Exported JSON metrics, PLY point clouds, contact sheets, sparse overview PNGs, and HTML viewers.
- Exported one raw LiDAR reference frame from the same best sequence for modality comparison:
  - Frame index: 26 (`15_43_03_838`)
  - Valid LiDAR points in the raw frame: 126,610
  - Browser sample: 50,000 points
  - Viewer: `results/viewers/lidar_raw_01-26-15-41-24_frame026_viewer.html`
  - PLY: `results/pointclouds/lidar_raw_01-26-15-41-24_frame026_sampled.ply`
  - Metrics: `results/metrics/lidar_raw_01-26-15-41-24_frame026.json`
- Packaged the current best COLMAP result as a 3DGS-ready scene:
  - Export folder: `exports/3dgs_rgb_colmap_well_lit_outdoor_seq30_fixed`
  - Registered images copied: 26 (`000020.png` to `000045.png`)
  - COLMAP binary model copied to `sparse/0`
  - Notes: `NEXT_STEPS_3DGS.md`
- Installed `msplat` 1.1.4 and ran local Apple Silicon/Metal 3DGS training:
  - Smoke run: 100 iterations, output `results/pointclouds/3dgs_msplat_smoke_100.ply`
  - Main local run: 3,000 iterations, output `results/pointclouds/3dgs_msplat_3000_down4.ply`
  - Initial Gaussians: 3,104
  - Final Gaussians: 78,071
  - Preview viewer: `results/viewers/3dgs_msplat_3000_down4_centers_viewer.html`
- Rendered selected 3DGS camera views from a 3,000-iteration `msplat` run:
  - Rendered images: `results/renders/3dgs_msplat_3000_down4/render_cam_000.png`, `render_cam_013.png`, `render_cam_025.png`
  - Contact sheet: `results/figures/3dgs_msplat_3000_down4_render_contact_sheet.jpg`
  - Metrics: `results/metrics/3dgs_msplat_3000_down4_rendered_run.json`
  - Final Gaussians in rendered run: 76,522
- Ran a longer 3DGS novel-view fly-through:
  - 8,000 iterations, downscale factor 2
  - Final Gaussians: 351,281
  - Rendered 48 interpolated camera-pose frames at 720 x 540
  - GIF: `results/renders/3dgs_msplat_8000_down2_flythrough/flythrough.gif`
  - Gallery: `results/renders/3dgs_msplat_8000_down2_flythrough/index.html`
  - Contact sheet: `results/figures/3dgs_msplat_8000_down2_flythrough_contact_sheet.jpg`
  - Metrics: `results/metrics/3dgs_msplat_8000_down2_flythrough.json`
- Exported match-graph diagnostics and baseline comparison:
  - `results/metrics/rgb_colmap_well_lit_outdoor_seq30_fixed_match_graph.csv`
  - `results/metrics/rgb_colmap_well_lit_outdoor_768_clahe_seq30_fixed_match_graph.csv`
  - `results/metrics/baseline_comparison.csv`
- Downloaded and extracted the same sampled-540p sequence used by the friend's LiDAR/KISS-ICP result:
  - Archive: `data/raw/sampled_540p/09-05-17-34-36.tar.gz`
  - SHA256: `c9b47cff1f6fb16f4d450a406ba82da661e5a1bfde9854c4584f62a3309cf24f`
  - Extracted location: `data/extracted/sampled_540p/09-05-17-34-36`
  - Frame count: 1,823
  - Frame range: `17_34_37_727` to `17_39_18_146`
  - RGB-left image size: 720 x 540
- Prepared three RGB-left COLMAP subsets from `09-05-17-34-36`:
  - Start: start 0, stride 5, 120 images
  - Mid: start 600, stride 5, 120 images
  - Tail: start 1200, stride 5, 120 images
- Ran same-sequence COLMAP sparse experiments:
  - `rgb_colmap_0905_start_s5_120_seq30_fixed` component 0: 89/120 registered, 17,619 sparse points.
  - `rgb_colmap_0905_start_s5_120_seq30_fixed` component 1: 14/120 registered, 1,905 sparse points.
  - `rgb_colmap_0905_mid_s5_120_seq30_fixed` component 0: 66/120 registered, 21,408 sparse points.
  - `rgb_colmap_0905_mid_s5_120_seq30_fixed` component 1: 54/120 registered, 13,756 sparse points.
  - Tail high-quality matching and reduced tail matching both hit repeated long CPU stalls and were stopped.
- Rechecked the same sequence after noticing the start-subset trajectory looked too straight compared with the friend's LiDAR top-view route:
  - `rgb_colmap_0905_full_s10_480_seq20_fixed`: almost-full-route sample, 183 images, sequential matching, split into 5 components.
  - `rgb_colmap_0905_full_s5_360_seq20_fixed`: almost-full-route sample, 365 images, sequential matching, split into 9 components. Largest component: 89/365 registered, source frames 30..470.
  - `rgb_colmap_0905_full_s5_360_exhaustive_fixed`: same 365-image input, exhaustive matching, split into 7 components. Largest component: 110/365 registered, source frames 1145..1765.
  - Conclusion: the earlier 1-like trajectory was mainly because it used the start subset. With the full route, RGB/COLMAP still does not make one global map; it reconstructs several local pieces.
- Ran the corrected all-RGB-frame version for the friend's `09-05-17-34-36` sequence:
  - Prepared all 1,823 RGB-left frames, not a stride-5 subset.
  - Input resized to 360 x 270 with fixed scaled calibration intrinsics.
  - Sequential matching with overlap 20 completed over the full image set.
  - COLMAP mapper registered 1,821 / 1,823 frames across 10 disconnected components.
  - Largest component: 444 frames, source frames 28..471, 12,098 sparse points.
  - Other large components: 356 frames at 1143..1498, 313 frames at 556..868, 270 frames at 867..1136.
  - Component index: `results/viewers/rgb_colmap_0905_all_s1_360_seq20_fixed_index.html`
- Ran NIR-left COLMAP diagnostics on the two key sequences:
  - `nir_colmap_0126_154124_768_exhaustive_fixed`: all 48 NIR-left frames from `01-26-15-41-24`, resized to 768 px, exhaustive matching, fixed intrinsics. Result: 2 components; best component 26/48 registered, 2,207 sparse points; second component 14/48 registered, 1,897 sparse points.
  - `nir_colmap_0905_all_s1_360_seq20_fixed`: all 1,823 NIR-left frames from `09-05-17-34-36`, resized to 360 x 270, sequential overlap 20, fixed intrinsics. Result: 1,782 unique frames registered across 10 disconnected components; largest component 545 frames at source frames 1142..1766, 19,568 sparse points.
- Exported NIR metrics, PLYs, overview PNGs, and HTML viewers:
  - `results/viewers/nir_colmap_0905_all_s1_360_seq20_fixed_index.html`
  - `results/viewers/nir_colmap_0905_all_s1_360_seq20_fixed_component0_viewer.html` through `component9_viewer.html`
  - `results/viewers/nir_colmap_0126_154124_768_exhaustive_fixed_component0_viewer.html`
  - `results/viewers/nir_colmap_0126_154124_768_exhaustive_fixed_component1_viewer.html`
  - Comparison table: `results/metrics/rgb_nir_colmap_comparison.csv`
- Tried to merge disconnected same-sequence components:
  - Official COLMAP `model_merger` succeeded for some overlapping clusters:
    - RGB `(component9 + component5) + component4`, loose threshold: 267 images, 11,740 points.
    - NIR `component1 + component7`: 569 images, 21,218 points.
    - NIR `component4 + component3`: 444 images, 12,221 points.
  - Official `model_merger` rejected weak bridges such as RGB `component1 + component0` and NIR `component0 + component2`, even with loose reprojection threshold.
- Created route-stitched approximate visualizations from all components:
  - Script: `scripts/stitch_colmap_components.py`
  - Index: `results/viewers/stitched_colmap_0905_index.html`
  - RGB stitched viewer: `results/viewers/rgb_colmap_0905_all_s1_360_stitched_viewer.html`
  - NIR stitched viewer: `results/viewers/nir_colmap_0905_all_s1_360_stitched_viewer.html`
  - RGB stitched PLY: `results/pointclouds/rgb_colmap_0905_all_s1_360_stitched_sparse.ply`
  - NIR stitched PLY: `results/pointclouds/nir_colmap_0905_all_s1_360_stitched_sparse.ply`
  - Report: `results/metrics/stitched_colmap_0905_report.json`
- Exported same-sequence comparison outputs:
  - Comparison table: `results/metrics/rgb_colmap_0905_comparison.csv`
  - Best viewer: `results/viewers/rgb_colmap_0905_start_s5_120_seq30_fixed_component0_viewer.html`
  - Best PLY: `results/pointclouds/rgb_colmap_0905_start_s5_120_seq30_fixed_component0_sparse.ply`
  - Best overview PNG: `results/figures/rgb_colmap_0905_start_s5_120_seq30_fixed_component0_sparse_overview.png`
  - Sparse Delaunay mesh: `results/meshes/rgb_colmap_0905_start_s5_120_seq30_fixed_component0_sparse_delaunay.ply`
  - Sparse Delaunay mesh viewer: `results/viewers/rgb_colmap_0905_start_s5_120_seq30_fixed_component0_sparse_delaunay_mesh_viewer.html`
  - Mid component viewers:
    - `results/viewers/rgb_colmap_0905_mid_s5_120_seq30_fixed_component0_viewer.html`
    - `results/viewers/rgb_colmap_0905_mid_s5_120_seq30_fixed_component1_viewer.html`
  - Full-route sequential component viewers:
    - `results/viewers/rgb_colmap_0905_full_s5_360_seq20_fixed_component0_viewer.html` through `component8_viewer.html`
  - Full-route exhaustive component viewers:
    - `results/viewers/rgb_colmap_0905_full_s5_360_exhaustive_fixed_component0_viewer.html` through `component6_viewer.html`
  - All-RGB-frame component viewers:
    - `results/viewers/rgb_colmap_0905_all_s1_360_seq20_fixed_component0_viewer.html` through `component9_viewer.html`
- Exported a raw LiDAR reference frame from the same `09-05-17-34-36` sequence:
  - Frame index: 300 (`17_35_18_394`)
  - Valid LiDAR points: 75,345
  - Browser sample: 50,000 points
  - Viewer: `results/viewers/lidar_raw_09-05-17-34-36_frame300_viewer.html`
  - PLY: `results/pointclouds/lidar_raw_09-05-17-34-36_frame300_sampled.ply`
  - Metrics: `results/metrics/lidar_raw_09-05-17-34-36_frame300.json`
- Created sparse mesh outputs from COLMAP sparse reconstructions:
  - Full-resolution current-best Delaunay mesh: `results/meshes/rgb_colmap_well_lit_outdoor_seq30_fixed_sparse_delaunay.ply`
  - Full-resolution mesh viewer: `results/viewers/rgb_colmap_well_lit_outdoor_seq30_fixed_sparse_delaunay_mesh_viewer.html`
  - Strengthened 768 exhaustive Delaunay mesh: `results/meshes/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed_component0_sparse_delaunay.ply`
  - Strengthened 768 mesh viewer: `results/viewers/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed_component0_sparse_delaunay_mesh_viewer.html`
  - Poisson meshing from sparse PLY failed because the sparse PLY has no normals.
- Tried a small COLMAP dense run on the stable registered range. `image_undistorter` succeeded, but `patch_match_stereo` failed because this COLMAP build requires CUDA for dense stereo and the Mac has no CUDA GPU.

## Historical RGB-left-only Baseline

`rgb_colmap_well_lit_outdoor_seq30_fixed` component 0 is the current best RGB-left-only SfM baseline:

- Registered images: 26 / 48
- Registered frame range: `000020.png` to `000045.png`
- Registration ratio: 0.5417
- Sparse points: 3,104
- Mean reprojection error: 0.4790 px
- Metrics: `results/metrics/rgb_colmap_well_lit_outdoor_seq30_fixed_component0.json`
- PLY: `results/pointclouds/rgb_colmap_well_lit_outdoor_seq30_fixed_component0_sparse.ply`
- Figure: `results/figures/rgb_colmap_well_lit_outdoor_seq30_fixed_sparse_overview.png`
- Interactive viewer: `results/viewers/rgb_colmap_well_lit_outdoor_seq30_fixed_viewer.html`

This keeps the same registered-frame count as the previous well-lit baseline but improves sparse point count and mean reprojection error while keeping the camera intrinsics fixed to calibration. It remains a monocular COLMAP SfM result with arbitrary scale and should not be described as a metric LiDAR-style map.

## Same-Sequence LiDAR Comparison Baseline

`rgb_colmap_0905_all_s1_360_seq20_fixed` is the cleanest RGB/COLMAP comparison for the friend's `09-05-17-34-36` LiDAR sequence because it uses every RGB-left frame from the same route:

- Input images: 1,823 / 1,823 RGB-left frames
- Registered images across all components: 1,821 / 1,823
- Component count: 10 disconnected sparse components
- Largest component: 444 registered frames, source frames 28..471, 12,098 sparse points
- Total sparse points across components: 58,409
- PLY point clouds: `results/pointclouds/rgb_colmap_0905_all_s1_360_seq20_fixed_component0_sparse.ply` through `component9_sparse.ply`
- Component index: `results/viewers/rgb_colmap_0905_all_s1_360_seq20_fixed_index.html`
- Component viewers: `results/viewers/rgb_colmap_0905_all_s1_360_seq20_fixed_component0_viewer.html` through `component9_viewer.html`

This should be described as an all-frame RGB-only sparse SfM reconstruction on the same sequence as the LiDAR result. It is not dense reconstruction and not a metric LiDAR map. The key finding is that RGB/COLMAP tracks almost every frame locally, but the route is split into disconnected components instead of one global trajectory.

`rgb_colmap_0905_start_s5_120_seq30_fixed` component 0 remains the strongest single local high-point RGB/COLMAP component:

- Registered images: 89 / 120
- Sampling: source frames 0..595 with stride 5
- Source frame range: `17_34_37_727` to `17_35_59_852`
- Registration ratio: 0.7417
- Sparse points: 17,619
- Mean reprojection error: 0.5673 px
- Metrics: `results/metrics/rgb_colmap_0905_start_s5_120_seq30_fixed_component0.json`
- Comparison table: `results/metrics/rgb_colmap_0905_comparison.csv`
- PLY: `results/pointclouds/rgb_colmap_0905_start_s5_120_seq30_fixed_component0_sparse.ply`
- Figure: `results/figures/rgb_colmap_0905_start_s5_120_seq30_fixed_component0_sparse_overview.png`
- Interactive viewer: `results/viewers/rgb_colmap_0905_start_s5_120_seq30_fixed_component0_viewer.html`
- Sparse mesh viewer: `results/viewers/rgb_colmap_0905_start_s5_120_seq30_fixed_component0_sparse_delaunay_mesh_viewer.html`

This local component should not be presented as the full route because it only covers source frames 0..595.

For whole-route comparison, the better evidence is the full-route rerun:

- `rgb_colmap_0905_all_s1_360_seq20_fixed`
- Input: all 1,823 RGB-left frames across the full route
- Result: 10 disconnected sparse components, not one global COLMAP trajectory
- Largest component: 444 / 1,823 registered, source frames 28..471, 12,098 sparse points
- Component index: `results/viewers/rgb_colmap_0905_all_s1_360_seq20_fixed_index.html`

This is the honest comparison point with the friend's bent LiDAR route: LiDAR SLAM can integrate scans along the route, but RGB-only COLMAP currently loses global visual continuity and breaks the route into local reconstructions.

NIR-left was tested on the same two sequences. It does not solve global image-only reconstruction by itself:

- On `01-26-15-41-24`, NIR-only is weaker than the best RGB 768 exhaustive result: best NIR component has 26/48 registered and 2,207 points, while best RGB 768 exhaustive has 29/48 registered and 3,518 points.
- On `09-05-17-34-36`, NIR registers 1,782 unique frames across 10 components. It is not globally connected, but its largest component covers frames 1142..1766 with 545 registered frames and 19,568 points, which is larger than any single RGB component on the same sequence.

This makes NIR useful as complementary evidence, especially for late-route local structure, but not as a replacement for LiDAR/global pose fusion.

For presentation, there are now route-stitched single-viewer outputs:

- RGB stitched: 1,788 cameras, 58,411 points.
- NIR stitched: 1,782 cameras, 51,040 points.

These are approximate visualizations created by aligning disconnected components. They are not official single-component COLMAP reconstructions and should not be reported as metric SLAM maps.

## Stronger Low-Resolution Variant

`rgb_colmap_well_lit_outdoor_768_exhaustive_fixed` component 0 is the strongest registered-count RGB-only variant:

- Registered images: 29 / 48
- Registered images: `000000.png`, `000001.png`, `000002.png`, and `000020.png` to `000045.png`
- Registration ratio: 0.6042
- Sparse points: 3,518
- Mean reprojection error: 0.3645 px
- Metrics: `results/metrics/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed_component0.json`
- PLY: `results/pointclouds/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed_sparse.ply`
- Figure: `results/figures/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed_component0_sparse_overview.png`
- Interactive viewer: `results/viewers/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed_component0_viewer.html`
- Sparse Delaunay mesh viewer: `results/viewers/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed_component0_sparse_delaunay_mesh_viewer.html`

This is not the official full-resolution baseline because it uses resized images, but it shows that exhaustive matching can attach a few early frames to the main reconstruction.

## Diagnosis

- The first main RGB sequence was very dark for many frames.
- Early dark-indoor frames often had near-zero SIFT features.
- The official well-lit outdoor sequence gives substantially better registration.
- Wider sequential matching improves robustness slightly but still leaves the sequence split into disconnected components.
- Match-graph diagnosis shows the main break is between the early component ending near `000016.png` and the current best component starting at `000020.png`.
- CLAHE contrast enhancement increases feature density and point count in a troubleshooting branch, but it does not connect the early and later components.
- A second well-lit outdoor archive, `01-26-15-34-37`, was tried as a sanity check. It did not beat the current best: full-resolution sequential matching only registered 3/26 frames, and 768px exhaustive matching registered 12/26 frames with higher reprojection error.
- A high-feature full-resolution exhaustive run was started, but it was too slow on CPU and was stopped. The practical variant is the 768px exhaustive run, which improved registered count to 29/48.
- For `09-05-17-34-36`, the corrected all-frame RGB run used all 1,823 RGB-left frames. COLMAP registered 1,821 frames across 10 components, so the problem is not that RGB frames were skipped; the problem is that the route is not globally stitched into one component.
- The all-frame NIR run on `09-05-17-34-36` also split into 10 components. NIR improved one late-route component but did not remove the image-only global-connectivity limitation.
- COLMAP dense reconstruction is blocked on this Mac because `patch_match_stereo` requires CUDA in the installed Homebrew COLMAP build.
- COLMAP sparse Delaunay meshing works locally, but it should be described as a sparse mesh approximation, not a true dense surface reconstruction.
- The LiDAR reference viewer is intentionally a single raw LiDAR frame, not a KISS-ICP/SLAM map. A friend using LiDAR SLAM is solving a different problem: sensor-frame LiDAR scans plus odometry/ICP accumulation, while this baseline estimates sparse scene geometry from RGB image correspondences.
- Official graphdeco 3DGS is blocked locally by the lack of CUDA, but Mac-native 3DGS via `msplat` works on this Apple Silicon Mac when run outside the managed sandbox.
- 3DGS rendering is confirmed locally: trained Gaussian scenes can be rendered back to 2D PNG views using `GaussianTrainer.render`.
- Novel-view rendering is also confirmed locally using interpolated camera poses and `GaussianTrainer.render_from_pose`.

## Next

1. Use the current best official RGB baseline for reporting.
2. For the friend's `09-05-17-34-36` sequence, present RGB and NIR COLMAP as local sparse SfM reconstructions, not as one fused metric map.
3. Use the NIR result as a complementary late-route component and compare it against the friend's LiDAR/KISS-ICP trajectory.
4. If a true global map is needed, use LiDAR/KISS-ICP poses or another global pose source as the backbone, then attach RGB/NIR appearance or local COLMAP geometry to it.
5. For a richer 3D-looking result on this Mac, use an alternative CPU/Metal-capable method or move dense reconstruction to a CUDA machine.

## Notes

- `git init` failed under the managed sandbox with `.git: Operation not permitted`; no remote push was attempted.
- Homebrew linked `/opt/homebrew/bin/python3` to Python 3.14 during COLMAP install; `.venv` remains Python 3.13.3.
- Matplotlib needed `Agg` backend and local cache dirs for headless PNG generation.
