# Project Notes

Research direction: adaptive 3D scene reconstruction from heterogeneous multi-modal visual data.

Immediate baseline:

```text
RGB left sequence -> COLMAP -> camera poses -> sparse point cloud -> optional dense -> optional 3DGS dataset preparation
```

Friend's LiDAR/KISS-ICP result is intentionally independent for now. Later comparison with COLMAP should use a Sim(3) alignment because monocular RGB SfM generally lacks metric scale, while LiDAR has metric scale.
