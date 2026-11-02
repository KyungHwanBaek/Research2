#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <image_dir> <sparse_model_dir> <dense_workspace>" >&2
  exit 2
fi

IMAGE_DIR="$1"
SPARSE_MODEL="$2"
DENSE_WORKSPACE="$3"
LOG_DIR="logs"
DENSE_MAX_IMAGE_SIZE="${COLMAP_DENSE_MAX_IMAGE_SIZE:-768}"
PATCH_MATCH_MAX_IMAGE_SIZE="${COLMAP_PATCH_MATCH_MAX_IMAGE_SIZE:-768}"
PATCH_MATCH_NUM_ITERATIONS="${COLMAP_PATCH_MATCH_NUM_ITERATIONS:-3}"
PATCH_MATCH_GEOM_CONSISTENCY="${COLMAP_PATCH_MATCH_GEOM_CONSISTENCY:-1}"
DENSE_OUTPUT_PLY="${COLMAP_DENSE_OUTPUT_PLY:-results/pointclouds/rgb_colmap_dense.ply}"

mkdir -p "$DENSE_WORKSPACE" "$LOG_DIR"
mkdir -p "$(dirname "$DENSE_OUTPUT_PLY")"

colmap image_undistorter \
  --image_path "$IMAGE_DIR" \
  --input_path "$SPARSE_MODEL" \
  --output_path "$DENSE_WORKSPACE" \
  --output_type COLMAP \
  --max_image_size "$DENSE_MAX_IMAGE_SIZE" \
  2>&1 | tee "$LOG_DIR/dense_image_undistorter.log"

colmap patch_match_stereo \
  --workspace_path "$DENSE_WORKSPACE" \
  --workspace_format COLMAP \
  --PatchMatchStereo.max_image_size "$PATCH_MATCH_MAX_IMAGE_SIZE" \
  --PatchMatchStereo.num_iterations "$PATCH_MATCH_NUM_ITERATIONS" \
  --PatchMatchStereo.geom_consistency "$PATCH_MATCH_GEOM_CONSISTENCY" \
  2>&1 | tee "$LOG_DIR/dense_patch_match_stereo.log"

colmap stereo_fusion \
  --workspace_path "$DENSE_WORKSPACE" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$DENSE_OUTPUT_PLY" \
  2>&1 | tee "$LOG_DIR/dense_stereo_fusion.log"
