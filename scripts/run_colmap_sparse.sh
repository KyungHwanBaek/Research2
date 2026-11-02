#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 <image_dir> <experiment_dir> [camera_params_fx_fy_cx_cy]" >&2
  exit 2
fi

IMAGE_DIR="$1"
EXP_DIR="$2"
CAMERA_PARAMS="${3:-}"
DB="$EXP_DIR/database.db"
SPARSE="$EXP_DIR/sparse"
LOG_DIR="logs"
MAX_IMAGE_SIZE="${COLMAP_MAX_IMAGE_SIZE:--1}"
MAX_NUM_FEATURES="${COLMAP_MAX_NUM_FEATURES:-8192}"
SIFT_PEAK_THRESHOLD="${COLMAP_SIFT_PEAK_THRESHOLD:-0.00667}"
MATCHER="${COLMAP_MATCHER:-sequential}"
SEQUENTIAL_OVERLAP="${COLMAP_SEQUENTIAL_OVERLAP:-10}"
BA_REFINE_FOCAL_LENGTH="${COLMAP_BA_REFINE_FOCAL_LENGTH:-0}"
BA_REFINE_PRINCIPAL_POINT="${COLMAP_BA_REFINE_PRINCIPAL_POINT:-0}"
BA_REFINE_EXTRA_PARAMS="${COLMAP_BA_REFINE_EXTRA_PARAMS:-0}"

mkdir -p "$EXP_DIR" "$SPARSE" "$LOG_DIR"

FEATURE_ARGS=(
  feature_extractor
  --database_path "$DB"
  --image_path "$IMAGE_DIR"
  --ImageReader.single_camera 1
  --ImageReader.camera_model PINHOLE
  --FeatureExtraction.use_gpu 0
  --FeatureExtraction.max_image_size "$MAX_IMAGE_SIZE"
  --SiftExtraction.max_num_features "$MAX_NUM_FEATURES"
  --SiftExtraction.peak_threshold "$SIFT_PEAK_THRESHOLD"
)

if [ -n "$CAMERA_PARAMS" ]; then
  FEATURE_ARGS+=(--ImageReader.camera_params "$CAMERA_PARAMS")
fi

colmap "${FEATURE_ARGS[@]}" 2>&1 | tee "$LOG_DIR/$(basename "$EXP_DIR")_feature_extractor.log"

case "$MATCHER" in
  sequential)
    colmap sequential_matcher \
      --database_path "$DB" \
      --FeatureMatching.use_gpu 0 \
      --SiftMatching.cpu_brute_force_matcher 1 \
      --SequentialMatching.overlap "$SEQUENTIAL_OVERLAP" \
      2>&1 | tee "$LOG_DIR/$(basename "$EXP_DIR")_sequential_matcher.log"
    ;;
  exhaustive)
    colmap exhaustive_matcher \
      --database_path "$DB" \
      --FeatureMatching.use_gpu 0 \
      --SiftMatching.cpu_brute_force_matcher 1 \
      2>&1 | tee "$LOG_DIR/$(basename "$EXP_DIR")_exhaustive_matcher.log"
    ;;
  *)
    echo "unsupported COLMAP_MATCHER=$MATCHER; use sequential or exhaustive" >&2
    exit 2
    ;;
esac

colmap mapper \
  --database_path "$DB" \
  --image_path "$IMAGE_DIR" \
  --output_path "$SPARSE" \
  --Mapper.ba_refine_focal_length "$BA_REFINE_FOCAL_LENGTH" \
  --Mapper.ba_refine_principal_point "$BA_REFINE_PRINCIPAL_POINT" \
  --Mapper.ba_refine_extra_params "$BA_REFINE_EXTRA_PARAMS" \
  2>&1 | tee "$LOG_DIR/$(basename "$EXP_DIR")_mapper.log"

if [ -d "$SPARSE/0" ]; then
  mkdir -p "$EXP_DIR/sparse_text"
  colmap model_converter \
    --input_path "$SPARSE/0" \
    --output_path "$EXP_DIR/sparse_text" \
    --output_type TXT \
    2>&1 | tee "$LOG_DIR/$(basename "$EXP_DIR")_model_converter.log"

  colmap model_converter \
    --input_path "$SPARSE/0" \
    --output_path "results/pointclouds/$(basename "$EXP_DIR")_sparse.ply" \
    --output_type PLY \
    2>&1 | tee "$LOG_DIR/$(basename "$EXP_DIR")_ply_export.log"
fi
