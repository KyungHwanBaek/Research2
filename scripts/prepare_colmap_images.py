#!/usr/bin/env python3
"""Prepare a time-ordered left-camera image sequence for COLMAP."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def find_frame_dirs(root: Path, modality: str) -> list[Path]:
    frames = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if (p / modality).is_dir():
            frames.append(p)
    return sorted(frames, key=lambda p: p.name)


def choose_left(frame: Path, modality: str) -> tuple[Path, bool] | None:
    undistorted = frame / modality / "left.png"
    distorted = frame / modality / "left_distorted.png"
    if undistorted.exists():
        return undistorted, False
    if distorted.exists():
        return distorted, True
    return None


def make_contact_sheet(images: list[Path], output: Path, thumb_width: int = 160) -> None:
    thumbs = []
    for path in images[:40]:
        img = Image.open(path).convert("RGB")
        scale = thumb_width / img.width
        thumb = img.resize((thumb_width, max(1, int(img.height * scale))))
        draw = ImageDraw.Draw(thumb)
        draw.text((4, 4), path.stem, fill=(255, 255, 255))
        thumbs.append(thumb)
    if not thumbs:
        return
    cols = 8
    rows = (len(thumbs) + cols - 1) // cols
    w = cols * thumbs[0].width
    h = rows * thumbs[0].height
    sheet = Image.new("RGB", (w, h), (20, 20, 20))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb.width
        y = (idx // cols) * thumb.height
        sheet.paste(thumb, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extracted_root", type=Path)
    parser.add_argument("output_images", type=Path)
    parser.add_argument(
        "--modality",
        choices=("rgb", "nir"),
        default="rgb",
        help="Image modality to prepare from each frame directory.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument(
        "--max-width",
        type=int,
        help="Resize prepared images to this maximum width while preserving aspect ratio.",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Apply LAB-luminance CLAHE contrast enhancement. Use only for labeled troubleshooting experiments.",
    )
    parser.add_argument(
        "--undistort-calibration",
        type=Path,
        help="calibration.npz containing mtx_left and dist_left. Required if selected images are distorted.",
    )
    parser.add_argument("--contact-sheet", type=Path, default=Path("results/figures/rgb_left_contact_sheet.jpg"))
    args = parser.parse_args()

    frames = find_frame_dirs(args.extracted_root, args.modality)
    selected_frames = frames[args.start : args.start + args.count * args.stride : args.stride]
    selected_images: list[tuple[Path, bool]] = []
    for frame in selected_frames:
        img = choose_left(frame, args.modality)
        if img is not None:
            selected_images.append(img)

    if not selected_images:
        raise SystemExit(f"No {args.modality}-left images found.")
    has_distorted = any(is_distorted for _, is_distorted in selected_images)
    calib = None
    if has_distorted:
        if args.symlink:
            raise SystemExit("Cannot symlink distorted images. Use --undistort-calibration to write corrected PNGs.")
        if args.undistort_calibration is None:
            raise SystemExit("Selected RGB-left images are distorted; pass --undistort-calibration calibration.npz.")
        import cv2

        z = np.load(args.undistort_calibration)
        calib = (z["mtx_left"], z["dist_left"])

    args.output_images.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_images.iterdir())
    if existing:
        raise SystemExit(f"Output image directory is not empty: {args.output_images}")

    prepared_paths = []
    scale_used: float | None = None
    for idx, (src, is_distorted) in enumerate(selected_images):
        dst = args.output_images / f"{idx:06d}{src.suffix.lower()}"
        if is_distorted:
            import cv2

            assert calib is not None
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            undistorted = cv2.undistort(img, calib[0], calib[1])
            if args.clahe:
                lab = cv2.cvtColor(undistorted, cv2.COLOR_BGR2LAB)
                l_chan, a_chan, b_chan = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_chan = clahe.apply(l_chan)
                undistorted = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
            if args.max_width and undistorted.shape[1] > args.max_width:
                scale = args.max_width / undistorted.shape[1]
                scale_used = scale
                new_size = (args.max_width, max(1, int(round(undistorted.shape[0] * scale))))
                undistorted = cv2.resize(undistorted, new_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), undistorted)
        elif args.symlink:
            if args.max_width:
                raise SystemExit("Cannot symlink when --max-width is set; resizing requires writing new image files.")
            os.symlink(src.resolve(), dst)
        else:
            if args.max_width:
                img = Image.open(src).convert("RGB")
                if img.width > args.max_width:
                    scale = args.max_width / img.width
                    scale_used = scale
                    new_size = (args.max_width, max(1, int(round(img.height * scale))))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                if args.clahe:
                    import cv2

                    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
                    l_chan, a_chan, b_chan = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l_chan = clahe.apply(l_chan)
                    arr = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2RGB)
                    img = Image.fromarray(arr)
                img.save(dst)
            else:
                if args.clahe:
                    import cv2

                    arr = cv2.imread(str(src), cv2.IMREAD_COLOR)
                    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
                    l_chan, a_chan, b_chan = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l_chan = clahe.apply(l_chan)
                    arr = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2RGB)
                    Image.fromarray(arr).save(dst)
                else:
                    shutil.copy2(src, dst)
        prepared_paths.append(dst)

    make_contact_sheet(prepared_paths, args.contact_sheet)
    print(f"frames_total={len(frames)}")
    print(f"modality={args.modality}")
    print(f"images_prepared={len(selected_images)}")
    print(f"first_source={selected_images[0][0]}")
    print(f"last_source={selected_images[-1][0]}")
    print(f"undistorted={has_distorted}")
    print(f"clahe={args.clahe}")
    if scale_used is not None:
        print(f"resize_scale={scale_used}")
    print(f"contact_sheet={args.contact_sheet}")


if __name__ == "__main__":
    main()
