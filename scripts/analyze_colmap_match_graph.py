#!/usr/bin/env python3
"""Inspect verified COLMAP pair matches around a sequence break."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

MAX_IMAGE_ID = 2_147_483_647


def decode_pair_id(pair_id: int) -> tuple[int, int]:
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return image_id1, image_id2


def frame_index(name: str) -> int:
    return int(Path(name).stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-inliers", type=int, default=15)
    args = parser.parse_args()

    con = sqlite3.connect(args.database)
    image_names = {
        image_id: name for image_id, name in con.execute("select image_id, name from images")
    }
    rows = []
    for pair_id, inliers, config in con.execute(
        "select pair_id, rows, config from two_view_geometries where rows >= ? order by rows desc",
        (args.min_inliers,),
    ):
        image_id1, image_id2 = decode_pair_id(pair_id)
        name1 = image_names.get(image_id1, f"#{image_id1}")
        name2 = image_names.get(image_id2, f"#{image_id2}")
        idx1 = frame_index(name1)
        idx2 = frame_index(name2)
        rows.append(
            {
                "image1": name1,
                "image2": name2,
                "frame1": idx1,
                "frame2": idx2,
                "frame_gap": abs(idx2 - idx1),
                "verified_inliers": inliers,
                "config": config,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image1",
                "image2",
                "frame1",
                "frame2",
                "frame_gap",
                "verified_inliers",
                "config",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"pairs={len(rows)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
