#!/usr/bin/env python3
"""Safely extract one tar.gz archive into an empty destination directory."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def is_safe_member(dest: Path, member_name: str) -> bool:
    target = (dest / member_name).resolve()
    return target == dest.resolve() or dest.resolve() in target.parents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()

    if not args.archive.exists():
        raise SystemExit(f"Archive not found: {args.archive}")
    args.dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.archive, "r:gz") as tf:
        members = tf.getmembers()
        top_level = {Path(m.name).parts[0] for m in members if Path(m.name).parts}
        for top in top_level:
            if (args.dest / top).exists():
                raise SystemExit(f"Refusing to overwrite existing extracted path: {args.dest / top}")
        for member in members:
            if not is_safe_member(args.dest, member.name):
                raise SystemExit(f"Unsafe tar member path: {member.name}")
        tf.extractall(args.dest)

    print(f"extracted={args.dest}")


if __name__ == "__main__":
    main()
