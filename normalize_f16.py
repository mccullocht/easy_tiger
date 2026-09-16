#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""Unit-normalize a BigANN-format little-endian f16 vector file in place-safe.

BigANN format: uint32 LE point count, uint32 LE dim, then dim f16 values per point.

Usage: uv run normalize_f16.py <input.fbin> <output.fbin>
"""

import sys

import numpy as np

CHUNK_POINTS = 1 << 20  # normalize ~1M points at a time


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <input.fbin> <output.fbin>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            sys.exit("input too small for BigANN header")
        n_points, dim = np.frombuffer(header, dtype="<u4")
        row_bytes = dim * 2  # f16
        size = f.seek(0, 2)
        expected = 8 + n_points * row_bytes
        if size != expected:
            sys.exit(f"file size {size} != expected {expected} for {n_points} points x {dim} dims")

        with open(out_path, "wb") as out:
            out.write(header)
            f.seek(8)
            remaining = int(n_points)
            while remaining > 0:
                count = min(CHUNK_POINTS, remaining)
                data = f.read(count * row_bytes)
                if not data:
                    break
                remaining -= count
                vecs = np.frombuffer(data, dtype="<f2").reshape(count, dim).astype(np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                np.divide(vecs, norms, out=vecs, where=norms != 0)  # leave zero vectors as-is
                out.write(vecs.astype("<f2").tobytes())

    print(f"normalized {n_points} points x {dim} dims: {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
