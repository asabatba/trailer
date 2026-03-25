#!/usr/bin/env python3
"""
sweep_chunking.py — Compare distance-based vs elevation-based chunking strategies.

Runs LOO-CV across a grid of chunk sizes for both strategies and prints a summary.
"""

import sys
from pathlib import Path

import numpy as np

from model import build_dataset, loo_cv

GPX_DIR = Path("hiking_tracks")
RIDGE_ALPHA = 0.5

# Distance strategy: sizes in metres (horizontal distance per chunk)
DISTANCE_SIZES = [500, 600, 700, 800, 1000, 1500]

ELEVATION_SIZES = []  # already tested, skip


def run(strategy: str, chunk_size: float, gpx_files) -> dict:
    X, y, names = build_dataset(
        gpx_files, chunk_size_m=chunk_size, chunk_strategy=strategy
    )
    if len(y) < 3:
        return {"mae_min": float("nan"), "mape_pct": float("nan")}
    result = loo_cv(
        X, y, names, ridge_alpha=RIDGE_ALPHA, chunk_size_m=chunk_size,
        chunk_strategy=strategy,
    )
    return result


def main():
    gpx_files = sorted(GPX_DIR.glob("*.gpx"))
    if not gpx_files:
        print(f"No .gpx files in {GPX_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(gpx_files)} GPX files\n")

    results = []

    print("=" * 60)
    print("DISTANCE-BASED CHUNKING")
    print("=" * 60)
    for size in DISTANCE_SIZES:
        print(f"\n--- distance / {size} m ---")
        r = run("distance", size, gpx_files)
        results.append(("distance", size, r["mae_min"], r["mape_pct"]))

    if ELEVATION_SIZES:
        print("\n" + "=" * 60)
        print("ELEVATION-BASED CHUNKING")
        print("=" * 60)
        for size in ELEVATION_SIZES:
            print(f"\n--- elevation / {size} m ele-change ---")
            r = run("elevation", size, gpx_files)
            results.append(("elevation", size, r["mae_min"], r["mape_pct"]))

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Strategy':12s} {'ChunkSize':>10} {'MAE (min)':>10} {'MAPE (%)':>10}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")
    best = min(results, key=lambda r: r[2])
    for strat, size, mae, mape in results:
        marker = "  <-- BEST" if (strat, size) == (best[0], best[1]) else ""
        print(f"  {strat:12s} {size:>10} {mae:>10.1f} {mape:>10.1f}{marker}")


if __name__ == "__main__":
    main()
