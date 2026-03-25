#!/usr/bin/env python3
"""
sweep_smoothing.py — Find the optimal elevation smoothing window.

Tests window sizes 1 (none) through 31 against 600m distance chunks, alpha=0.5.
"""

import sys
from pathlib import Path

from model import build_dataset, loo_cv

GPX_DIR = Path("hiking_tracks")
CHUNK_SIZE = 600.0
RIDGE_ALPHA = 0.5
WINDOWS = [1, 3, 5, 7, 9, 11, 15, 21, 31]


def main():
    gpx_files = sorted(GPX_DIR.glob("*.gpx"))
    if not gpx_files:
        print(f"No .gpx files in {GPX_DIR}", file=sys.stderr)
        sys.exit(1)

    results = []
    for w in WINDOWS:
        X, y, names = build_dataset(
            gpx_files, chunk_size_m=CHUNK_SIZE, ele_smooth_window=w
        )
        r = loo_cv(
            X,
            y,
            names,
            ridge_alpha=RIDGE_ALPHA,
            chunk_size_m=CHUNK_SIZE,
            ele_smooth_window=w,
        )
        results.append((w, r["mae_min"], r["mape_pct"]))
        print(f"  window={w:3d}  MAE={r['mae_min']:.1f} min  MAPE={r['mape_pct']:.1f}%")

    print("\n--- Summary ---")
    print(f"  {'Window':>8} {'MAE (min)':>10} {'MAPE (%)':>10}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 10}")
    best = min(results, key=lambda r: r[1])
    for w, mae, mape in results:
        marker = "  <-- BEST" if w == best[0] else ""
        print(f"  {w:>8} {mae:>10.1f} {mape:>10.1f}{marker}")


if __name__ == "__main__":
    main()
