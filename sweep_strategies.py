#!/usr/bin/env python3
"""
sweep_strategies.py — Evaluate four new chunking strategies against the
distance-600m baseline.

Strategies tested
─────────────────
  direction   : split at ascent/descent reversals
  grade_band  : split when terrain category changes
  fixed_count : always N equal-distance chunks per route
  multiscale  : 600m base features + 200m distribution stats (25 features)

All use two-stage LOO-CV (physics OLS → Ridge residuals), ridge_alpha=0.5.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from gpx_features import (
    FEATURE_NAMES,
    Chunk,
    aggregate_features,
    chunk_track,
    gpx_to_features,
    parse_gpx,
)
from model import build_dataset

GPX_DIR = Path("hiking_tracks")
RIDGE_ALPHA = 0.5
BASELINE = ("distance-600m", 13.8)  # from previous sweeps

# Indices used by the two-stage model
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
TOBLER_COL = _IDX["total_tobler_min"]
RESIDUAL_COLS = [i for i in range(len(FEATURE_NAMES)) if i != TOBLER_COL]

# Distribution features used for the fine-scale layer in multiscale
DIST_FEATURE_NAMES = [
    "p75_tobler_min", "p90_tobler_min", "grade_std_mean",
    "frac_steep", "frac_very_steep", "n_chunks",
]
DIST_IDXS = [_IDX[f] for f in DIST_FEATURE_NAMES]


# ─────────────────────────────────────────────────────────────────────────────
# Generic two-stage LOO-CV (works with any feature matrix width)
# ─────────────────────────────────────────────────────────────────────────────

def loo_cv_generic(
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    tobler_col: int = TOBLER_COL,
    ridge_alpha: float = RIDGE_ALPHA,
    min_n_residual: int = 12,
    label: str = "",
    verbose: bool = False,
) -> Tuple[float, float]:
    n_features = X.shape[1]
    residual_cols = [i for i in range(n_features) if i != tobler_col]

    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        # Stage 1 – physics calibration
        X_phys = X[:, [tobler_col]]
        phys = LinearRegression(fit_intercept=False)
        phys.fit(X_phys[train_idx], y[train_idx])
        pred = phys.predict(X_phys[test_idx])

        # Stage 2 – residual correction
        if len(train_idx) >= min_n_residual:
            residuals = y[train_idx] - phys.predict(X_phys[train_idx])
            X_res = X[:, residual_cols]
            scaler = StandardScaler()
            X_res_tr = scaler.fit_transform(X_res[train_idx])
            ridge = Ridge(alpha=ridge_alpha)
            ridge.fit(X_res_tr, residuals)
            pred += ridge.predict(scaler.transform(X_res[test_idx]))

        y_pred[test_idx] = np.maximum(pred, 0.0)

    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred) * 100

    if verbose:
        errors = y_pred - y
        print(f"\n  {'Route':42s} {'Actual':>8} {'Pred':>8} {'Error':>8}")
        print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*8}")
        for name, actual, pred, err in zip(names, y, y_pred, errors):
            flag = "  <!" if abs(err) > 30 else ""
            print(f"  {name:42s} {actual:8.1f} {pred:8.1f} {err:+8.1f}{flag}")

    return mae, mape


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders for each strategy
# ─────────────────────────────────────────────────────────────────────────────

def build_standard(gpx_files, chunk_size_m, strategy):
    """Use existing build_dataset for direction / grade_band / elevation / tobler."""
    X, y, names = build_dataset(
        gpx_files, chunk_size_m=chunk_size_m, chunk_strategy=strategy
    )
    return X, y, names


def build_fixed_count(gpx_files, n_chunks):
    """Each route is split into exactly n_chunks equal-distance segments."""
    import warnings
    X_rows, y_vals, names = [], [], []
    for path in sorted(gpx_files):
        try:
            segments, duration = parse_gpx(path)
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {exc}")
            continue
        if duration is None:
            warnings.warn(f"{path.name}: no timestamps, skipping.")
            continue

        total_dist_m = sum(
            Chunk.haversine(a, b)
            for seg in segments
            for a, b in zip(seg[:-1], seg[1:])
        )
        chunk_size = max(total_dist_m / n_chunks, 50.0)
        chunks = chunk_track(segments, chunk_size, strategy="distance")
        x_vec = aggregate_features(chunks)
        X_rows.append(x_vec)
        y_vals.append(duration)
        names.append(path.stem)

    return np.array(X_rows), np.array(y_vals), names


def build_multiscale(gpx_files, fine_m, coarse_m):
    """
    Compute features at two distance scales and concatenate:
      - All 19 coarse-scale features
      - 6 distribution-only fine-scale features (p75, p90, grade_std_mean,
        frac_steep, frac_very_steep, n_chunks)
    Result: 25-feature vector.
    total_tobler_min is still at coarse index 3 = TOBLER_COL.
    """
    import warnings
    X_rows, y_vals, names = [], [], []
    for path in sorted(gpx_files):
        try:
            x_coarse, duration = gpx_to_features(path, coarse_m, "distance")
            x_fine, _ = gpx_to_features(path, fine_m, "distance")
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {exc}")
            continue
        if duration is None:
            warnings.warn(f"{path.name}: no timestamps, skipping.")
            continue

        x = np.concatenate([x_coarse, x_fine[DIST_IDXS]])
        X_rows.append(x)
        y_vals.append(duration)
        names.append(path.stem)

    return np.array(X_rows), np.array(y_vals), names


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep(label, X, y, names, tobler_col=TOBLER_COL, verbose=False):
    mae, mape = loo_cv_generic(X, y, names, tobler_col=tobler_col,
                               label=label, verbose=verbose)
    print(f"  {label:40s}  MAE={mae:5.1f} min  MAPE={mape:4.1f}%")
    return mae, mape


def main():
    gpx_files = sorted(GPX_DIR.glob("*.gpx"))
    if not gpx_files:
        print(f"No .gpx files in {GPX_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(gpx_files)} GPX files\n")

    results = []

    # ── Baseline (for reference) ─────────────────────────────────────────────
    print("=== BASELINE ===")
    name, mae_ref = BASELINE
    print(f"  {name:40s}  MAE={mae_ref:5.1f} min  (from previous sweep)\n")

    # ── Ascent/Descent ───────────────────────────────────────────────────────
    print("=== DIRECTION (ascent/descent split) ===")
    for min_d in [50, 100, 150, 200, 300]:
        X, y, names = build_standard(gpx_files, min_d, "direction")
        label = f"direction  min={min_d}m"
        mae, mape = sweep(label, X, y, names)
        results.append((label, mae, mape))
    print()

    # ── Grade-band ───────────────────────────────────────────────────────────
    print("=== GRADE_BAND ===")
    for min_d in [50, 100, 150, 200, 300]:
        X, y, names = build_standard(gpx_files, min_d, "grade_band")
        label = f"grade_band min={min_d}m"
        mae, mape = sweep(label, X, y, names)
        results.append((label, mae, mape))
    print()

    # ── Fixed chunk count ────────────────────────────────────────────────────
    print("=== FIXED CHUNK COUNT ===")
    for n in [10, 15, 20, 25, 30, 40]:
        X, y, names = build_fixed_count(gpx_files, n)
        label = f"fixed_count N={n}"
        mae, mape = sweep(label, X, y, names)
        results.append((label, mae, mape))
    print()

    # ── Multi-scale ──────────────────────────────────────────────────────────
    print("=== MULTISCALE (coarse + fine distribution stats) ===")
    for fine, coarse in [(100, 600), (200, 600), (300, 600), (200, 800)]:
        X, y, names = build_multiscale(gpx_files, fine, coarse)
        label = f"multiscale fine={fine}m coarse={coarse}m"
        mae, mape = sweep(label, X, y, names)
        results.append((label, mae, mape))
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 65)
    print("SUMMARY  (baseline = distance-600m, MAE=13.8 min)")
    print("=" * 65)
    print(f"  {'Config':40s} {'MAE':>8} {'MAPE':>8} {'vs base':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    best = min(results, key=lambda r: r[1])
    for label, mae, mape in sorted(results, key=lambda r: r[1]):
        delta = mae - mae_ref
        marker = "  <-- BEST" if label == best[0] else ""
        print(f"  {label:40s} {mae:8.1f} {mape:8.1f} {delta:+8.1f}{marker}")

    # Show per-route breakdown for the best strategy
    best_label, best_mae, _ = best
    if best_mae < mae_ref:
        print(f"\n*** {best_label} beats baseline by {mae_ref - best_mae:.1f} min ***")
        print(f"\nPer-route detail for: {best_label}")
        # Re-run verbose
        strategy_parts = best_label.split()
        if "direction" in best_label:
            min_d = int(strategy_parts[-1].replace("m", "").split("=")[1])
            X, y, names = build_standard(gpx_files, min_d, "direction")
        elif "grade_band" in best_label:
            min_d = int(strategy_parts[-1].replace("m", "").split("=")[1])
            X, y, names = build_standard(gpx_files, min_d, "grade_band")
        elif "fixed_count" in best_label:
            n = int(strategy_parts[-1].split("=")[1])
            X, y, names = build_fixed_count(gpx_files, n)
        elif "multiscale" in best_label:
            fine = int(strategy_parts[1].replace("fine=", "").replace("m", ""))
            coarse = int(strategy_parts[2].replace("coarse=", "").replace("m", ""))
            X, y, names = build_multiscale(gpx_files, fine, coarse)
        loo_cv_generic(X, y, names, verbose=True)


if __name__ == "__main__":
    main()
