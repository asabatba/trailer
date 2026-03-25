#!/usr/bin/env python3
"""
sweep_advanced.py — Four advanced feature-engineering approaches.

A. Overlapping windows
   600 m window, stride 300/200/150 m.
   Replace p75/p90/grade_std distribution features with smoother sliding-
   window estimates.  Same 19-feature vector — drop-in for the model.

B. Coarse dual-scale (600+1200, 500+1000)
   All 19 features from the fine scale, plus 6 distribution stats from the
   coarse scale (p75, p90, grade_std_mean, frac_steep, frac_very_steep,
   n_chunks) → 25 features.

C. Position-aware features  (+6 appended to 600 m base)
   first/second-half Tobler, gain and loss fractions;
   hardest-25%-chunk gain/loss fraction and their mean position.

D. Event features  (+4 appended to 600 m base)
   reversal count; longest steep-descent / steep-ascent run (km);
   number of steep-descent runs > 500 m.

All use two-stage LOO-CV (physics OLS → Ridge), ridge_alpha=0.5, N=20.
Baseline: distance-600m, MAE=13.8 min.
"""

import sys
import warnings
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
    TrackPoint,
    aggregate_features,
    chunk_track,
    gpx_to_features,
    parse_gpx,
)

GPX_DIR = Path("hiking_tracks")
RIDGE_ALPHA = 0.5
BASELINE_MAE = 13.8

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
TOBLER_COL = _IDX["total_tobler_min"]

# Distribution features replaced / supplemented in A and B
DIST_NAMES = [
    "p75_tobler_min",
    "p90_tobler_min",
    "grade_std_mean",
    "frac_steep",
    "frac_very_steep",
    "n_chunks",
]
DIST_IDXS = [_IDX[f] for f in DIST_NAMES]

# Steep threshold used in D
STEEP_THRESH = 0.15  # |mean_grade| > 15 %


# ─────────────────────────────────────────────────────────────────────────────
# Generic two-stage LOO-CV
# ─────────────────────────────────────────────────────────────────────────────


def loo_cv(
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    tobler_col: int = TOBLER_COL,
    ridge_alpha: float = RIDGE_ALPHA,
    min_n_residual: int = 12,
    verbose: bool = False,
) -> Tuple[float, float]:
    n_feat = X.shape[1]
    res_cols = [i for i in range(n_feat) if i != tobler_col]

    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        Xp_tr = X[train_idx][:, [tobler_col]]
        Xp_te = X[test_idx][:, [tobler_col]]
        phys = LinearRegression(fit_intercept=False)
        phys.fit(Xp_tr, y[train_idx])
        pred = phys.predict(Xp_te)

        if len(train_idx) >= min_n_residual:
            resid = y[train_idx] - phys.predict(Xp_tr)
            scaler = StandardScaler()
            Xr_tr = scaler.fit_transform(X[train_idx][:, res_cols])
            ridge = Ridge(alpha=ridge_alpha)
            ridge.fit(Xr_tr, resid)
            pred += ridge.predict(scaler.transform(X[test_idx][:, res_cols]))

        y_pred[test_idx] = np.maximum(pred, 0.0)

    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred) * 100

    if verbose:
        errs = y_pred - y
        print(f"\n  {'Route':44s} {'Act':>7} {'Pred':>7} {'Err':>7}")
        print(f"  {'-' * 44} {'-' * 7} {'-' * 7} {'-' * 7}")
        for n, a, p, e in zip(names, y, y_pred, errs):
            flag = "  <!" if abs(e) > 30 else ""
            print(f"  {n:44s} {a:7.1f} {p:7.1f} {e:+7.1f}{flag}")

    return mae, mape


def row(label, mae, mape, pad=44):
    delta = mae - BASELINE_MAE
    sign = "+" if delta >= 0 else ""
    print(f"  {label:{pad}s}  MAE={mae:5.1f}  MAPE={mape:4.1f}%  ({sign}{delta:.1f})")
    return mae, mape


# ─────────────────────────────────────────────────────────────────────────────
# A — Overlapping windows
# ─────────────────────────────────────────────────────────────────────────────


def sliding_windows(
    segments: List[List[TrackPoint]],
    window_m: float,
    stride_m: float,
) -> List[Chunk]:
    """Overlapping sliding-window chunks of width window_m."""
    all_w: List[Chunk] = []
    for seg in segments:
        n = len(seg)
        if n < 2:
            continue
        cum = [0.0]
        for a, b in zip(seg[:-1], seg[1:]):
            cum.append(cum[-1] + Chunk.haversine(a, b))
        total = cum[-1]
        if total <= window_m:
            all_w.append(Chunk(points=list(seg)))
            continue
        start = 0.0
        while start < total:
            end = start + window_m
            pts = [seg[i] for i, d in enumerate(cum) if start <= d <= end]
            if len(pts) >= 2:
                all_w.append(Chunk(points=pts))
            start += stride_m
    return all_w


def build_overlapping(gpx_files, stride_m: float, window_m: float = 600.0):
    """
    19-feature vector: global totals from non-overlapping 600m chunks,
    distribution stats (p75, p90, grade_std_mean, frac_steep, frac_very_steep,
    n_chunks) replaced by overlapping-window versions.
    """
    X_rows, y_vals, names = [], [], []
    for path in sorted(gpx_files):
        try:
            segments, duration = parse_gpx(path)
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {exc}")
            continue
        if duration is None:
            continue

        # Base vector from non-overlapping chunks (global totals are correct)
        x_base = aggregate_features(chunk_track(segments, window_m, "distance"))

        # Distribution stats from overlapping windows
        wins = sliding_windows(segments, window_m, stride_m)
        if wins:
            wf = [w.compute() for w in wins]
            toblers = np.array([f["tobler_min"] for f in wf])
            grades = np.array([f["mean_grade"] for f in wf])
            stds = np.array([f["grade_std"] for f in wf])
            x_base[_IDX["p75_tobler_min"]] = float(np.percentile(toblers, 75))
            x_base[_IDX["p90_tobler_min"]] = float(np.percentile(toblers, 90))
            x_base[_IDX["grade_std_mean"]] = float(np.mean(stds))
            x_base[_IDX["frac_steep"]] = float(np.mean(np.abs(grades) > 0.25))
            x_base[_IDX["frac_very_steep"]] = float(np.mean(np.abs(grades) > 0.40))
            x_base[_IDX["n_chunks"]] = float(len(wins))

        X_rows.append(x_base)
        y_vals.append(duration)
        names.append(path.stem)

    return np.array(X_rows), np.array(y_vals), names


# ─────────────────────────────────────────────────────────────────────────────
# B — Coarse dual-scale
# ─────────────────────────────────────────────────────────────────────────────


def build_dualscale(gpx_files, fine_m: float, coarse_m: float):
    """19 fine-scale features + 6 coarse-scale distribution stats = 25 features."""
    X_rows, y_vals, names = [], [], []
    for path in sorted(gpx_files):
        try:
            x_fine, duration = gpx_to_features(path, fine_m, "distance")
            x_coarse, _ = gpx_to_features(path, coarse_m, "distance")
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {exc}")
            continue
        if duration is None:
            continue
        x = np.concatenate([x_fine, x_coarse[DIST_IDXS]])
        X_rows.append(x)
        y_vals.append(duration)
        names.append(path.stem)
    return np.array(X_rows), np.array(y_vals), names


# ─────────────────────────────────────────────────────────────────────────────
# C — Position-aware features
# ─────────────────────────────────────────────────────────────────────────────


def position_features(feats: List[dict]) -> np.ndarray:
    """
    6 position-aware features derived from the ordered chunk sequence.
      0  second_half_tobler_frac  — fraction of Tobler time in 2nd half
      1  second_half_gain_frac    — fraction of gain in 2nd half
      2  second_half_loss_frac    — fraction of loss in 2nd half
      3  hard25_gain_frac         — gain fraction in hardest 25% chunks
      4  hard25_loss_frac         — loss fraction in hardest 25% chunks
      5  hard25_center_pos        — mean positional index (0=start, 1=end)
    """
    n = len(feats)
    eps = 1e-6
    half = max(n // 2, 1)

    total_tobler = sum(f["tobler_min"] for f in feats) + eps
    total_gain = sum(f["gain_m"] for f in feats) + eps
    total_loss = sum(f["loss_m"] for f in feats) + eps

    sh_tobler = sum(f["tobler_min"] for f in feats[half:])
    sh_gain = sum(f["gain_m"] for f in feats[half:])
    sh_loss = sum(f["loss_m"] for f in feats[half:])

    # Hardest 25% chunks by Tobler time per km
    tk = [f["tobler_min"] / (f["dist_m"] / 1000.0 + eps) for f in feats]
    threshold = np.percentile(tk, 75) if n >= 4 else max(tk)
    hard_idx = [i for i, t in enumerate(tk) if t >= threshold]

    hard_gain = sum(feats[i]["gain_m"] for i in hard_idx)
    hard_loss = sum(feats[i]["loss_m"] for i in hard_idx)
    hard_center = np.mean([i / (n - 1 + eps) for i in hard_idx]) if hard_idx else 0.5

    return np.array(
        [
            sh_tobler / total_tobler,
            sh_gain / total_gain,
            sh_loss / total_loss,
            hard_gain / total_gain,
            hard_loss / total_loss,
            hard_center,
        ],
        dtype=np.float64,
    )


# ─────────────────────────────────────────────────────────────────────────────
# D — Event features
# ─────────────────────────────────────────────────────────────────────────────


def event_features(feats: List[dict], min_run_m: float = 500.0) -> np.ndarray:
    """
    4 event features derived from the ordered chunk sequence.
      0  n_reversals              — chunk-direction flips (up→down or down→up)
      1  longest_steep_descent_km — longest contiguous steep-descent run
      2  longest_steep_ascent_km  — longest contiguous steep-ascent run
      3  n_long_steep_descents    — steep-descent runs > min_run_m
    """
    # Direction of each chunk
    dirs = [1 if f["gain_m"] >= f["loss_m"] else -1 for f in feats]
    reversals = sum(1 for a, b in zip(dirs[:-1], dirs[1:]) if a != b)

    def _longest_run(condition_fn):
        best = 0.0
        cur = 0.0
        for f in feats:
            if condition_fn(f):
                cur += f["dist_m"]
                best = max(best, cur)
            else:
                cur = 0.0
        return best / 1000.0  # → km

    def _count_runs(condition_fn, min_m):
        count = 0
        cur = 0.0
        for f in feats:
            if condition_fn(f):
                cur += f["dist_m"]
            else:
                if cur >= min_m:
                    count += 1
                cur = 0.0
        if cur >= min_m:
            count += 1
        return count

    steep_desc = lambda f: f["mean_grade"] < -STEEP_THRESH
    steep_asc = lambda f: f["mean_grade"] > STEEP_THRESH

    return np.array(
        [
            float(reversals),
            _longest_run(steep_desc),
            _longest_run(steep_asc),
            float(_count_runs(steep_desc, min_run_m)),
        ],
        dtype=np.float64,
    )


def build_with_extra(gpx_files, extra_fn, chunk_size_m=600.0):
    """
    19-feature base vector (600m distance) plus extra features from extra_fn.
    extra_fn(feats: List[dict]) -> np.ndarray
    """
    X_rows, y_vals, names = [], [], []
    for path in sorted(gpx_files):
        try:
            segments, duration = parse_gpx(path)
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {exc}")
            continue
        if duration is None:
            continue

        chunks = chunk_track(segments, chunk_size_m, "distance")
        feats = [c.compute() for c in chunks]
        x_base = aggregate_features(chunks)
        x_extra = extra_fn(feats)
        X_rows.append(np.concatenate([x_base, x_extra]))
        y_vals.append(duration)
        names.append(path.stem)

    return np.array(X_rows), np.array(y_vals), names


# ─────────────────────────────────────────────────────────────────────────────
# Standard 600 m baseline builder (for consistent comparison)
# ─────────────────────────────────────────────────────────────────────────────


def build_base(gpx_files, chunk_size_m=600.0):
    X_rows, y_vals, names = [], [], []
    for path in sorted(gpx_files):
        try:
            x, duration = gpx_to_features(path, chunk_size_m, "distance")
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {exc}")
            continue
        if duration is None:
            continue
        X_rows.append(x)
        y_vals.append(duration)
        names.append(path.stem)
    return np.array(X_rows), np.array(y_vals), names


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    gpx_files = sorted(GPX_DIR.glob("*.gpx"))
    if not gpx_files:
        print(f"No .gpx files in {GPX_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(gpx_files)} GPX files")
    print(f"Baseline (distance-600m): MAE={BASELINE_MAE} min\n")

    results = {}

    # ── A: Overlapping windows ───────────────────────────────────────────────
    print("=== A: Overlapping windows (600 m window) ===")
    for stride in [150, 200, 300]:
        X, y, names = build_overlapping(gpx_files, stride_m=stride)
        mae, mape = loo_cv(X, y, names)
        label = f"A  stride={stride}m"
        results[label] = row(label, mae, mape)
    print()

    # ── B: Coarse dual-scale ─────────────────────────────────────────────────
    print("=== B: Coarse dual-scale ===")
    for fine, coarse in [(500, 1000), (600, 1200), (600, 900)]:
        X, y, names = build_dualscale(gpx_files, fine, coarse)
        mae, mape = loo_cv(X, y, names)
        label = f"B  {fine}+{coarse}m"
        results[label] = row(label, mae, mape)
    print()

    # ── C: Position features ─────────────────────────────────────────────────
    print("=== C: Position-aware features (+6) ===")
    X, y, names = build_with_extra(gpx_files, position_features)
    mae, mape = loo_cv(X, y, names)
    label = "C  position"
    results[label] = row(label, mae, mape)
    print()

    # ── D: Event features ────────────────────────────────────────────────────
    print("=== D: Event features (+4) ===")
    X, y, names = build_with_extra(gpx_files, event_features)
    mae, mape = loo_cv(X, y, names)
    label = "D  events"
    results[label] = row(label, mae, mape)
    print()

    # ── C+D combined ─────────────────────────────────────────────────────────
    print("=== C+D: Position + Event features (+10) ===")

    def cd_features(feats):
        return np.concatenate([position_features(feats), event_features(feats)])

    X, y, names = build_with_extra(gpx_files, cd_features)
    mae, mape = loo_cv(X, y, names)
    label = "C+D  pos+events"
    results[label] = row(label, mae, mape)
    print()

    # ── Best of A/B combined with C+D ────────────────────────────────────────
    print("=== Combinations with C+D ===")

    # A(best stride) + C+D
    best_a_stride = min(
        [(s, results.get(f"A  stride={s}m", (99, 0))[0]) for s in [150, 200, 300]],
        key=lambda x: x[1],
    )[0]
    X_a, y, names = build_overlapping(gpx_files, stride_m=best_a_stride)
    X_cd, _, _ = build_with_extra(gpx_files, cd_features)
    X_acd = np.concatenate([X_a, X_cd[:, 19:]], axis=1)  # keep A base, add C+D extras
    mae, mape = loo_cv(X_acd, y, names)
    label = f"A(stride={best_a_stride})+C+D"
    results[label] = row(label, mae, mape)

    # B(best) + C+D
    best_b = min(
        [(k, v[0]) for k, v in results.items() if k.startswith("B ")],
        key=lambda x: x[1],
    )
    best_b_params = best_b[0].replace("B  ", "").replace("m", "").split("+")
    fine_b, coarse_b = int(best_b_params[0]), int(best_b_params[1])
    X_b, y, names = build_dualscale(gpx_files, fine_b, coarse_b)
    X_cd_extra = X_cd[:, 19:]
    X_bcd = np.concatenate([X_b, X_cd_extra], axis=1)
    mae, mape = loo_cv(X_bcd, y, names)
    label = f"B({fine_b}+{coarse_b})+C+D"
    results[label] = row(label, mae, mape)
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"SUMMARY  (baseline = distance-600m, MAE={BASELINE_MAE} min)")
    print("=" * 60)
    print(f"  {'Config':44s} {'MAE':>6} {'MAPE':>6} {'vs base':>8}")
    print(f"  {'-' * 44} {'-' * 6} {'-' * 6} {'-' * 8}")
    for label, (mae, mape) in sorted(results.items(), key=lambda x: x[1][0]):
        delta = mae - BASELINE_MAE
        sign = "+" if delta >= 0 else ""
        marker = "  <-- BEST" if mae == min(v[0] for v in results.values()) else ""
        print(f"  {label:44s} {mae:6.1f} {mape:6.1f}% {sign}{delta:+.1f}{marker}")

    # ── Per-route for best ────────────────────────────────────────────────────
    best_label = min(results, key=lambda k: results[k][0])
    if results[best_label][0] < BASELINE_MAE:
        print(
            f"\n*** {best_label} beats baseline by "
            f"{BASELINE_MAE - results[best_label][0]:.1f} min ***"
        )
        print(f"\nPer-route: {best_label}")
        # Re-run verbose for best
        if best_label.startswith("A") and "+C" not in best_label:
            s = int(best_label.split("=")[1].replace("m", "").replace(")", ""))
            X, y, names = build_overlapping(gpx_files, stride_m=s)
        elif best_label.startswith("B") and "+C" not in best_label:
            p = best_label.replace("B  ", "").replace("m", "").split("+")
            X, y, names = build_dualscale(gpx_files, int(p[0]), int(p[1]))
        elif best_label == "C  position":
            X, y, names = build_with_extra(gpx_files, position_features)
        elif best_label == "D  events":
            X, y, names = build_with_extra(gpx_files, event_features)
        elif best_label == "C+D  pos+events":
            X, y, names = build_with_extra(gpx_files, cd_features)
        else:
            X, y, names = None, None, None
        if X is not None:
            loo_cv(X, y, names, verbose=True)


if __name__ == "__main__":
    main()
