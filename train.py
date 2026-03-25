#!/usr/bin/env python3
"""
train.py — Train the hiking time predictor on a folder of GPX files.

Usage
─────
  # GPX files have timestamps (most GPS recordings):
  python train.py --gpx-dir ./my_hikes --output model.pkl

  # Provide manual labels (CSV: filename_stem,minutes):
  python train.py --gpx-dir ./my_hikes --labels labels.csv --output model.pkl

  # Tune chunk size and Ridge alpha:
  python train.py --gpx-dir ./my_hikes --chunk-size 150 --alpha 5.0

Labels CSV format
─────────────────
  filename_stem,minutes
  hike_montserrat,187
  hike_pedraforca,310
  ...
  (no header required; or include "filename_stem,minutes" header)
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from gpx_features import describe_gpx
from model import HikingTimeModel, build_dataset, loo_cv


def load_labels(csv_path: Path) -> dict:
    labels = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if (
                not row
                or len(row) < 2
                or row[0].strip().lower() in ("filename_stem", "filename")
            ):
                continue
            stem = row[0].strip()
            minutes = float(row[1].strip())
            labels[stem] = minutes
    return labels


def main():
    ap = argparse.ArgumentParser(description="Train hiking time predictor")
    ap.add_argument("--gpx-dir", required=True, help="Directory of .gpx files")
    ap.add_argument("--output", default="model.pkl", help="Output model path")
    ap.add_argument("--labels", default=None, help="CSV of manual labels")
    ap.add_argument(
        "--chunk-size",
        type=float,
        default=200.0,
        help="Chunk size in metres (default 200)",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Ridge regularisation alpha (default 10.0)",
    )
    ap.add_argument("--no-cv", action="store_true", help="Skip LOO cross-validation")
    ap.add_argument(
        "--describe",
        action="store_true",
        help="Print feature summary for each GPX and exit",
    )
    args = ap.parse_args()

    gpx_dir = Path(args.gpx_dir)
    gpx_files = sorted(gpx_dir.glob("*.gpx"))

    if not gpx_files:
        print(f"No .gpx files found in {gpx_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(gpx_files)} GPX file(s) in {gpx_dir}")

    # ── Describe mode ────────────────────────────────────────────────────────
    if args.describe:
        for p in gpx_files:
            info = describe_gpx(p, args.chunk_size)
            print(f"\n{'─' * 60}")
            print(f"  {p.name}")
            for k, v in info.items():
                if isinstance(v, float):
                    print(f"    {k:30s}: {v:.3f}")
                else:
                    print(f"    {k:30s}: {v}")
        return

    # ── Load optional labels ─────────────────────────────────────────────────
    labels = None
    if args.labels:
        labels = load_labels(Path(args.labels))
        print(f"Loaded {len(labels)} manual labels from {args.labels}")

    # ── Build dataset ────────────────────────────────────────────────────────
    print(f"\nExtracting features (chunk_size={args.chunk_size} m)…")
    X, y, names = build_dataset(gpx_files, labels=labels, chunk_size_m=args.chunk_size)

    if len(y) < 3:
        print(
            f"Need at least 3 labelled samples to train (got {len(y)}).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDataset: {len(y)} samples")
    print(
        f"  Duration range : {y.min():.0f} – {y.max():.0f} min  "
        f"(mean {y.mean():.0f} min)"
    )

    # ── Cross-validation ─────────────────────────────────────────────────────
    if not args.no_cv and len(y) >= 3:
        print(f"\nRunning LOO-CV…")
        cv_results = loo_cv(
            X, y, names, ridge_alpha=args.alpha, chunk_size_m=args.chunk_size
        )
    else:
        cv_results = None

    # ── Fit final model on all data ──────────────────────────────────────────
    print(f"\nFitting final model on all {len(y)} samples…")
    model = HikingTimeModel(
        ridge_alpha=args.alpha,
        chunk_size_m=args.chunk_size,
    )
    model.fit(X, y)

    # ── Feature importance ───────────────────────────────────────────────────
    print("\nFeature importance (normalised):")
    for feat, score in list(model.feature_importance().items())[:8]:
        bar = "█" * int(score * 30)
        print(f"  {feat:30s} {bar:<30s} {score:.3f}")

    # ── Physics stage diagnostics ────────────────────────────────────────────
    pm = model._physics_model
    if pm is not None:
        coefs = dict(zip(["tobler_min", "total_gain_m", "total_loss_m"], pm.coef_))
        print(f"\nPhysics calibration (Tobler stage):")
        print(f"  α·tobler_min  : {coefs['tobler_min']:.4f}  (1.0 = perfect Tobler)")
        print(f"  β·gain_m      : {coefs['total_gain_m']:.4f} min/m")
        print(f"  γ·loss_m      : {coefs['total_loss_m']:.4f} min/m")
        print(f"  intercept     : {pm.intercept_:.2f} min")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = Path(args.output)
    model.save(output)

    if cv_results:
        print(
            f"\n✓ Final model saved.  LOO-CV MAE: {cv_results['mae_min']:.1f} min "
            f"({cv_results['mape_pct']:.1f}%)"
        )


if __name__ == "__main__":
    main()
