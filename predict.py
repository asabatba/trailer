#!/usr/bin/env python3
"""
predict.py — Predict hiking time for one or more GPX files.

Usage
─────
  python predict.py model.pkl hike.gpx
  python predict.py model.pkl hike1.gpx hike2.gpx
  python predict.py model.pkl --dir ./routes/
"""

import argparse
import sys
from pathlib import Path

from gpx_features import FEATURE_NAMES, describe_gpx, gpx_to_features
from model import HikingTimeModel

_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def format_duration(minutes: float) -> str:
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def main():
    ap = argparse.ArgumentParser(description="Predict hiking time from GPX")
    ap.add_argument("model", help="Trained model .pkl file")
    ap.add_argument("gpx_files", nargs="*", help="GPX file(s) to predict")
    ap.add_argument("--dir", help="Predict all .gpx in a directory")
    ap.add_argument("--verbose", action="store_true", help="Show feature breakdown")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    model = HikingTimeModel.load(model_path)
    print(f"Loaded model from {model_path}\n")

    # Collect GPX files
    paths = [Path(p) for p in args.gpx_files]
    if args.dir:
        paths += sorted(Path(args.dir).glob("*.gpx"))

    if not paths:
        print("No GPX files specified.", file=sys.stderr)
        ap.print_help()
        sys.exit(1)

    print(
        f"{'Route':40s} {'Distance':>10} {'Gain':>7} {'Predicted':>12} {'Actual':>10}"
    )
    print(f"{'─' * 40} {'─' * 10} {'─' * 7} {'─' * 12} {'─' * 10}")

    for path in paths:
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue
        try:
            chunk_strategy = getattr(model, "chunk_strategy", "distance")
            ele_smooth = getattr(model, "ele_smooth_window", 1)
            x_vec, actual = gpx_to_features(
                path, model.chunk_size_m, chunk_strategy, ele_smooth
            )
            pred_min = model.predict_one(x_vec)

            dist_km = x_vec[_IDX["total_dist_km"]]
            gain_m = x_vec[_IDX["total_gain_m"]]
            actual_str = format_duration(actual) if actual else "–"

            print(
                f"  {path.stem:38s} {dist_km:8.1f} km "
                f"{gain_m:+6.0f}m  "
                f"{format_duration(pred_min):>10s}  "
                f"{actual_str:>10s}"
            )

            if args.verbose:
                info = describe_gpx(
                    path, model.chunk_size_m, chunk_strategy, ele_smooth
                )
                print(f"    tobler_baseline : {info['total_tobler_min']:.1f} min")
                print(
                    f"    max_grade       : {info['max_grade']:.2f}  "
                    f"({info['max_grade'] * 100:.0f}%)"
                )
                print(f"    frac_steep      : {info['frac_steep'] * 100:.0f}%")
                print(f"    terrain_roughness: {info['grade_std_mean']:.4f}")
                print()

        except Exception as exc:
            print(f"  ERROR processing {path.name}: {exc}")


if __name__ == "__main__":
    main()
