#!/usr/bin/env python3
"""
generate_demo_gpx.py
────────────────────
Generates synthetic GPX files with realistic hiking profiles for testing.
Each file has a slightly different character:
  - flat coastal walk
  - long mountain approach
  - steep technical scramble
  - rolling hills
  - high alpine plateau
  - ...
"""

import math
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


GPX_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="hike_predictor_demo">
  <trk>
    <name>{name}</name>
    <trkseg>
{points}
    </trkseg>
  </trk>
</gpx>"""

TRKPT_TEMPLATE = ('      <trkpt lat="{lat:.6f}" lon="{lon:.6f}">'
                  '<ele>{ele:.1f}</ele><time>{time}</time></trkpt>')


def tobler_speed(gradient: float) -> float:
    """km/h given rise/run gradient."""
    return max(6.0 * math.exp(-3.5 * abs(gradient + 0.05)), 0.1)


def generate_route(
    name: str,
    total_dist_km: float,
    elevation_profile: list,   # [(frac_along_route, elevation_m)]
    start_lat: float = 41.7,
    start_lon: float = 1.9,
    point_spacing_m: float = 50.0,
    noise_m: float = 3.0,
) -> tuple:
    """
    Returns (gpx_string, actual_duration_minutes).
    elevation_profile is interpolated along the route.
    """
    n_points = int(total_dist_km * 1000 / point_spacing_m) + 1
    rng = random.Random(hash(name))

    # Interpolate elevation profile
    fracs   = [p[0] for p in elevation_profile]
    eles    = [p[1] for p in elevation_profile]

    def interp_ele(frac):
        for i in range(len(fracs) - 1):
            if fracs[i] <= frac <= fracs[i + 1]:
                t = (frac - fracs[i]) / (fracs[i + 1] - fracs[i] + 1e-9)
                return eles[i] + t * (eles[i + 1] - eles[i])
        return eles[-1]

    # Build trackpoints
    start_time = datetime(2024, 6, 15, 7, 0, tzinfo=timezone.utc)
    current_time = start_time
    lat, lon = start_lat, start_lon
    # Walk roughly north-east
    dlat_per_m = 1 / 111_000
    dlon_per_m = 1 / (111_000 * math.cos(math.radians(start_lat)))

    pts = []
    total_time_min = 0.0

    for i in range(n_points):
        frac = i / max(n_points - 1, 1)
        ele = interp_ele(frac) + rng.gauss(0, noise_m)

        # Gradient to next point
        if i < n_points - 1:
            next_frac = (i + 1) / max(n_points - 1, 1)
            de = interp_ele(next_frac) - interp_ele(frac)
            grad = de / point_spacing_m
            speed = tobler_speed(grad)
            dt_h  = (point_spacing_m / 1000) / speed
            dt_min = dt_h * 60
            total_time_min += dt_min
            current_time += timedelta(hours=dt_h)

        time_str = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append(TRKPT_TEMPLATE.format(
            lat=lat, lon=lon, ele=ele, time=time_str
        ))
        lat += dlat_per_m * point_spacing_m * (0.7 + rng.random() * 0.3)
        lon += dlon_per_m * point_spacing_m * (0.5 + rng.random() * 0.3)

    gpx = GPX_TEMPLATE.format(name=name, points="\n".join(pts))
    return gpx, total_time_min


DEMO_ROUTES = [
    # (name, dist_km, elevation_profile)
    ("flat_coastal_walk",       8.0,  [(0, 10), (1, 15)]),
    ("gentle_hill_loop",       12.0,  [(0, 200), (0.4, 480), (0.7, 350), (1, 200)]),
    ("montserrat_classic",     18.5,  [(0, 300), (0.3, 900), (0.6, 1200), (0.8, 950), (1, 300)]),
    ("pedraforca_southeast",   14.0,  [(0, 1000), (0.35, 1800), (0.5, 2006), (0.65, 1800), (1, 1000)]),
    ("pyrenees_long_trail",    35.0,  [(0, 1200), (0.2, 1600), (0.45, 2300), (0.6, 1900), (0.8, 2100), (1, 1400)]),
    ("steep_scramble",          6.0,  [(0, 800), (0.5, 1700), (1, 800)]),
    ("alpine_plateau",         22.0,  [(0, 2000), (0.1, 2400), (0.5, 2500), (0.9, 2380), (1, 2000)]),
    ("valley_walk_easy",       10.0,  [(0, 500), (0.3, 600), (0.7, 550), (1, 500)]),
    ("technical_ridge",         9.0,  [(0, 1400), (0.2, 1900), (0.4, 1750), (0.6, 2050), (0.8, 1900), (1, 1400)]),
    ("forest_circuit",         16.0,  [(0, 400), (0.25, 750), (0.5, 600), (0.75, 820), (1, 400)]),
    ("high_col_crossing",      20.0,  [(0, 1600), (0.4, 2700), (0.6, 2680), (1, 1500)]),
    ("coastal_cliffs",         13.0,  [(0, 50), (0.2, 280), (0.4, 60), (0.6, 310), (0.8, 120), (1, 50)]),
    ("summit_pyramid",          7.5,  [(0, 1200), (0.6, 2800), (1, 1200)]),
    ("river_valley_traverse",  28.0,  [(0, 700), (0.15, 900), (0.5, 1300), (0.85, 950), (1, 700)]),
    ("col_and_back",           11.0,  [(0, 1800), (0.5, 2500), (1, 1800)]),
    ("long_ridge_walk",        25.0,  [(0, 1500), (0.1, 1900), (0.3, 2100), (0.5, 2200), (0.7, 2050), (0.9, 1800), (1, 1600)]),
    ("exposed_scramble_hard",   5.0,  [(0, 1600), (0.4, 2300), (0.6, 2200), (1, 1600)]),
    ("glacier_approach",       17.0,  [(0, 1400), (0.3, 2200), (0.7, 2900), (1, 2200)]),
    ("easy_family_walk",        6.0,  [(0, 250), (0.5, 380), (1, 250)]),
    ("mixed_terrain_epic",     40.0,  [(0, 900), (0.1, 1400), (0.3, 2100), (0.5, 1600), (0.7, 2300), (0.9, 1700), (1, 900)]),
]


def main():
    out_dir = Path("demo_gpx")
    out_dir.mkdir(exist_ok=True)

    print(f"Generating {len(DEMO_ROUTES)} demo GPX files in ./{out_dir}/\n")
    print(f"{'Name':40s} {'Dist':>8} {'Duration':>12}")
    print(f"{'─'*40} {'─'*8} {'─'*12}")

    for name, dist, profile in DEMO_ROUTES:
        gpx_str, dur = generate_route(name, dist, profile)
        fpath = out_dir / f"{name}.gpx"
        fpath.write_text(gpx_str)
        h, m = divmod(int(round(dur)), 60)
        print(f"  {name:38s} {dist:6.1f} km   {h}h {m:02d}m  ({dur:.0f} min)")

    print(f"\n✓ Files written to ./{out_dir}/")
    print(f"\nNext steps:")
    print(f"  python train.py --gpx-dir {out_dir} --output model.pkl")
    print(f"  python predict.py model.pkl {out_dir}/montserrat_classic.gpx")


if __name__ == "__main__":
    main()
