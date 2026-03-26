"""
gpx_features.py
───────────────
GPX → chunk → features pipeline.

Chunks the track into fixed-distance segments (default 200 m), computes
per-chunk features (distance, elevation, slope, Tobler time, roughness),
then aggregates them into a flat feature vector suitable for training.

Tobler's hiking function
  speed (km/h) = 6 · exp(−3.5 · |tan(θ) + 0.05|)
where θ is the slope angle.  The +0.05 offset biases the optimum to a
gentle downhill (~2.86°), matching empirical data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import fastgpx
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TrackPoint:
    lat: float
    lon: float
    ele: float  # metres (0.0 if missing)
    time_s: float  # Unix timestamp (0.0 if missing)


@dataclass
class Chunk:
    """One ~CHUNK_SIZE_M segment of the track."""

    points: List[TrackPoint] = field(default_factory=list)

    # Computed lazily
    _features: Optional[dict] = field(default=None, repr=False)

    # ── geometry helpers ─────────────────────────────────────────────────────

    @staticmethod
    def haversine(p1: TrackPoint, p2: TrackPoint) -> float:
        """Horizontal distance in metres (WGS-84 sphere approx)."""
        R = 6_371_000.0
        φ1, φ2 = math.radians(p1.lat), math.radians(p2.lat)
        dφ = math.radians(p2.lat - p1.lat)
        dλ = math.radians(p2.lon - p1.lon)
        a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    def compute(self) -> dict:
        if self._features is not None:
            return self._features

        pts = self.points
        if len(pts) < 2:
            self._features = _empty_chunk_features()
            return self._features

        horiz_distances: List[float] = []
        gradients: List[float] = []
        ele_deltas: List[float] = []

        for a, b in zip(pts[:-1], pts[1:]):
            d_h = self.haversine(a, b)
            d_e = b.ele - a.ele
            horiz_distances.append(d_h)
            ele_deltas.append(d_e)
            # Gradient (rise / run); guard against zero distance
            grad = d_e / d_h if d_h > 0.1 else 0.0
            gradients.append(grad)

        total_dist_m = sum(horiz_distances)
        gain_m = sum(d for d in ele_deltas if d > 0)
        loss_m = abs(sum(d for d in ele_deltas if d < 0))

        grads = np.array(gradients)
        mean_grade = float(np.mean(grads))
        max_grade = float(np.max(np.abs(grads)))
        grade_std = float(np.std(grads))

        # ── Tobler time (minutes) per segment ───────────────────────────────
        # Track ascent/descent Tobler totals separately for diagnostics and
        # experiments.  Tobler significantly overestimates speed on steep
        # descent (>25% grade), so the split also helps define descent
        # penalties such as total_steep_loss_m.
        tobler_min = 0.0
        tobler_ascent_min = 0.0
        tobler_descent_min = 0.0
        for d_h, grad in zip(horiz_distances, gradients):
            speed_kmh = 6.0 * math.exp(-3.5 * abs(grad + 0.05))
            speed_kmh = max(speed_kmh, 0.1)  # floor: 100 m/h
            seg_time = (d_h / 1000.0) / speed_kmh * 60.0
            tobler_min += seg_time
            if grad >= 0:
                tobler_ascent_min += seg_time
            else:
                tobler_descent_min += seg_time

        self._features = {
            "dist_m": total_dist_m,
            "gain_m": gain_m,
            "loss_m": loss_m,
            "mean_grade": mean_grade,
            "max_grade": max_grade,
            "grade_std": grade_std,
            "tobler_min": tobler_min,
            "tobler_ascent_min": tobler_ascent_min,
            "tobler_descent_min": tobler_descent_min,
            # Weighted difficulty: uphill costs more than downhill
            "difficulty": gain_m * 1.0 + loss_m * 0.5 + total_dist_m * 0.001,
        }
        return self._features


def _empty_chunk_features() -> dict:
    return dict(
        dist_m=0,
        gain_m=0,
        loss_m=0,
        mean_grade=0,
        max_grade=0,
        grade_std=0,
        tobler_min=0,
        tobler_ascent_min=0,
        tobler_descent_min=0,
        difficulty=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GPX → TrackPoints
# ─────────────────────────────────────────────────────────────────────────────


def parse_gpx(path: str | Path) -> Tuple[List[List[TrackPoint]], Optional[float]]:
    """
    Returns (segments, actual_duration_minutes).
    segments is a list of TrackPoint lists, one per GPX segment — kept
    separate so that chunk_track does not connect non-adjacent waypoints
    across segment boundaries (e.g. device restarts mid-hike).
    actual_duration_minutes is None when the GPX has no timestamps.
    """
    return parse_gpx_xml(fastgpx.load(Path(path)))


def parse_gpx_xml(
    gpx: str | fastgpx.Gpx,
) -> Tuple[List[List[TrackPoint]], Optional[float]]:
    """Parse GPX XML content or a preloaded GPX object."""
    parsed = fastgpx.parse(gpx) if isinstance(gpx, str) else gpx

    segments: List[List[TrackPoint]] = []
    for track in parsed.tracks:
        for seg in track.segments:
            seg_pts: List[TrackPoint] = []
            for pt in seg.points:
                seg_pts.append(
                    TrackPoint(
                        lat=pt.latitude,
                        lon=pt.longitude,
                        ele=pt.elevation or 0.0,
                        time_s=pt.time.timestamp() if pt.time else 0.0,
                    )
                )
            if seg_pts:
                segments.append(seg_pts)

    if not segments:
        wpt_pts = [
            TrackPoint(w.latitude, w.longitude, w.elevation or 0.0, 0.0)
            for w in getattr(parsed, "waypoints", ())
        ]
        if wpt_pts:
            segments = [wpt_pts]

    if not segments:
        return [], None

    return segments, _moving_duration(segments)


# Implied speed below this threshold is treated as a pause (rest, lunch, photos).
# 0.5 km/h ≈ 8 m/min; even on very steep terrain hikers move faster than this.
_MIN_MOVING_SPEED_KMH: float = 0.7


def _moving_duration(segments: List[List[TrackPoint]]) -> Optional[float]:
    """
    Sum the time on GPS steps where implied speed >= _MIN_MOVING_SPEED_KMH.

    Using moving time as the training label removes the effect of breaks,
    making labels consistent across routes with different rest habits and
    avoiding the inflation from long summit stops or lunch pauses.
    Returns None when the GPX has no timestamps.
    """
    moving_s = 0.0
    timed_steps = 0

    for seg_pts in segments:
        for a, b in zip(seg_pts[:-1], seg_pts[1:]):
            if a.time_s == 0 or b.time_s == 0:
                continue
            dt = b.time_s - a.time_s
            if dt <= 0:
                continue
            timed_steps += 1
            dist_km = Chunk.haversine(a, b) / 1000.0
            speed_kmh = dist_km / (dt / 3600.0)
            if speed_kmh >= _MIN_MOVING_SPEED_KMH:
                moving_s += dt

    if timed_steps == 0:
        return None
    return moving_s / 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Elevation smoothing
# ─────────────────────────────────────────────────────────────────────────────


def _smooth_elevations(points: List[TrackPoint], window: int) -> List[TrackPoint]:
    """
    Return a new list of TrackPoints with elevations replaced by a rolling
    median over `window` consecutive points.  lat/lon/time are unchanged.

    Rolling median is robust to outlier GPS elevation spikes (bad fixes).
    Edge points use whatever samples are available (no padding artefacts).
    window=1 is a no-op.
    """
    if window <= 1 or len(points) < 2:
        return points

    eles = np.array([p.ele for p in points])
    half = window // 2
    smoothed = np.empty_like(eles)
    for i in range(len(eles)):
        lo = max(0, i - half)
        hi = min(len(eles), i + half + 1)
        smoothed[i] = np.median(eles[lo:hi])

    return [
        TrackPoint(p.lat, p.lon, float(smoothed[i]), p.time_s)
        for i, p in enumerate(points)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────


def _chunk_segment(points: List[TrackPoint], chunk_size_m: float) -> List[Chunk]:
    """Split one contiguous segment of trackpoints into ~chunk_size_m chunks."""
    if len(points) < 2:
        return [Chunk(points=points)] if points else []

    chunks: List[Chunk] = []
    current_pts: List[TrackPoint] = [points[0]]
    accumulated_m = 0.0

    for prev, curr in zip(points[:-1], points[1:]):
        d = Chunk.haversine(prev, curr)
        accumulated_m += d
        current_pts.append(curr)

        if accumulated_m >= chunk_size_m:
            chunks.append(Chunk(points=current_pts.copy()))
            # Carry the last point as start of next chunk
            current_pts = [curr]
            accumulated_m = 0.0

    if len(current_pts) >= 2:
        chunks.append(Chunk(points=current_pts))

    return chunks


def _chunk_segment_by_elevation(
    points: List[TrackPoint], chunk_ele_m: float
) -> List[Chunk]:
    """
    Split one contiguous segment into chunks of ~chunk_ele_m cumulative
    absolute elevation change (|gain| + |loss|).  On flat terrain each
    chunk will be long; on steep terrain chunks will be short, giving
    finer resolution where the route is hardest.
    """
    if len(points) < 2:
        return [Chunk(points=points)] if points else []

    chunks: List[Chunk] = []
    current_pts: List[TrackPoint] = [points[0]]
    accumulated_ele = 0.0

    for prev, curr in zip(points[:-1], points[1:]):
        accumulated_ele += abs(curr.ele - prev.ele)
        current_pts.append(curr)

        if accumulated_ele >= chunk_ele_m:
            chunks.append(Chunk(points=current_pts.copy()))
            current_pts = [curr]
            accumulated_ele = 0.0

    if len(current_pts) >= 2:
        chunks.append(Chunk(points=current_pts))

    return chunks


def _chunk_segment_by_tobler(
    points: List[TrackPoint], chunk_tobler_min: float
) -> List[Chunk]:
    """
    Split one contiguous segment into chunks of ~chunk_tobler_min accumulated
    theoretical Tobler time.

    Steep sections (slow Tobler speed) produce short, fine-grained chunks;
    flat sections produce long coarse chunks.  This gives the most resolution
    exactly where difficulty varies — the natural unit for time prediction.
    """
    if len(points) < 2:
        return [Chunk(points=points)] if points else []

    chunks: List[Chunk] = []
    current_pts: List[TrackPoint] = [points[0]]
    accumulated_min = 0.0

    for prev, curr in zip(points[:-1], points[1:]):
        d_h = Chunk.haversine(prev, curr)
        d_e = curr.ele - prev.ele
        grad = d_e / d_h if d_h > 0.1 else 0.0
        speed_kmh = max(6.0 * math.exp(-3.5 * abs(grad + 0.05)), 0.1)
        accumulated_min += (d_h / 1000.0) / speed_kmh * 60.0
        current_pts.append(curr)

        if accumulated_min >= chunk_tobler_min:
            chunks.append(Chunk(points=current_pts.copy()))
            current_pts = [curr]
            accumulated_min = 0.0

    if len(current_pts) >= 2:
        chunks.append(Chunk(points=current_pts))

    return chunks


def _chunk_segment_by_direction(
    points: List[TrackPoint], min_dist_m: float = 150.0
) -> List[Chunk]:
    """
    Split at ascent→descent and descent→ascent reversals.

    Uses smoothed elevation (window=11) to find macro direction, so GPS
    noise doesn't trigger false splits.  Only splits when at least
    min_dist_m of horizontal distance has been covered since the last split.
    Raw (unsmoothed) points are used in the returned chunks.
    """
    n = len(points)
    if n < 2:
        return [Chunk(points=points)] if points else []

    w = min(11, max(3, (n // 2) * 2 - 1))  # largest odd window ≤ 11
    smoothed = _smooth_elevations(points, window=w)

    split_starts = [0]
    dist_since_last = 0.0
    current_dir: Optional[int] = None  # +1 ascending, -1 descending

    for i in range(1, n):
        d_h = Chunk.haversine(points[i - 1], points[i])
        dist_since_last += d_h
        d_e = smoothed[i].ele - smoothed[i - 1].ele

        if abs(d_e) < 0.05:  # ignore negligible elevation change
            continue

        step_dir = 1 if d_e > 0 else -1

        if (
            current_dir is not None
            and step_dir != current_dir
            and dist_since_last >= min_dist_m
        ):
            split_starts.append(i)
            dist_since_last = 0.0

        current_dir = step_dir

    # Build chunks: points[split_starts[j] … split_starts[j+1]] (boundary shared)
    chunks = []
    for j in range(len(split_starts)):
        s = split_starts[j]
        e = split_starts[j + 1] if j + 1 < len(split_starts) else n - 1
        chunk_pts = points[s : e + 1]
        if len(chunk_pts) >= 2:
            chunks.append(Chunk(points=chunk_pts))

    return chunks


def _grade_band(grade: float) -> int:
    """
    Map a gradient to a band index (sign-aware):
      -3 very steep descent  (<-0.20)
      -2 steep descent       (-0.20 … -0.10)
      -1 gentle descent      (-0.10 … -0.03)
       0 flat                (-0.03 …  0.03)
      +1 gentle climb        ( 0.03 …  0.10)
      +2 steep climb         ( 0.10 …  0.20)
      +3 very steep climb    (>0.20)
    """
    if grade > 0.20:
        return 3
    elif grade > 0.10:
        return 2
    elif grade > 0.03:
        return 1
    elif grade >= -0.03:
        return 0
    elif grade >= -0.10:
        return -1
    elif grade >= -0.20:
        return -2
    else:
        return -3


def _chunk_segment_by_grade_band(
    points: List[TrackPoint], min_dist_m: float = 150.0
) -> List[Chunk]:
    """
    Split when the local terrain grade-band changes category.

    Grade is estimated from the smoothed elevation profile to avoid
    noise-induced band flickering.  Splits are only made after at
    least min_dist_m of horizontal travel since the previous split.
    """
    n = len(points)
    if n < 2:
        return [Chunk(points=points)] if points else []

    w = min(11, max(3, (n // 2) * 2 - 1))
    smoothed = _smooth_elevations(points, window=w)

    split_starts = [0]
    dist_since_last = 0.0
    current_band: Optional[int] = None

    for i in range(1, n):
        d_h = Chunk.haversine(points[i - 1], points[i])
        dist_since_last += d_h
        d_e = smoothed[i].ele - smoothed[i - 1].ele
        grad = d_e / d_h if d_h > 0.1 else 0.0
        band = _grade_band(grad)

        if (
            current_band is not None
            and band != current_band
            and dist_since_last >= min_dist_m
        ):
            split_starts.append(i)
            dist_since_last = 0.0

        current_band = band

    chunks = []
    for j in range(len(split_starts)):
        s = split_starts[j]
        e = split_starts[j + 1] if j + 1 < len(split_starts) else n - 1
        chunk_pts = points[s : e + 1]
        if len(chunk_pts) >= 2:
            chunks.append(Chunk(points=chunk_pts))

    return chunks


def chunk_track(
    segments: List[List[TrackPoint]],
    chunk_size_m: float = 200.0,
    strategy: str = "distance",
    ele_smooth_window: int = 1,
) -> List[Chunk]:
    """
    Split track segments into chunks.

    strategy='distance'   : chunk by horizontal distance (chunk_size_m metres)
    strategy='elevation'  : chunk by cumulative |elevation change| (chunk_size_m metres)
    strategy='tobler'     : chunk by accumulated Tobler time (chunk_size_m = minutes)
    strategy='direction'  : split at ascent/descent reversals (chunk_size_m = min dist m)
    strategy='grade_band' : split at terrain-category changes (chunk_size_m = min dist m)

    ele_smooth_window : rolling-median window applied to elevation values before
        chunking. 1 = disabled (default). Larger values remove GPS elevation noise
        but also reduce the steep-descent signal the model relies on.

    Each GPX segment is processed independently to avoid connecting
    non-adjacent waypoints across segment boundaries.
    """
    if strategy == "elevation":
        fn = _chunk_segment_by_elevation
    elif strategy == "tobler":
        fn = _chunk_segment_by_tobler
    elif strategy == "direction":
        fn = _chunk_segment_by_direction
    elif strategy == "grade_band":
        fn = _chunk_segment_by_grade_band
    else:
        fn = _chunk_segment
    chunks: List[Chunk] = []
    for seg_pts in segments:
        smoothed = _smooth_elevations(seg_pts, ele_smooth_window)
        chunks.extend(fn(smoothed, chunk_size_m))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Feature aggregation
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # ── Global totals (strong signals)
    "total_dist_km",
    "total_gain_m",
    "total_loss_m",
    "total_tobler_min",  # primary effort baseline for the constrained model
    # Best-case movement at Tobler's maximum 6 km/h (10 min/km).
    "best_case_distance_min",
    # Extra time above the best-case distance floor due to slope profile.
    "slope_penalty_min",
    # Split Tobler remains useful for diagnostics and experiments, even though
    # the default model only learns on the monotone subset below.
    "total_tobler_ascent_min",
    "total_tobler_descent_min",
    # Total vertical work, regardless of sign.
    "vertical_change_m",
    # ── Slope statistics (terrain difficulty shape)
    "mean_grade",
    "mean_abs_grade",
    "max_grade",
    "grade_std_mean",  # mean of per-chunk std (roughness)
    # ── Segment-level distribution
    "p75_tobler_min",  # upper quartile chunk time (bottleneck terrain)
    "p90_tobler_min",
    "frac_steep",  # fraction of chunks with |grade| > 0.25
    "frac_very_steep",  # fraction of chunks with |grade| > 0.40
    # Explicit steep-descent penalty: Tobler overestimates speed on grade < -0.25
    # (knees, scrambling, unstable terrain). Used directly by the constrained
    # model as a nonnegative penalty term.
    "total_steep_loss_m",
    # ── Derived
    "gain_per_km",
    "loss_per_km",
    "tobler_efficiency",  # total_dist_km / total_tobler_min (pace proxy)
    "n_chunks",  # route granularity
]


def aggregate_features(chunks: List[Chunk]) -> np.ndarray:
    """
    Compute per-chunk features, then aggregate into a fixed-size vector.
    Returns a 1-D numpy array aligned to FEATURE_NAMES.
    """
    if not chunks:
        return np.zeros(len(FEATURE_NAMES))

    feats = [c.compute() for c in chunks]

    total_dist_m = sum(f["dist_m"] for f in feats)
    total_gain_m = sum(f["gain_m"] for f in feats)
    total_loss_m = sum(f["loss_m"] for f in feats)
    total_tobler = sum(f["tobler_min"] for f in feats)
    best_case_distance_min = total_dist_m / 100.0
    slope_penalty_min = max(total_tobler - best_case_distance_min, 0.0)
    total_tobler_ascent = sum(f["tobler_ascent_min"] for f in feats)
    total_tobler_descent = sum(f["tobler_descent_min"] for f in feats)
    vertical_change_m = total_gain_m + total_loss_m
    total_dist_km = total_dist_m / 1000.0

    grades = np.array([f["mean_grade"] for f in feats])
    max_grades = np.array([f["max_grade"] for f in feats])
    stds = np.array([f["grade_std"] for f in feats])
    toblers = np.array([f["tobler_min"] for f in feats])

    steep_descent_mask = grades < -0.25
    frac_steep = float(np.mean(np.abs(grades) > 0.25))
    frac_very_steep = float(np.mean(np.abs(grades) > 0.40))
    total_steep_loss_m = sum(
        feats[i]["loss_m"] for i in range(len(feats)) if steep_descent_mask[i]
    )

    eps = 1e-6
    vec = [
        total_dist_km,
        total_gain_m,
        total_loss_m,
        total_tobler,
        best_case_distance_min,
        slope_penalty_min,
        total_tobler_ascent,
        total_tobler_descent,
        vertical_change_m,
        float(np.mean(grades)),
        float(np.mean(np.abs(grades))),
        float(np.max(max_grades)),
        float(np.mean(stds)),
        float(np.percentile(toblers, 75)),
        float(np.percentile(toblers, 90)),
        frac_steep,
        frac_very_steep,
        total_steep_loss_m,
        total_gain_m / (total_dist_km + eps),
        total_loss_m / (total_dist_km + eps),
        total_dist_km / (total_tobler + eps),
        float(len(chunks)),
    ]
    return np.array(vec, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: path → feature vector + optional label
# ─────────────────────────────────────────────────────────────────────────────


def gpx_to_features(
    path: str | Path,
    chunk_size_m: float = 200.0,
    chunk_strategy: str = "distance",
    ele_smooth_window: int = 1,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    End-to-end: GPX file → (feature_vector, actual_duration_minutes).
    actual_duration_minutes is None if the file has no timestamps.
    chunk_strategy: 'distance' or 'elevation'
    ele_smooth_window: rolling-median window for elevation denoising (1 = off)
    """
    segments, duration_min = parse_gpx(path)
    return _segments_to_features(
        segments, duration_min, chunk_size_m, chunk_strategy, ele_smooth_window
    )


def gpx_xml_to_features(
    gpx: str | fastgpx.Gpx,
    chunk_size_m: float = 200.0,
    chunk_strategy: str = "distance",
    ele_smooth_window: int = 1,
) -> Tuple[np.ndarray, Optional[float]]:
    """End-to-end: GPX XML in memory → (feature_vector, actual_duration_minutes)."""
    segments, duration_min = parse_gpx_xml(gpx)
    return _segments_to_features(
        segments, duration_min, chunk_size_m, chunk_strategy, ele_smooth_window
    )


def _segments_to_features(
    segments: List[List[TrackPoint]],
    duration_min: Optional[float],
    chunk_size_m: float,
    chunk_strategy: str,
    ele_smooth_window: int,
) -> Tuple[np.ndarray, Optional[float]]:
    chunks = chunk_track(
        segments,
        chunk_size_m,
        strategy=chunk_strategy,
        ele_smooth_window=ele_smooth_window,
    )
    X = aggregate_features(chunks)
    return X, duration_min


def describe_gpx(
    path: str | Path,
    chunk_size_m: float = 200.0,
    chunk_strategy: str = "distance",
    ele_smooth_window: int = 1,
) -> dict:
    """Human-readable summary of a GPX file."""
    segments, moving_min = parse_gpx(path)
    chunks = chunk_track(
        segments,
        chunk_size_m,
        strategy=chunk_strategy,
        ele_smooth_window=ele_smooth_window,
    )
    X = aggregate_features(chunks)
    summary = dict(zip(FEATURE_NAMES, X.tolist()))
    summary["moving_duration_min"] = moving_min

    # Also compute total (first→last) duration for comparison
    all_pts = [p for s in segments for p in s]
    if all_pts:
        t0, t1 = all_pts[0].time_s, all_pts[-1].time_s
        summary["total_duration_min"] = (t1 - t0) / 60.0 if t0 > 0 and t1 > t0 else None
    else:
        summary["total_duration_min"] = None

    summary["n_trackpoints"] = sum(len(s) for s in segments)
    return summary
