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

import gpxpy
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
        tobler_min = 0.0
        for d_h, grad in zip(horiz_distances, gradients):
            speed_kmh = 6.0 * math.exp(-3.5 * abs(grad + 0.05))
            speed_kmh = max(speed_kmh, 0.1)  # floor: 100 m/h
            tobler_min += (d_h / 1000.0) / speed_kmh * 60.0

        self._features = {
            "dist_m": total_dist_m,
            "gain_m": gain_m,
            "loss_m": loss_m,
            "mean_grade": mean_grade,
            "max_grade": max_grade,
            "grade_std": grade_std,
            "tobler_min": tobler_min,
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
    with open(path) as f:
        gpx = gpxpy.parse(f)

    segments: List[List[TrackPoint]] = []
    for track in gpx.tracks:
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
            for w in gpx.waypoints
        ]
        if wpt_pts:
            segments = [wpt_pts]

    if not segments:
        return [], None

    # Duration: first point of first segment → last point of last segment
    all_pts = [p for s in segments for p in s]
    t_start = all_pts[0].time_s
    t_end = all_pts[-1].time_s
    duration_min: Optional[float] = None
    if t_start > 0 and t_end > t_start:
        duration_min = (t_end - t_start) / 60.0

    return segments, duration_min


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


def chunk_track(
    segments: List[List[TrackPoint]], chunk_size_m: float = 200.0
) -> List[Chunk]:
    """
    Split track segments into chunks of ~chunk_size_m horizontal distance.
    Each GPX segment is processed independently to avoid connecting
    non-adjacent waypoints across segment boundaries.
    The last chunk of each segment may be shorter than chunk_size_m.
    """
    chunks: List[Chunk] = []
    for seg_pts in segments:
        chunks.extend(_chunk_segment(seg_pts, chunk_size_m))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Feature aggregation
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # ── Global totals (strong signals)
    "total_dist_km",
    "total_gain_m",
    "total_loss_m",
    "total_tobler_min",  # ← single best predictor
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
    total_dist_km = total_dist_m / 1000.0

    grades = np.array([f["mean_grade"] for f in feats])
    max_grades = np.array([f["max_grade"] for f in feats])
    stds = np.array([f["grade_std"] for f in feats])
    toblers = np.array([f["tobler_min"] for f in feats])

    frac_steep = float(np.mean(np.abs(grades) > 0.25))
    frac_very_steep = float(np.mean(np.abs(grades) > 0.40))

    eps = 1e-6
    vec = [
        total_dist_km,
        total_gain_m,
        total_loss_m,
        total_tobler,
        float(np.mean(grades)),
        float(np.mean(np.abs(grades))),
        float(np.max(max_grades)),
        float(np.mean(stds)),
        float(np.percentile(toblers, 75)),
        float(np.percentile(toblers, 90)),
        frac_steep,
        frac_very_steep,
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
) -> Tuple[np.ndarray, Optional[float]]:
    """
    End-to-end: GPX file → (feature_vector, actual_duration_minutes).
    actual_duration_minutes is None if the file has no timestamps.
    """
    segments, duration_min = parse_gpx(path)
    chunks = chunk_track(segments, chunk_size_m)
    X = aggregate_features(chunks)
    return X, duration_min


def describe_gpx(path: str | Path, chunk_size_m: float = 200.0) -> dict:
    """Human-readable summary of a GPX file."""
    segments, duration_min = parse_gpx(path)
    chunks = chunk_track(segments, chunk_size_m)
    X = aggregate_features(chunks)
    summary = dict(zip(FEATURE_NAMES, X.tolist()))
    summary["actual_duration_min"] = duration_min
    summary["n_trackpoints"] = sum(len(s) for s in segments)
    return summary
