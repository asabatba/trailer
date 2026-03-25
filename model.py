"""
model.py
────────
Hiking time prediction model designed for small datasets (~20 GPX files).

Architecture
────────────
With N=20, classic ML models overfit badly.  We use a two-stage approach:

  Stage 1 – Physics prior (Tobler calibration)
    A simple 4-parameter OLS fit:
        ŷ = α·tobler_min + β·total_gain_m + γ·total_loss_m + δ
    This is essentially "how much does your actual pace deviate from
    Tobler's theoretical formula?"  Works well even with 5 samples.

  Stage 2 – Residual correction (optional Ridge)
    If N ≥ 12, a Ridge regressor is stacked on the Stage-1 residuals
    using the remaining features.  This captures terrain roughness and
    gradient distribution effects that Tobler misses.

The final prediction is stage1 + stage2_residual_correction.

Evaluation
──────────
Uses Leave-One-Out cross-validation (LOO-CV), the correct choice when
N < 30: every sample gets to be the test set exactly once.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from gpx_features import FEATURE_NAMES, gpx_to_features

# Indices into the feature vector (keep in sync with FEATURE_NAMES)
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

PHYSICS_FEATURES = ["total_tobler_min", "total_gain_m", "total_loss_m"]
PHYSICS_IDX = [_IDX[f] for f in PHYSICS_FEATURES]
RESIDUAL_FEATURES = [f for f in FEATURE_NAMES if f not in PHYSICS_FEATURES]
RESIDUAL_IDX = [_IDX[f] for f in RESIDUAL_FEATURES]


# ─────────────────────────────────────────────────────────────────────────────
# Model class
# ─────────────────────────────────────────────────────────────────────────────


class HikingTimeModel:
    """
    Parameters
    ----------
    ridge_alpha : float
        Regularisation strength for the residual Ridge corrector.
        Higher → more conservative corrections.  Default 10.0.
    min_samples_for_residual : int
        Minimum N before the residual corrector is added.  Below this
        only the physics-calibration stage is used.
    chunk_size_m : float
        Chunk size used during feature extraction.
    """

    def __init__(
        self,
        ridge_alpha: float = 10.0,
        min_samples_for_residual: int = 12,
        chunk_size_m: float = 200.0,
    ):
        self.ridge_alpha = ridge_alpha
        self.min_samples_for_residual = min_samples_for_residual
        self.chunk_size_m = chunk_size_m

        self._physics_model: Optional[LinearRegression] = None
        self._residual_model: Optional[Ridge] = None
        self._scaler: Optional[StandardScaler] = None
        self._use_residual = False
        self.feature_names = FEATURE_NAMES

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HikingTimeModel":
        """
        X : (N, len(FEATURE_NAMES)) feature matrix
        y : (N,) actual hiking times in minutes
        """
        N = len(y)
        assert X.shape[1] == len(FEATURE_NAMES), (
            f"Expected {len(FEATURE_NAMES)} features, got {X.shape[1]}"
        )

        # Stage 1 – physics calibration
        X_phys = X[:, PHYSICS_IDX]
        self._physics_model = LinearRegression()
        self._physics_model.fit(X_phys, y)

        # Stage 2 – residual correction
        self._use_residual = N >= self.min_samples_for_residual
        if self._use_residual:
            residuals = y - self._physics_model.predict(X_phys)
            X_resid = X[:, RESIDUAL_IDX]
            self._scaler = StandardScaler()
            X_resid_scaled = self._scaler.fit_transform(X_resid)
            self._residual_model = Ridge(alpha=self.ridge_alpha)
            self._residual_model.fit(X_resid_scaled, residuals)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._physics_model is not None, "Call fit() first"
        pred = self._physics_model.predict(X[:, PHYSICS_IDX])
        if self._use_residual:
            X_resid_scaled = self._scaler.transform(X[:, RESIDUAL_IDX])
            pred += self._residual_model.predict(X_resid_scaled)
        return np.maximum(pred, 0.0)  # times can't be negative

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict(x.reshape(1, -1))[0])

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        joblib.dump(self, path)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "HikingTimeModel":
        return joblib.load(path)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def feature_importance(self) -> Dict[str, float]:
        """
        Returns approximate importance scores.
        Physics stage coefficients are on original scale (interpretable).
        Residual stage uses scaled coefficients.
        """
        importance: Dict[str, float] = {}

        if self._physics_model is not None:
            for name, coef in zip(PHYSICS_FEATURES, self._physics_model.coef_):
                importance[name] = abs(coef)

        if self._use_residual and self._residual_model is not None:
            for name, coef in zip(RESIDUAL_FEATURES, self._residual_model.coef_):
                importance[name] = abs(coef)

        # Normalise to [0, 1]
        if importance:
            max_val = max(importance.values()) or 1.0
            importance = {
                k: v / max_val
                for k, v in sorted(importance.items(), key=lambda kv: -kv[1])
            }
        return importance


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────


def build_dataset(
    gpx_paths: List[Path],
    labels: Optional[Dict[str, float]] = None,
    chunk_size_m: float = 200.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build (X, y, names) from a list of GPX files.

    labels : dict mapping filename stem → actual_minutes.
             If None, durations are extracted from GPX timestamps.
             Files without a valid label are skipped with a warning.
    """
    X_rows, y_vals, names = [], [], []

    for path in sorted(gpx_paths):
        try:
            x_vec, gps_duration = gpx_to_features(path, chunk_size_m)
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {exc}")
            continue

        # Resolve label
        if labels is not None:
            duration = labels.get(path.stem) or labels.get(path.name)
            if duration is None:
                warnings.warn(f"No label for {path.name}, skipping.")
                continue
        else:
            duration = gps_duration
            if duration is None:
                warnings.warn(
                    f"{path.name} has no timestamps and no manual label — skipping."
                )
                continue

        X_rows.append(x_vec)
        y_vals.append(duration)
        names.append(path.stem)
        print(
            f"  ✓ {path.name:40s}  {duration:7.1f} min  "
            f"({x_vec[_IDX['total_dist_km']]:.1f} km, "
            f"+{x_vec[_IDX['total_gain_m']]:.0f} m)"
        )

    return np.array(X_rows), np.array(y_vals), names


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────


def loo_cv(
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    ridge_alpha: float = 10.0,
    chunk_size_m: float = 200.0,
) -> Dict:
    """Leave-One-Out cross-validation report."""
    loo = LeaveOneOut()
    y_pred_loo = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        m = HikingTimeModel(ridge_alpha=ridge_alpha, chunk_size_m=chunk_size_m)
        m.fit(X[train_idx], y[train_idx])
        y_pred_loo[test_idx] = m.predict(X[test_idx])

    errors = y_pred_loo - y
    mae = mean_absolute_error(y, y_pred_loo)
    mape = mean_absolute_percentage_error(y, y_pred_loo) * 100

    print("\n─── LOO-CV Results ─────────────────────────────────────────────")
    print(f"  MAE  : {mae:.1f} min")
    print(f"  MAPE : {mape:.1f} %")
    print(f"\n  {'Route':40s} {'Actual':>8} {'Pred':>8} {'Error':>8}")
    print(f"  {'─' * 40} {'─' * 8} {'─' * 8} {'─' * 8}")
    for name, actual, pred, err in zip(names, y, y_pred_loo, errors):
        flag = "  ← !" if abs(err) > 30 else ""
        print(f"  {name:40s} {actual:8.1f} {pred:8.1f} {err:+8.1f}{flag}")

    return {
        "mae_min": mae,
        "mape_pct": mape,
        "y_true": y,
        "y_pred": y_pred_loo,
        "names": names,
    }
