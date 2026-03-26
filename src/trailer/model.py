"""
model.py
────────
Hiking time prediction model designed for small datasets (~20 GPX files).

Architecture
────────────
With N≈20, unconstrained ML models can overfit and learn nonsensical
relationships.  This model uses a small monotone design matrix made of
nonnegative, physics-aligned route features:

    ŷ = α·tobler_min
      + β·steep_descent_penalty
      + γ·roughness_penalty
      + δ·steep_fraction_penalty
      + ε·very_steep_fraction_penalty

All coefficients are constrained nonnegative and there is no intercept, so
increasing any learned effort channel cannot reduce predicted time.

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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from trailer.features import FEATURE_NAMES, gpx_to_features

# Indices into the feature vector (keep in sync with FEATURE_NAMES)
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Nonnegative, physically meaningful channels only.  The model ignores the rest
# of the aggregated feature vector to keep monotonicity obvious by construction.
CONSTRAINED_FEATURES = [
    "total_tobler_min",
    "total_steep_loss_m",
    "grade_std_mean",
    "frac_steep",
    "frac_very_steep",
]
CONSTRAINED_IDX = [_IDX[f] for f in CONSTRAINED_FEATURES]


def build_monotone_design_matrix(X: np.ndarray) -> np.ndarray:
    """
    Select the constrained feature subset used by the model.

    The chosen channels are naturally nonnegative for valid route features.
    Clamp at zero as a final safeguard against malformed inputs or tiny
    numerical negatives.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    assert X.shape[1] == len(FEATURE_NAMES), (
        f"Expected {len(FEATURE_NAMES)} features, got {X.shape[1]}"
    )
    return np.maximum(X[:, CONSTRAINED_IDX], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Model class
# ─────────────────────────────────────────────────────────────────────────────


class HikingTimeModel:
    """
    Parameters
    ----------
    ridge_alpha : float
        Regularisation strength for the constrained Ridge regressor.
        Higher → more conservative penalties.  Default 0.5.
    min_samples_for_residual : int
        Retained for compatibility with earlier call sites; unused by the
        constrained model.
    chunk_size_m : float
        Chunk size used during feature extraction.
    """

    def __init__(
        self,
        ridge_alpha: float = 0.5,
        min_samples_for_residual: int = 12,
        chunk_size_m: float = 600.0,
        chunk_strategy: str = "distance",
        ele_smooth_window: int = 1,
    ):
        self.ridge_alpha = ridge_alpha
        self.min_samples_for_residual = min_samples_for_residual
        self.chunk_size_m = chunk_size_m
        self.chunk_strategy = chunk_strategy
        self.ele_smooth_window = ele_smooth_window

        self._model: Optional[Ridge] = None
        self._scaler: Optional[StandardScaler] = None
        self.feature_names = FEATURE_NAMES
        self.design_feature_names = CONSTRAINED_FEATURES

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HikingTimeModel":
        """
        X : (N, len(FEATURE_NAMES)) feature matrix
        y : (N,) actual hiking times in minutes
        """
        assert X.shape[1] == len(FEATURE_NAMES), (
            f"Expected {len(FEATURE_NAMES)} features, got {X.shape[1]}"
        )

        X_design = build_monotone_design_matrix(X)
        # Keep zero as the physical origin; centering would destroy the direct
        # monotonic interpretation of positive coefficients.
        self._scaler = StandardScaler(with_mean=False)
        X_scaled = self._scaler.fit_transform(X_design)
        self._model = Ridge(
            alpha=self.ridge_alpha,
            fit_intercept=False,
            positive=True,
        )
        self._model.fit(X_scaled, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None and self._scaler is not None, "Call fit() first"
        X_scaled = self._scaler.transform(build_monotone_design_matrix(X))
        pred = self._model.predict(X_scaled)
        return np.maximum(pred, 0.0)  # times can't be negative

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict(x.reshape(1, -1))[0])

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        joblib.dump(self, path)
        print(f"  Model saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "HikingTimeModel":
        return joblib.load(path)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def coefficients(self) -> Dict[str, float]:
        """
        Returns fitted coefficients on the original feature scale.

        StandardScaler(with_mean=False) divides inputs by scale_, so the raw
        coefficient for feature i is coef_i / scale_i.
        """
        if self._model is None or self._scaler is None:
            return {}

        scale = np.where(self._scaler.scale_ == 0, 1.0, self._scaler.scale_)
        raw_coef = self._model.coef_ / scale
        return {
            name: float(coef)
            for name, coef in zip(self.design_feature_names, raw_coef.tolist())
        }

    def feature_importance(self) -> Dict[str, float]:
        """
        Returns approximate importance scores for constrained coefficients.
        """
        importance = {name: abs(coef) for name, coef in self.coefficients().items()}

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
    chunk_strategy: str = "distance",
    ele_smooth_window: int = 7,
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
            x_vec, gps_duration = gpx_to_features(
                path, chunk_size_m, chunk_strategy, ele_smooth_window
            )
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
            f"  + {path.name:40s}  {duration:7.1f} min  "
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
    chunk_strategy: str = "distance",
    ele_smooth_window: int = 7,
) -> Dict:
    """Leave-One-Out cross-validation report."""
    loo = LeaveOneOut()
    y_pred_loo = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        m = HikingTimeModel(
            ridge_alpha=ridge_alpha,
            chunk_size_m=chunk_size_m,
            chunk_strategy=chunk_strategy,
            ele_smooth_window=ele_smooth_window,
        )
        m.fit(X[train_idx], y[train_idx])
        y_pred_loo[test_idx] = m.predict(X[test_idx])

    errors = y_pred_loo - y
    mae = mean_absolute_error(y, y_pred_loo)
    mape = mean_absolute_percentage_error(y, y_pred_loo) * 100

    print("\n--- LOO-CV Results " + "-" * 44)
    print(f"  MAE  : {mae:.1f} min")
    print(f"  MAPE : {mape:.1f} %")
    print(f"\n  {'Route':40s} {'Actual':>8} {'Pred':>8} {'Error':>8}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 8} {'-' * 8}")
    for name, actual, pred, err in zip(names, y, y_pred_loo, errors):
        flag = "  <- !" if abs(err) > 30 else ""
        print(f"  {name:40s} {actual:8.1f} {pred:8.1f} {err:+8.1f}{flag}")

    return {
        "mae_min": mae,
        "mape_pct": mape,
        "y_true": y,
        "y_pred": y_pred_loo,
        "names": names,
    }
