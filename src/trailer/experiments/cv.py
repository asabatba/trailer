"""Shared helpers for experiment sweeps."""

from typing import List, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut

from trailer.model import HikingTimeModel


def loo_cv_two_stage(
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    *,
    tobler_col: int,
    ridge_alpha: float,
    min_n_residual: int = 12,
    verbose: bool = False,
    route_width: int = 42,
    actual_header: str = "Actual",
    pred_header: str = "Pred",
    error_header: str = "Error",
) -> Tuple[float, float]:
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        # Preserve the historical helper signature for sweep scripts, but use
        # the project's current constrained model implementation.
        _ = tobler_col
        model = HikingTimeModel(
            ridge_alpha=ridge_alpha,
            min_samples_for_residual=min_n_residual,
        )
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])

    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred) * 100

    if verbose:
        errors = y_pred - y
        print(
            f"\n  {'Route':{route_width}s} "
            f"{actual_header:>8} {pred_header:>8} {error_header:>8}"
        )
        print(f"  {'-' * route_width} {'-' * 8} {'-' * 8} {'-' * 8}")
        for name, actual, pred, err in zip(names, y, y_pred, errors):
            flag = "  <!" if abs(err) > 30 else ""
            print(
                f"  {name:{route_width}s} {actual:8.1f} {pred:8.1f} {err:+8.1f}{flag}"
            )

    return mae, mape
