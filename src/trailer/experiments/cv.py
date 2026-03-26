"""Shared helpers for experiment sweeps."""

from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


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
    residual_cols = [i for i in range(X.shape[1]) if i != tobler_col]

    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        X_phys = X[:, [tobler_col]]
        phys = LinearRegression(fit_intercept=False)
        phys.fit(X_phys[train_idx], y[train_idx])
        pred = phys.predict(X_phys[test_idx])

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
