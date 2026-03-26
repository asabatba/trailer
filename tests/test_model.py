import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trailer.features import FEATURE_NAMES, gpx_to_features
from trailer.model import (
    CONSTRAINED_FEATURES,
    HikingTimeModel,
    build_monotone_design_matrix,
)
from trailer.services.predictor import load_model, predict_from_gpx_bytes

_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def make_feature_row(
    *,
    total_dist_km: float,
    total_tobler_min: float,
    best_case_distance_min: float,
    slope_penalty_min: float,
    vertical_change_m: float,
) -> np.ndarray:
    row = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    row[_IDX["total_dist_km"]] = total_dist_km
    row[_IDX["total_tobler_min"]] = total_tobler_min
    row[_IDX["best_case_distance_min"]] = best_case_distance_min
    row[_IDX["slope_penalty_min"]] = slope_penalty_min
    row[_IDX["vertical_change_m"]] = vertical_change_m
    return row


def synthetic_dataset() -> tuple[np.ndarray, np.ndarray]:
    rows = np.array(
        [
            make_feature_row(
                total_dist_km=9.0,
                total_tobler_min=118,
                best_case_distance_min=90,
                slope_penalty_min=28,
                vertical_change_m=320,
            ),
            make_feature_row(
                total_dist_km=12.0,
                total_tobler_min=168,
                best_case_distance_min=120,
                slope_penalty_min=48,
                vertical_change_m=540,
            ),
            make_feature_row(
                total_dist_km=15.0,
                total_tobler_min=224,
                best_case_distance_min=150,
                slope_penalty_min=74,
                vertical_change_m=760,
            ),
            make_feature_row(
                total_dist_km=19.0,
                total_tobler_min=292,
                best_case_distance_min=190,
                slope_penalty_min=102,
                vertical_change_m=1040,
            ),
            make_feature_row(
                total_dist_km=24.0,
                total_tobler_min=378,
                best_case_distance_min=240,
                slope_penalty_min=138,
                vertical_change_m=1460,
            ),
        ]
    )
    y = (
        rows[:, _IDX["best_case_distance_min"]] * 1.22
        + rows[:, _IDX["slope_penalty_min"]] * 0.09
        + rows[:, _IDX["vertical_change_m"]] * 0.031
    )
    return rows, y


class ConstrainedModelTests(unittest.TestCase):
    def test_new_physics_features_match_real_route_aggregates(self):
        path = ROOT / "hiking_tracks" / "Asni Hiking - 11.5km +1358m -70m.gpx"
        x_vec, _ = gpx_to_features(path)

        dist_km = x_vec[_IDX["total_dist_km"]]
        gain_m = x_vec[_IDX["total_gain_m"]]
        loss_m = x_vec[_IDX["total_loss_m"]]
        tobler_min = x_vec[_IDX["total_tobler_min"]]

        self.assertAlmostEqual(x_vec[_IDX["best_case_distance_min"]], dist_km * 10.0)
        self.assertAlmostEqual(
            x_vec[_IDX["slope_penalty_min"]],
            max(tobler_min - dist_km * 10.0, 0.0),
        )
        self.assertAlmostEqual(
            x_vec[_IDX["vertical_change_m"]],
            gain_m + loss_m,
        )

    def test_monotone_design_matrix_is_nonnegative_for_real_route(self):
        path = ROOT / "hiking_tracks" / "Asni Hiking - 11.5km +1358m -70m.gpx"
        x_vec, _ = gpx_to_features(path)

        design = build_monotone_design_matrix(x_vec)

        self.assertEqual(design.shape, (1, len(CONSTRAINED_FEATURES)))
        self.assertTrue(np.all(design >= 0.0))

    def test_excluded_features_do_not_participate(self):
        X, y = synthetic_dataset()
        model = HikingTimeModel(ridge_alpha=0.01).fit(X, y)
        base = X[2].copy()
        pred = model.predict_one(base)

        changed = base.copy()
        changed[_IDX["mean_grade"]] = 10.0
        changed[_IDX["tobler_efficiency"]] = 99.0
        changed[_IDX["n_chunks"]] = 500.0
        changed[_IDX["total_steep_loss_m"]] = 700.0

        self.assertAlmostEqual(model.predict_one(changed), pred, places=9)

    def test_fitted_coefficients_are_nonnegative(self):
        X, y = synthetic_dataset()
        model = HikingTimeModel(ridge_alpha=0.01).fit(X, y)

        coeffs = model.coefficients()

        self.assertEqual(set(coeffs), set(CONSTRAINED_FEATURES))
        self.assertTrue(all(coef >= 0.0 for coef in coeffs.values()))

    def test_increasing_any_constrained_feature_cannot_reduce_prediction(self):
        X, y = synthetic_dataset()
        model = HikingTimeModel(ridge_alpha=0.01).fit(X, y)
        base = X[1].copy()
        base_pred = model.predict_one(base)

        deltas = {
            "best_case_distance_min": 30.0,
            "slope_penalty_min": 15.0,
            "vertical_change_m": 180.0,
        }
        for name, delta in deltas.items():
            with self.subTest(feature=name):
                changed = base.copy()
                changed[_IDX[name]] += delta
                self.assertGreaterEqual(model.predict_one(changed), base_pred)

    def test_zero_vector_predicts_zero_or_positive(self):
        X, y = synthetic_dataset()
        model = HikingTimeModel(ridge_alpha=0.01).fit(X, y)

        pred = model.predict_one(np.zeros(len(FEATURE_NAMES), dtype=np.float64))

        self.assertGreaterEqual(pred, 0.0)
        self.assertAlmostEqual(pred, 0.0, places=9)


class BundledModelTests(unittest.TestCase):
    def test_bundled_model_loads_and_predicts_real_gpx(self):
        model = load_model()
        gpx_path = ROOT / "hiking_tracks" / "Funchal Hiking - 14.7km +1019m -595m.gpx"

        response = predict_from_gpx_bytes(gpx_path.read_bytes(), model)

        self.assertGreater(response.predicted_min, 0.0)
        self.assertTrue(response.predicted_hhmm)
        self.assertGreater(response.distance_km, 0.0)
        self.assertGreaterEqual(response.gain_m, 0)
        self.assertGreaterEqual(response.loss_m, 0)
        self.assertGreaterEqual(response.tobler_baseline_min, 0.0)


if __name__ == "__main__":
    unittest.main()
