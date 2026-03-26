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
    total_tobler_min: float,
    total_steep_loss_m: float,
    grade_std_mean: float,
    frac_steep: float,
    frac_very_steep: float,
) -> np.ndarray:
    row = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    row[_IDX["total_tobler_min"]] = total_tobler_min
    row[_IDX["total_steep_loss_m"]] = total_steep_loss_m
    row[_IDX["grade_std_mean"]] = grade_std_mean
    row[_IDX["frac_steep"]] = frac_steep
    row[_IDX["frac_very_steep"]] = frac_very_steep
    return row


def synthetic_dataset() -> tuple[np.ndarray, np.ndarray]:
    rows = np.array(
        [
            make_feature_row(
                total_tobler_min=120,
                total_steep_loss_m=10,
                grade_std_mean=0.03,
                frac_steep=0.10,
                frac_very_steep=0.02,
            ),
            make_feature_row(
                total_tobler_min=180,
                total_steep_loss_m=60,
                grade_std_mean=0.05,
                frac_steep=0.15,
                frac_very_steep=0.04,
            ),
            make_feature_row(
                total_tobler_min=240,
                total_steep_loss_m=120,
                grade_std_mean=0.08,
                frac_steep=0.22,
                frac_very_steep=0.07,
            ),
            make_feature_row(
                total_tobler_min=320,
                total_steep_loss_m=180,
                grade_std_mean=0.11,
                frac_steep=0.28,
                frac_very_steep=0.10,
            ),
            make_feature_row(
                total_tobler_min=390,
                total_steep_loss_m=260,
                grade_std_mean=0.16,
                frac_steep=0.35,
                frac_very_steep=0.14,
            ),
        ]
    )
    y = (
        rows[:, _IDX["total_tobler_min"]] * 0.92
        + rows[:, _IDX["total_steep_loss_m"]] * 0.05
        + rows[:, _IDX["grade_std_mean"]] * 180.0
        + rows[:, _IDX["frac_steep"]] * 35.0
        + rows[:, _IDX["frac_very_steep"]] * 70.0
    )
    return rows, y


class ConstrainedModelTests(unittest.TestCase):
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
            "total_tobler_min": 30.0,
            "total_steep_loss_m": 50.0,
            "grade_std_mean": 0.03,
            "frac_steep": 0.10,
            "frac_very_steep": 0.05,
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
