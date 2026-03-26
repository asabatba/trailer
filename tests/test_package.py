import asyncio
import importlib
import io
import sys
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from starlette.datastructures import UploadFile
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import trailer
import trailer.server
from trailer.features import FEATURE_NAMES
from trailer.server import _default_model_path, create_app

api_app_module = importlib.import_module("trailer.api.app")


def make_request(app):
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/",
        "raw_path": b"/",
        "scheme": "http",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope, receive=receive)


class FakeModel:
    chunk_size_m = 600.0
    chunk_strategy = "distance"
    ele_smooth_window = 1

    def predict_one(self, x_vec):
        return 184.2


class PackageTests(unittest.TestCase):
    def test_imports_and_bundled_model_resource_exist(self):
        self.assertIs(trailer.app, trailer.server.app)
        self.assertIs(trailer.app, api_app_module.app)
        self.assertTrue(files("trailer.data").joinpath("model.pkl").is_file())
        self.assertTrue(_default_model_path().exists())

    def test_create_app_exposes_expected_routes(self):
        app = create_app()
        route_paths = {route.path for route in app.routes if hasattr(route, "path")}
        self.assertIn("/health", route_paths)
        self.assertIn("/predict", route_paths)
        self.assertIn("/predict-body", route_paths)
        self.assertIsNone(app.state.model)

    def test_health_returns_model_state_without_startup(self):
        app = create_app()
        health = next(route.endpoint for route in app.routes if route.path == "/health")
        self.assertEqual(health(), {"status": "ok", "model_loaded": False})

    def test_predict_endpoints_match_for_same_input(self):
        app = create_app()
        app.state.model = FakeModel()
        request = make_request(app)
        request.scope["app"] = app

        vector = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
        vector[idx["total_dist_km"]] = 12.37
        vector[idx["total_gain_m"]] = 842
        vector[idx["total_loss_m"]] = 839
        vector[idx["total_tobler_min"]] = 171.6

        predict = next(
            route.endpoint for route in app.routes if route.path == "/predict"
        )
        predict_body = next(
            route.endpoint for route in app.routes if route.path == "/predict-body"
        )

        with patch(
            "trailer.services.predictor.gpx_to_features", return_value=(vector, 180.0)
        ):
            file = UploadFile(filename="route.gpx", file=io.BytesIO(b"<gpx></gpx>"))
            multipart_result = asyncio.run(predict(request, file))
            body_result = asyncio.run(predict_body(request, b"<gpx></gpx>"))

        self.assertEqual(multipart_result.model_dump(), body_result.model_dump())
        self.assertEqual(multipart_result.predicted_hhmm, "3h 04m")

    def test_predict_rejects_non_gpx_filename(self):
        app = create_app()
        app.state.model = FakeModel()
        request = make_request(app)
        request.scope["app"] = app
        predict = next(
            route.endpoint for route in app.routes if route.path == "/predict"
        )
        bad_file = UploadFile(filename="route.txt", file=io.BytesIO(b"not-gpx"))

        with self.assertRaises(HTTPException) as exc:
            asyncio.run(predict(request, bad_file))

        self.assertEqual(exc.exception.status_code, 400)

    def test_predict_body_rejects_empty_payload(self):
        app = create_app()
        app.state.model = FakeModel()
        request = make_request(app)
        request.scope["app"] = app
        predict_body = next(
            route.endpoint for route in app.routes if route.path == "/predict-body"
        )

        with self.assertRaises(HTTPException) as exc:
            asyncio.run(predict_body(request, b""))

        self.assertEqual(exc.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
