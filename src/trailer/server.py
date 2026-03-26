"""FastAPI prediction server for hiking time estimation."""

import os
import tempfile
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from trailer.features import FEATURE_NAMES, gpx_to_features
from trailer.model import HikingTimeModel

_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


class PredictionResponse(BaseModel):
    predicted_min: float
    predicted_hhmm: str
    distance_km: float
    gain_m: int
    loss_m: int
    tobler_baseline_min: float
    actual_moving_min: Optional[float] = None


def _fmt(minutes: float) -> str:
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def _default_model_path() -> Path:
    override = os.environ.get("MODEL_PATH")
    if override:
        return Path(override)
    return Path(str(files("trailer.data").joinpath("model.pkl")))


def _predict_from_gpx_bytes(
    gpx_bytes: bytes, model: HikingTimeModel
) -> PredictionResponse:
    if not gpx_bytes.strip():
        raise HTTPException(status_code=400, detail="Request body is empty.")

    # gpxpy expects a seekable file path, so persist the incoming payload briefly.
    with tempfile.NamedTemporaryFile(suffix=".gpx", delete=False) as tmp:
        tmp.write(gpx_bytes)
        tmp_path = tmp.name

    try:
        strategy = getattr(model, "chunk_strategy", "distance")
        smooth = getattr(model, "ele_smooth_window", 1)
        x_vec, actual_min = gpx_to_features(
            tmp_path, model.chunk_size_m, strategy, smooth
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse GPX: {exc}")
    finally:
        os.unlink(tmp_path)

    pred_min = float(model.predict_one(x_vec))

    return PredictionResponse(
        predicted_min=round(pred_min, 1),
        predicted_hhmm=_fmt(pred_min),
        distance_km=round(float(x_vec[_IDX["total_dist_km"]]), 2),
        gain_m=int(round(x_vec[_IDX["total_gain_m"]])),
        loss_m=int(round(x_vec[_IDX["total_loss_m"]])),
        tobler_baseline_min=round(float(x_vec[_IDX["total_tobler_min"]]), 1),
        actual_moving_min=round(actual_min, 1) if actual_min is not None else None,
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model_path = _default_model_path()
        if not model_path.exists():
            raise RuntimeError(
                f"Model not found: {model_path}. "
                "Run: trailer-train --gpx-dir hiking_tracks --output model.pkl"
            )
        app.state.model = HikingTimeModel.load(model_path)
        print(
            f"Loaded model from {model_path}  "
            f"(chunk_size={app.state.model.chunk_size_m}m, "
            f"strategy={getattr(app.state.model, 'chunk_strategy', 'distance')})"
        )
        yield
        app.state.model = None

    app = FastAPI(
        title="Hiking Time Predictor",
        description="Predicts moving time for a GPX hiking route.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.model = None

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.model is not None}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: Request,
        file: UploadFile = File(..., description="GPX track file"),
    ):
        if not file.filename or not file.filename.lower().endswith(".gpx"):
            raise HTTPException(status_code=400, detail="File must be a .gpx file.")
        return _predict_from_gpx_bytes(await file.read(), request.app.state.model)

    @app.post("/predict-body", response_model=PredictionResponse)
    async def predict_body(
        request: Request,
        gpx_body: bytes = Body(
            ...,
            media_type="application/gpx+xml",
            description="Raw GPX XML content in the request body.",
        ),
    ):
        return _predict_from_gpx_bytes(gpx_body, request.app.state.model)

    return app


app = create_app()
