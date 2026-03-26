"""
server.py — FastAPI prediction server for hiking time estimation.

Usage
─────
  uv run uvicorn server:app --reload           # development
  uv run uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints
─────────
  POST /predict   multipart GPX file → predicted time + route stats
  GET  /health    liveness check
"""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gpx_features import FEATURE_NAMES, gpx_to_features
from model import HikingTimeModel

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model.pkl"))

_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# ── App + model lifecycle ─────────────────────────────────────────────────────

_model: Optional[HikingTimeModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found: {MODEL_PATH}. "
            "Run: python train.py --gpx-dir hiking_tracks --output model.pkl"
        )
    _model = HikingTimeModel.load(MODEL_PATH)
    print(
        f"Loaded model from {MODEL_PATH}  "
        f"(chunk_size={_model.chunk_size_m}m, "
        f"strategy={getattr(_model, 'chunk_strategy', 'distance')})"
    )
    yield
    _model = None


app = FastAPI(
    title="Hiking Time Predictor",
    description="Predicts moving time for a GPX hiking route.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Response schema ───────────────────────────────────────────────────────────


class PredictionResponse(BaseModel):
    predicted_min: float
    predicted_hhmm: str
    distance_km: float
    gain_m: int
    loss_m: int
    tobler_baseline_min: float
    actual_moving_min: Optional[float] = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt(minutes: float) -> str:
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="GPX track file")):
    if not file.filename or not file.filename.lower().endswith(".gpx"):
        raise HTTPException(status_code=400, detail="File must be a .gpx file.")

    # Write upload to a temp file (gpxpy needs a seekable file path)
    suffix = ".gpx"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        strategy = getattr(_model, "chunk_strategy", "distance")
        smooth = getattr(_model, "ele_smooth_window", 1)
        x_vec, actual_min = gpx_to_features(
            tmp_path, _model.chunk_size_m, strategy, smooth
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse GPX: {exc}")
    finally:
        os.unlink(tmp_path)

    pred_min = float(_model.predict_one(x_vec))

    return PredictionResponse(
        predicted_min=round(pred_min, 1),
        predicted_hhmm=_fmt(pred_min),
        distance_km=round(float(x_vec[_IDX["total_dist_km"]]), 2),
        gain_m=int(round(x_vec[_IDX["total_gain_m"]])),
        loss_m=int(round(x_vec[_IDX["total_loss_m"]])),
        tobler_baseline_min=round(float(x_vec[_IDX["total_tobler_min"]]), 1),
        actual_moving_min=round(actual_min, 1) if actual_min is not None else None,
    )
