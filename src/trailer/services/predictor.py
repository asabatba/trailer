"""Prediction and model-loading services."""

import os
import re
from importlib.resources import files
from pathlib import Path

from fastapi import HTTPException

from trailer.api.schemas import PredictionResponse
from trailer.features import FEATURE_NAMES, gpx_xml_to_features
from trailer.model import HikingTimeModel

_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
_XML_ENCODING_RE = re.compile(
    rb"""<\?xml[^>]*encoding=["']([^"']+)["']""", re.IGNORECASE
)


def format_duration(minutes: float) -> str:
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def default_model_path() -> Path:
    override = os.environ.get("MODEL_PATH")
    if override:
        return Path(override)
    return Path(str(files("trailer.data").joinpath("model.pkl")))


def load_model() -> HikingTimeModel:
    model_path = default_model_path()
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found: {model_path}. "
            "Run: trailer-train --gpx-dir hiking_tracks --output model.pkl"
        )
    return HikingTimeModel.load(model_path)


def predict_from_gpx_bytes(
    gpx_bytes: bytes,
    model: HikingTimeModel,
) -> PredictionResponse:
    if not gpx_bytes.strip():
        raise HTTPException(status_code=400, detail="Request body is empty.")

    try:
        strategy = getattr(model, "chunk_strategy", "distance")
        smooth = getattr(model, "ele_smooth_window", 1)
        gpx_xml = _decode_gpx_bytes(gpx_bytes)
        x_vec, actual_min = gpx_xml_to_features(
            gpx_xml, model.chunk_size_m, strategy, smooth
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse GPX: {exc}")

    pred_min = float(model.predict_one(x_vec))

    return PredictionResponse(
        predicted_min=round(pred_min, 1),
        predicted_hhmm=format_duration(pred_min),
        distance_km=round(float(x_vec[_IDX["total_dist_km"]]), 2),
        gain_m=int(round(x_vec[_IDX["total_gain_m"]])),
        loss_m=int(round(x_vec[_IDX["total_loss_m"]])),
        tobler_baseline_min=round(float(x_vec[_IDX["total_tobler_min"]]), 1),
        actual_moving_min=round(actual_min, 1) if actual_min is not None else None,
    )


def _decode_gpx_bytes(gpx_bytes: bytes) -> str:
    header = gpx_bytes[:256]
    match = _XML_ENCODING_RE.search(header)
    encoding = match.group(1).decode("ascii") if match else "utf-8-sig"
    try:
        return gpx_bytes.decode(encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not decode GPX XML using {encoding!r}: {exc}") from exc
