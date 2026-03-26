"""API request and response schemas."""

from typing import Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_min: float
    predicted_hhmm: str
    distance_km: float
    gain_m: int
    loss_m: int
    tobler_baseline_min: float
    actual_moving_min: Optional[float] = None
