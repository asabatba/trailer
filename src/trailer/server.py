"""Compatibility wrapper for the public server import surface."""

from trailer.api.app import app, create_app
from trailer.services.predictor import default_model_path as _default_model_path

__all__ = ["app", "create_app", "_default_model_path"]
