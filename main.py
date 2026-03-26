"""Compatibility ASGI entrypoint for uvicorn main:app."""

from trailer.api.app import app, create_app

__all__ = ["app", "create_app"]
