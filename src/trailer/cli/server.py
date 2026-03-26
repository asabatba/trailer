"""Console entrypoint for the ASGI server."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trailer API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload")
    args = parser.parse_args()

    uvicorn.run(
        "trailer.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
