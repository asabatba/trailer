"""FastAPI application wiring."""

from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile

from trailer.api.schemas import PredictionResponse
from trailer.services.predictor import load_model, predict_from_gpx_bytes


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = load_model()
        print(
            f"Loaded model "
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
        return predict_from_gpx_bytes(await file.read(), request.app.state.model)

    @app.post("/predict-body", response_model=PredictionResponse)
    async def predict_body(
        request: Request,
        gpx_body: bytes = Body(
            ...,
            media_type="application/gpx+xml",
            description="Raw GPX XML content in the request body.",
        ),
    ):
        return predict_from_gpx_bytes(gpx_body, request.app.state.model)

    return app


app = create_app()
