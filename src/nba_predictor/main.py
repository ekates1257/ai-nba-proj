from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from nba_predictor.inference import ArtifactsNotReadyError, PredictionService
from nba_predictor.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


service = PredictionService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load_artifacts()
    yield


app = FastAPI(
    title="NBA Rookie Progression Predictor API",
    version="0.1.0",
    description="Serve sophomore stat predictions from rookie-year inputs.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(**service.health())


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**service.model_info())


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        return PredictionResponse(**service.predict(request))
    except ArtifactsNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

