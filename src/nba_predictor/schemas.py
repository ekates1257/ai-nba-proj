from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    GP_r: float = Field(..., description="Rookie games played")
    MIN_r: float = Field(..., description="Rookie minutes per game")
    FG_PCT_r: float = Field(..., description="Rookie field goal percentage")
    REB_r: float = Field(..., description="Rookie rebounds per game")
    AST_r: float = Field(..., description="Rookie assists per game")
    PTS_r: float = Field(..., description="Rookie points per game")
    TOV_r: float = Field(..., description="Rookie turnovers per game")


class PredictionResponse(BaseModel):
    predictions: dict[str, float]
    confidence_interval_90: dict[str, float] | None = None
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    artifacts_loaded: bool
    missing_files: list[str]
    error: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str | None = None
    trained_at: str | None = None
    feature_cols: list[str]
    target_cols: list[str]
    metrics: dict[str, float]
    confidence_interval_90: dict[str, float] | None = None

