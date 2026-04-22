from pydantic import BaseModel, Field


# Bounds are based on the historical rookie stat ranges present in training.csv
# and prediction.csv, with a small buffer for plausible edge cases.
GP_MIN = 1.0
GP_MAX = 82.0
MINUTES_MIN = 0.0
MINUTES_MAX = 42.0
FG_PCT_MIN = 0.0
FG_PCT_MAX = 1.0
REB_MIN = 0.0
REB_MAX = 20.0
AST_MIN = 0.0
AST_MAX = 20.0
PTS_MIN = 0.0
PTS_MAX = 35.0
TOV_MIN = 0.0
TOV_MAX = 10.0


class PredictionRequest(BaseModel):
    GP_r: float = Field(
        ...,
        ge=GP_MIN,
        le=GP_MAX,
        description="Rookie games played. Historical rookie data in this project ranges from 1 to 82.",
    )
    MIN_r: float = Field(
        ...,
        ge=MINUTES_MIN,
        le=MINUTES_MAX,
        description="Rookie minutes per game. Historical rookie data in this project peaks around 39.5 MPG.",
    )
    FG_PCT_r: float = Field(
        ...,
        ge=FG_PCT_MIN,
        le=FG_PCT_MAX,
        description="Rookie field goal percentage on a 0 to 1 scale, for example 0.472.",
    )
    REB_r: float = Field(
        ...,
        ge=REB_MIN,
        le=REB_MAX,
        description="Rookie rebounds per game. Historical rookie data in this project peaks around 10.6 RPG.",
    )
    AST_r: float = Field(
        ...,
        ge=AST_MIN,
        le=AST_MAX,
        description="Rookie assists per game. Historical rookie data in this project peaks around 8.2 APG.",
    )
    PTS_r: float = Field(
        ...,
        ge=PTS_MIN,
        le=PTS_MAX,
        description="Rookie points per game. Historical rookie data in this project peaks around 22.5 PPG.",
    )
    TOV_r: float = Field(
        ...,
        ge=TOV_MIN,
        le=TOV_MAX,
        description="Rookie turnovers per game. Historical rookie data in this project peaks around 3.8 TOV.",
    )


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
