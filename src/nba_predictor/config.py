import os
from pathlib import Path


ROOT_DIR = Path(os.getenv("NBA_PREDICTOR_ROOT", Path.cwd())).resolve()
ARTIFACTS_DIR = Path(os.getenv("NBA_PREDICTOR_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))
MODEL_PATH = Path(os.getenv("NBA_PREDICTOR_MODEL_PATH", ARTIFACTS_DIR / "model.keras"))
SCALER_PATH = Path(os.getenv("NBA_PREDICTOR_SCALER_PATH", ARTIFACTS_DIR / "scaler.joblib"))
METADATA_PATH = Path(
    os.getenv("NBA_PREDICTOR_METADATA_PATH", ARTIFACTS_DIR / "metadata.json")
)
DEFAULT_TRAINING_DATA_PATH = Path(
    os.getenv("NBA_PREDICTOR_TRAINING_DATA_PATH", ROOT_DIR / "training.csv")
)
DEFAULT_PREDICTION_DATA_PATH = Path(
    os.getenv("NBA_PREDICTOR_PREDICTION_DATA_PATH", ROOT_DIR / "prediction.csv")
)
