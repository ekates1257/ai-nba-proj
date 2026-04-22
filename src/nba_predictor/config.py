from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
DEFAULT_TRAINING_DATA_PATH = ROOT_DIR / "training.csv"
DEFAULT_PREDICTION_DATA_PATH = ROOT_DIR / "prediction.csv"
