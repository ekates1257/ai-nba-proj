from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from tensorflow import keras

from nba_predictor.config import METADATA_PATH, MODEL_PATH, SCALER_PATH
from nba_predictor.schemas import PredictionRequest


class ArtifactsNotReadyError(RuntimeError):
    pass


class PredictionService:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        self.model = None
        self.scaler = None
        self.metadata: dict = {}
        self.loaded = False
        self.last_error: str | None = None

    def load_artifacts(self) -> None:
        missing_files = self.missing_files()
        if missing_files:
            self.loaded = False
            self.last_error = (
                "Missing model artifacts. Run `python -m nba_predictor.training` first."
            )
            return

        try:
            self.model = keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.metadata = json.loads(self.metadata_path.read_text())
            self.loaded = True
            self.last_error = None
        except Exception as exc:  # pragma: no cover - defensive guard for runtime failures
            self.model = None
            self.scaler = None
            self.metadata = {}
            self.loaded = False
            self.last_error = f"{type(exc).__name__}: {exc}"

    def missing_files(self) -> list[str]:
        missing = []
        for path in (self.model_path, self.scaler_path, self.metadata_path):
            if not path.exists():
                missing.append(path.name)
        return missing

    def health(self) -> dict:
        return {
            "status": "ok" if self.loaded else "degraded",
            "artifacts_loaded": self.loaded,
            "missing_files": self.missing_files(),
            "error": self.last_error,
        }

    def model_info(self) -> dict:
        return {
            "model_version": self.metadata.get("model_version"),
            "trained_at": self.metadata.get("trained_at"),
            "feature_cols": self.metadata.get("feature_cols", []),
            "target_cols": self.metadata.get("target_cols", []),
            "metrics": self.metadata.get("metrics", {}),
            "confidence_interval_90": self.metadata.get("confidence_interval_90"),
        }

    def predict(self, request: PredictionRequest) -> dict:
        if not self.loaded or self.model is None or self.scaler is None:
            raise ArtifactsNotReadyError(
                self.last_error
                or "Prediction artifacts are unavailable. Train the model before serving."
            )

        feature_cols = self.metadata.get("feature_cols", [])
        target_cols = self.metadata.get("target_cols", [])
        if not feature_cols or not target_cols:
            raise ArtifactsNotReadyError(
                "Metadata is incomplete. Re-run training to regenerate artifacts."
            )

        raw_features = np.array(
            [[getattr(request, feature_name) for feature_name in feature_cols]],
            dtype=float,
        )
        scaled_features = self.scaler.transform(raw_features)
        prediction = self.model.predict(scaled_features, verbose=0)[0]

        return {
            "predictions": {
                target_name: float(prediction[idx])
                for idx, target_name in enumerate(target_cols)
            },
            "confidence_interval_90": self.metadata.get("confidence_interval_90"),
            "model_version": self.metadata.get("model_version"),
        }

