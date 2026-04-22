from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

from nba_predictor.config import (
    ARTIFACTS_DIR,
    DEFAULT_TRAINING_DATA_PATH,
    METADATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
)


FEATURE_COLS = ["GP_r", "MIN_r", "FG_PCT_r", "REB_r", "AST_r", "PTS_r", "TOV_r"]
TARGET_COLS = ["PTS_s", "REB_s", "AST_s"]


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=regularizers.l1(0.001),
            ),
            keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizers.l1(0.001),
            ),
            keras.layers.Dense(len(TARGET_COLS)),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_and_save(
    training_data_path: Path = DEFAULT_TRAINING_DATA_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    epochs: int = 300,
    batch_size: int = 32,
) -> dict:
    df = pd.read_csv(training_data_path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(input_dim=X_train_scaled.shape[1])
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    predictions = model.predict(X_test_scaled, verbose=0)
    errors = y_test.to_numpy() - predictions
    confidence = 1.645 * np.std(errors, axis=0)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "model_version": "v0.1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "training_data_path": str(training_data_path.name),
        "metrics": {
            "test_loss": float(loss),
            "test_mae": float(mae),
            "epochs_ran": int(len(history.history.get("loss", []))),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "confidence_interval_90": {
            target_name: float(confidence[idx])
            for idx, target_name in enumerate(TARGET_COLS)
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the NBA predictor model and save serving artifacts."
    )
    parser.add_argument(
        "--training-data",
        type=Path,
        default=DEFAULT_TRAINING_DATA_PATH,
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used during training.",
    )
    args = parser.parse_args()

    metadata = train_and_save(
        training_data_path=args.training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

