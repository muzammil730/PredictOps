from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "rul_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "features_train.csv"


# --------------------------------------------------
# Model and feature schema
# --------------------------------------------------

model = None

FEATURE_COLUMNS = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
    "sensor_2_rolling_mean",
    "sensor_2_rolling_std",
    "sensor_3_rolling_mean",
    "sensor_3_rolling_std",
    "sensor_4_rolling_mean",
    "sensor_4_rolling_std",
    "sensor_6_rolling_mean",
    "sensor_6_rolling_std",
    "sensor_7_rolling_mean",
    "sensor_7_rolling_std",
    "sensor_8_rolling_mean",
    "sensor_8_rolling_std",
    "sensor_9_rolling_mean",
    "sensor_9_rolling_std",
    "sensor_11_rolling_mean",
    "sensor_11_rolling_std",
    "sensor_12_rolling_mean",
    "sensor_12_rolling_std",
    "sensor_13_rolling_mean",
    "sensor_13_rolling_std",
    "sensor_14_rolling_mean",
    "sensor_14_rolling_std",
    "sensor_15_rolling_mean",
    "sensor_15_rolling_std",
    "sensor_17_rolling_mean",
    "sensor_17_rolling_std",
    "sensor_20_rolling_mean",
    "sensor_20_rolling_std",
    "sensor_21_rolling_mean",
    "sensor_21_rolling_std",
]


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="PredictOps API",
    description="Machine Learning API for Remaining Useful Life prediction",
    version="1.0.0"
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class PredictionRequest(BaseModel):
    features: Dict[str, float]


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PredictOps API"
    }


# --------------------------------------------------
# RUL prediction
# --------------------------------------------------

@app.post("/predict")
def predict_rul(request: PredictionRequest):

    global model

    # Check for missing features
    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in request.features
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required features",
                "missing_features": missing_features
            }
        )

    # Check for unexpected features
    extra_features = [
        feature
        for feature in request.features
        if feature not in FEATURE_COLUMNS
    ]

    if extra_features:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unexpected features received",
                "extra_features": extra_features
            }
        )

    # Load model only when prediction is requested
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model file is not available"
            )

        model = joblib.load(MODEL_PATH)

    # Create input DataFrame in exact training order
    input_data = pd.DataFrame(
        [[request.features[feature] for feature in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    # Make prediction
    prediction = model.predict(input_data)

    predicted_rul = round(float(prediction[0]), 2)

    return {
        "predicted_rul_cycles": predicted_rul,
        "unit": "cycles",
        "model": "RUL prediction model"
    }