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
# Load model and feature schema
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

TARGET_COLUMN = "RUL"

FEATURE_COLUMNS = [
    column for column in df.columns
    if column != TARGET_COLUMN
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

    # Create input DataFrame in the exact training order
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