from pathlib import Path
from typing import Dict
import time

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest
)

from src.drift_detection import calculate_drift_percentage


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "rul_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "features_train.csv"
LIVE_DATA_PATH = PROJECT_ROOT / "data" / "features_test.csv"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="PredictOps API",
    description="Predictive Maintenance API using Random Forest RUL Model",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL
# ============================================================

model = None


def load_model():

    global model

    if model is None:

        if not MODEL_PATH.exists():

            raise HTTPException(
                status_code=503,
                detail="RUL model file is not available."
            )

        model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# FEATURE SCHEMA
# ============================================================

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


# ============================================================
# DATASET
# ============================================================

def load_dataset(path: Path = LIVE_DATA_PATH):

    if not path.exists():

        raise HTTPException(
            status_code=503,
            detail="Feature dataset is not available."
        )

    return pd.read_csv(path)


# ============================================================
# PROMETHEUS
# ============================================================

REQUEST_COUNT = Counter(
    "predictops_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "predictops_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)

DRIFT_METRIC = Gauge(
    "predictops_data_drift_percentage",
    "Percentage of features showing data drift"
)


# ============================================================
# DRIFT
# ============================================================

try:

    if DATA_PATH.exists():

        DRIFT_METRIC.set(
            calculate_drift_percentage()
        )

    else:

        DRIFT_METRIC.set(0.0)

except Exception:

    DRIFT_METRIC.set(0.0)


# ============================================================
# MONITORING MIDDLEWARE
# ============================================================

@app.middleware("http")
async def monitoring_middleware(
    request: Request,
    call_next
):

    start_time = time.time()

    try:

        response = await call_next(request)

        status_code = response.status_code

        return response

    except Exception:

        status_code = 500

        raise

    finally:

        duration = time.time() - start_time
        endpoint = request.url.path

        if endpoint != "/metrics":

            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status=status_code
            ).inc()

            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "PredictOps API",
        "model": "Random Forest RUL"
    }


# ============================================================
# MACHINES
# ============================================================

@app.get("/machines")
def get_machines():

    df = load_dataset()

    latest = (
        df.sort_values("cycle")
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
    )

    machines = []

    for _, row in latest.iterrows():

        rul = float(row["RUL"])

        if rul <= 30:

            status = "Critical"

        elif rul <= 100:

            status = "Maintenance Required"

        else:

            status = "Healthy"

        machines.append({

            "machine_id": int(row["unit"]),

            "cycle": int(row["cycle"]),

            "temperature": round(
                float(row["sensor_2"]), 2
            ),

            "vibration": round(
                float(row["sensor_21"]), 2
            ),

            "rul": round(rul, 2),

            "status": status

        })

    healthy = sum(
        1 for machine in machines
        if machine["status"] == "Healthy"
    )

    maintenance = sum(
        1 for machine in machines
        if machine["status"] != "Healthy"
    )

    return {

        "count": len(machines),

        "healthy": healthy,

        "maintenance_required": maintenance,

        "machines": machines

    }


# ============================================================
# MACHINE RUL PREDICTION
# ============================================================

@app.post("/predict-machine/{unit_id}")
def predict_machine(unit_id: int):

    df = load_dataset()

    machine_data = df[
        df["unit"] == unit_id
    ]

    if machine_data.empty:

        raise HTTPException(
            status_code=404,
            detail=f"Machine {unit_id} not found."
        )

    # Latest available cycle
    latest = (
        machine_data
        .sort_values("cycle")
        .tail(1)
    )

    # Check required features
    missing_features = [

        feature
        for feature in FEATURE_COLUMNS
        if feature not in latest.columns

    ]

    if missing_features:

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Required model features missing.",
                "missing_features": missing_features
            }
        )

    # Exact feature order used during training
    input_data = latest[
        FEATURE_COLUMNS
    ]

    # Load model
    prediction_model = load_model()

    # Prediction
    prediction = prediction_model.predict(
        input_data
    )

    predicted_rul = round(
        float(prediction[0]),
        2
    )

    # Health status
    if predicted_rul <= 30:

        status = "Critical"

    elif predicted_rul <= 100:

        status = "Maintenance Required"

    else:

        status = "Healthy"

    return {

        "machine_id": unit_id,

        "cycle": int(
            latest["cycle"].iloc[0]
        ),

        "predicted_rul_cycles": predicted_rul,

        "status": status,

        "model": "Random Forest RUL Model",

        "features_used": len(FEATURE_COLUMNS)

    }


# ============================================================
# GENERIC PREDICTION ENDPOINT
# ============================================================

class PredictionRequest(BaseModel):

    features: Dict[str, float]


@app.post("/predict")
def predict_rul(
    request: PredictionRequest
):

    prediction_model = load_model()

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

    input_data = pd.DataFrame(
        [
            [
                request.features[feature]
                for feature in FEATURE_COLUMNS
            ]
        ],
        columns=FEATURE_COLUMNS
    )

    prediction = prediction_model.predict(
        input_data
    )

    predicted_rul = round(
        float(prediction[0]),
        2
    )

    return {

        "predicted_rul_cycles": predicted_rul,

        "unit": "cycles",

        "model": "Random Forest RUL Model"

    }


# ============================================================
# PROMETHEUS METRICS
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


# ============================================================
# SAGEMAKER
# ============================================================

@app.get("/ping")
def ping():

    return Response(
        status_code=200
    )


@app.post("/invocations")
def invocations(
    request: PredictionRequest
):

    return predict_rul(request)