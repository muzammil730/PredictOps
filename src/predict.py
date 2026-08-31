import pandas as pd
import joblib

from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features_train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "rul_model.pkl"


# Load trained model
print("Loading trained RUL model...")
model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# Load feature dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded.")
print("Dataset shape:", df.shape)


# Separate features and target
TARGET_COLUMN = "RUL"

X = df.drop(columns=[TARGET_COLUMN])


# Select one sample for prediction
sample = X.iloc[[0]]


# Make prediction
prediction = model.predict(sample)


print("\n===== RUL Prediction =====")
print("Predicted Remaining Useful Life:", round(prediction[0], 2), "cycles")