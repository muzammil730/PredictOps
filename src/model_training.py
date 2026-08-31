import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features_train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "rul_model.pkl"


# Load engineered dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# Target variable
TARGET = "RUL"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset.")


# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# Train model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

print("Training completed.")


# Make predictions
y_pred = model.predict(X_test)


# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("\n===== Model Evaluation =====")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# Create models directory if it doesn't exist
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# Save trained model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully at: {MODEL_PATH}")