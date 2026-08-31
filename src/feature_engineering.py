import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

input_path = DATA_DIR / "processed_train.csv"
output_path = DATA_DIR / "features_train.csv"


# Load processed dataset
df = pd.read_csv(input_path)

print("Input dataset shape:", df.shape)


# Sensor columns
sensor_columns = [
    column for column in df.columns
    if column.startswith("sensor_")
]


# Create rolling features for each engine
for sensor in sensor_columns:

    # Rolling mean = recent average sensor value
    df[f"{sensor}_rolling_mean"] = (
        df.groupby("unit")[sensor]
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )

    # Rolling std = recent sensor variation
    df[f"{sensor}_rolling_std"] = (
        df.groupby("unit")[sensor]
        .transform(lambda x: x.rolling(window=5, min_periods=1).std())
        .fillna(0)
    )


# Save feature dataset
df.to_csv(output_path, index=False)

print("Feature dataset shape:", df.shape)
print("Number of features created:", len(df.columns))
print("Saved features to:", output_path)

print("\nSample columns:")
print(df.columns.tolist())