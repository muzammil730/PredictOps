import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

input_path = DATA_DIR / "processed_test.csv"
output_path = DATA_DIR / "features_test.csv"

df = pd.read_csv(input_path)
print("Input dataset shape:", df.shape)

sensor_columns = [c for c in df.columns if c.startswith("sensor_")]

for sensor in sensor_columns:
    df[f"{sensor}_rolling_mean"] = (
        df.groupby("unit")[sensor]
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )
    df[f"{sensor}_rolling_std"] = (
        df.groupby("unit")[sensor]
        .transform(lambda x: x.rolling(window=5, min_periods=1).std())
        .fillna(0)
    )

df.to_csv(output_path, index=False)
print("Feature dataset shape:", df.shape)
print("Saved to:", output_path)