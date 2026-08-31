import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "train_FD001.txt"

# NASA C-MAPSS column names
columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load dataset
df = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)

print("\n=== PredictOps Dataset Inspection ===")

print(f"\nShape: {df.shape}")

print("\nFirst 5 rows:")
print(df.head())

print(f"\nNumber of engines: {df['unit'].nunique()}")

print(f"\nMaximum cycle: {df['cycle'].max()}")

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDataset information:")
print(df.info())