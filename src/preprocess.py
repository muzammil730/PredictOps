import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# Column names for CMAPSS FD001
columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# Load training data
train_path = DATA_DIR / "train_FD001.txt"

df = pd.read_csv(
    train_path,
    sep=r"\s+",
    header=None,
    names=columns
)


print("Original dataset shape:", df.shape)


# Calculate maximum cycle for every engine
max_cycle = df.groupby("unit")["cycle"].max()


# Calculate Remaining Useful Life (RUL)
df["RUL"] = df.apply(
    lambda row: max_cycle[row["unit"]] - row["cycle"],
    axis=1
)


# Remove columns with no useful variation
constant_columns = [
    column for column in df.columns
    if df[column].nunique() <= 1
]

df = df.drop(columns=constant_columns)


# Save processed dataset
output_path = DATA_DIR / "processed_train.csv"

df.to_csv(output_path, index=False)


print("Processed dataset shape:", df.shape)
print("Removed constant columns:", constant_columns)
print("Saved processed dataset to:", output_path)

print("\nRUL statistics:")
print(df["RUL"].describe())