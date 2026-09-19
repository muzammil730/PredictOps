import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

test_path = DATA_DIR / "test_FD001.txt"
rul_path = DATA_DIR / "RUL_FD001.txt"

df = pd.read_csv(test_path, sep=r"\s+", header=None, names=columns)
print("Original test dataset shape:", df.shape)

# Same columns that survived preprocessing on the TRAINING set.
# Hardcoded (not recomputed) so the schema matches training exactly.
KEEP_SENSORS = [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
keep_columns = (
    ["unit", "cycle", "setting_1", "setting_2"]
    + [f"sensor_{i}" for i in KEEP_SENSORS]
)
df = df[keep_columns]

# RUL_FD001.txt gives the true remaining life at each unit's LAST
# recorded test cycle (test set is deliberately truncated before
# failure). Reconstruct per-row RUL by counting forward from there.
true_rul = pd.read_csv(rul_path, header=None, names=["RUL_at_last_cycle"])
true_rul["unit"] = true_rul.index + 1

max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
df = df.merge(max_cycle, on="unit").merge(true_rul, on="unit")

df["RUL"] = df["RUL_at_last_cycle"] + (df["max_cycle"] - df["cycle"])

# Same cap as training, so the model is evaluated on the same
# target definition it was trained on.
df["RUL"] = df["RUL"].clip(upper=125)

df = df.drop(columns=["max_cycle", "RUL_at_last_cycle"])

output_path = DATA_DIR / "processed_test.csv"
df.to_csv(output_path, index=False)

print("Processed test dataset shape:", df.shape)
print("Saved to:", output_path)
print("\nRUL statistics (test set):")
print(df["RUL"].describe())