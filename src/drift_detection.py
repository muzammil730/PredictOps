import pandas as pd
from scipy.stats import ks_2samp


REFERENCE_DATA = "data/features_train.csv"

# Reference = training data
reference = pd.read_csv(REFERENCE_DATA)

# Simulate current production batch using a later portion
current = reference.sample(frac=0.2, random_state=42)


feature_columns = [col for col in reference.columns if col != "RUL"]

drifted_features = 0
results = []

for feature in feature_columns:
    statistic, p_value = ks_2samp(
        reference[feature],
        current[feature]
    )

    is_drifted = p_value < 0.05

    if is_drifted:
        drifted_features += 1

    results.append({
        "feature": feature,
        "ks_statistic": statistic,
        "p_value": p_value,
        "drift_detected": is_drifted
    })


drift_percentage = (drifted_features / len(feature_columns)) * 100

print("\n===== PredictOps Data Drift Report =====")
print(f"Total features: {len(feature_columns)}")
print(f"Drifted features: {drifted_features}")
print(f"Drift percentage: {drift_percentage:.2f}%")

if drift_percentage > 20:
    print("STATUS: DRIFT DETECTED")
else:
    print("STATUS: NO SIGNIFICANT DRIFT")

print("\nFeature details:")
for result in results:
    print(
        f"{result['feature']}: "
        f"p-value={result['p_value']:.4f}, "
        f"drift={result['drift_detected']}"
    )