import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_DATA = "data/features_train.csv"


def calculate_drift_percentage():
    reference = pd.read_csv(REFERENCE_DATA)

    # Temporary production-like sample
    current = reference.sample(frac=0.2, random_state=42)

    feature_columns = [col for col in reference.columns if col != "RUL"]

    drifted_features = 0

    for feature in feature_columns:
        _, p_value = ks_2samp(
            reference[feature],
            current[feature]
        )

        if p_value < 0.05:
            drifted_features += 1

    return (drifted_features / len(feature_columns)) * 100


if __name__ == "__main__":
    drift_percentage = calculate_drift_percentage()

    print(f"Total features: 49")
    print(f"Drift percentage: {drift_percentage:.2f}%")

    if drift_percentage > 20:
        print("STATUS: DRIFT DETECTED")
    else:
        print("STATUS: NO SIGNIFICANT DRIFT")