import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model = joblib.load("models/rul_model.pkl")
df = pd.read_csv("data/features_test.csv")

FEATURE_COLUMNS = [
    "unit", "cycle", "setting_1", "setting_2",
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13",
    "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
    "sensor_2_rolling_mean", "sensor_2_rolling_std",
    "sensor_3_rolling_mean", "sensor_3_rolling_std",
    "sensor_4_rolling_mean", "sensor_4_rolling_std",
    "sensor_6_rolling_mean", "sensor_6_rolling_std",
    "sensor_7_rolling_mean", "sensor_7_rolling_std",
    "sensor_8_rolling_mean", "sensor_8_rolling_std",
    "sensor_9_rolling_mean", "sensor_9_rolling_std",
    "sensor_11_rolling_mean", "sensor_11_rolling_std",
    "sensor_12_rolling_mean", "sensor_12_rolling_std",
    "sensor_13_rolling_mean", "sensor_13_rolling_std",
    "sensor_14_rolling_mean", "sensor_14_rolling_std",
    "sensor_15_rolling_mean", "sensor_15_rolling_std",
    "sensor_17_rolling_mean", "sensor_17_rolling_std",
    "sensor_20_rolling_mean", "sensor_20_rolling_std",
    "sensor_21_rolling_mean", "sensor_21_rolling_std",
]

X_test = df[FEATURE_COLUMNS]
y_test = df["RUL"]

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("===== Evaluation on TRULY UNSEEN test machines (test_FD001) =====")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")