from fastapi.testclient import TestClient

from app import main


client = TestClient(main.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


class FakeModel:
    def predict(self, data):
        return [100.0]


def test_predict(monkeypatch):

    # Use a fake model so CI does not need the real 248 MB model file
    monkeypatch.setattr(main, "model", FakeModel())

    response = client.post(
        "/predict",
        json={
            "features": {
                "unit": 1,
                "cycle": 1,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "sensor_2": 0.0,
                "sensor_3": 0.0,
                "sensor_4": 0.0,
                "sensor_6": 0.0,
                "sensor_7": 0.0,
                "sensor_8": 0.0,
                "sensor_9": 0.0,
                "sensor_11": 0.0,
                "sensor_12": 0.0,
                "sensor_13": 0.0,
                "sensor_14": 0.0,
                "sensor_15": 0.0,
                "sensor_17": 0.0,
                "sensor_20": 0.0,
                "sensor_21": 0.0,
                "sensor_2_rolling_mean": 0.0,
                "sensor_2_rolling_std": 0.0,
                "sensor_3_rolling_mean": 0.0,
                "sensor_3_rolling_std": 0.0,
                "sensor_4_rolling_mean": 0.0,
                "sensor_4_rolling_std": 0.0,
                "sensor_6_rolling_mean": 0.0,
                "sensor_6_rolling_std": 0.0,
                "sensor_7_rolling_mean": 0.0,
                "sensor_7_rolling_std": 0.0,
                "sensor_8_rolling_mean": 0.0,
                "sensor_8_rolling_std": 0.0,
                "sensor_9_rolling_mean": 0.0,
                "sensor_9_rolling_std": 0.0,
                "sensor_11_rolling_mean": 0.0,
                "sensor_11_rolling_std": 0.0,
                "sensor_12_rolling_mean": 0.0,
                "sensor_12_rolling_std": 0.0,
                "sensor_13_rolling_mean": 0.0,
                "sensor_13_rolling_std": 0.0,
                "sensor_14_rolling_mean": 0.0,
                "sensor_14_rolling_std": 0.0,
                "sensor_15_rolling_mean": 0.0,
                "sensor_15_rolling_std": 0.0,
                "sensor_17_rolling_mean": 0.0,
                "sensor_17_rolling_std": 0.0,
                "sensor_20_rolling_mean": 0.0,
                "sensor_20_rolling_std": 0.0,
                "sensor_21_rolling_mean": 0.0,
                "sensor_21_rolling_std": 0.0
            }
        }
    )

    assert response.status_code == 200
    assert "predicted_rul_cycles" in response.json()
    assert response.json()["predicted_rul_cycles"] == 100.0