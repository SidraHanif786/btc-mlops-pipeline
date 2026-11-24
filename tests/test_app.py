import pytest
from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "model_version" in data


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "MAE" in data
    assert "RMSE" in data
    assert "MAPE" in data


def test_predict_default_7_days(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["forecast_days"] == 7
    assert len(data["predictions"]) == 7
    assert "date" in data["predictions"][0]
    assert "predicted_close" in data["predictions"][0]


def test_predict_custom_history(client):
    payload = {
        "forecast_days": 5,
        "historical_closes": [61000, 61500, 62000, 61800, 62500, 63000, 63200],
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["predictions"]) == 5


def test_predict_invalid_days_returns_400(client):
    resp = client.post("/predict", json={"forecast_days": 0})
    assert resp.status_code == 400


def test_predict_invalid_history_returns_400(client):
    resp = client.post("/predict", json={"historical_closes": [100, -5]})
    assert resp.status_code == 400
