"""
API test suite for FastAPI backend endpoints.
"""
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "NYC Ride Demand" in data["service"]


def test_model_endpoint():
    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()
    assert data["champion_version"] in ["model_v1", "model_v1_feb_review"]
    assert "LightGBM" in data["model_type"]
    assert len(data["features"]) == 17
    assert isinstance(data["validation_metrics"], dict)


def test_predict_endpoint_valid():
    payload = {
        "zone": 138,
        "timestamp": "2025-02-20 18:00:00",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["zone"] == 138
    assert data["timestamp"] == "2025-02-20 18:00:00"
    assert isinstance(data["predicted_demand"], (int, float))
    assert data["predicted_demand"] >= 0.0
    assert not np_isnan(data["predicted_demand"])
    assert data["model_version"] == "model_v1"


def test_predict_endpoint_invalid_zone():
    payload = {
        "zone": 999,  # Out of range 1-263
        "timestamp": "2025-02-20 18:00:00",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [400, 422]



def test_predict_endpoint_invalid_timestamp():
    payload = {
        "zone": 138,
        "timestamp": "invalid-timestamp-string",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_monitoring_endpoint():
    response = client.get("/monitoring")
    assert response.status_code == 200
    data = response.json()
    assert data["overall_mae"] > 0
    assert data["overall_rmse"] > 0
    assert data["degradation_threshold_mae"] > 0
    assert data["is_degraded"] is False
    assert data["degradation_status"] == "HEALTHY"
    assert data["total_predictions_logged"] > 0


def test_drift_endpoint():
    response = client.get("/drift")
    assert response.status_code == 200
    data = response.json()
    assert data["drift_status"] in ["STABLE", "WARNING"]
    assert len(data["features"]) >= 3
    for feat in data["features"]:
        assert feat["psi"] >= 0.0
        assert not np_isnan(feat["psi"])


def test_governance_endpoint():
    response = client.get("/governance")
    assert response.status_code == 200
    data = response.json()
    assert "model_v1" in data["champion_version"]
    assert data["decision"] == "REJECT CHALLENGER"
    assert data["observed_improvement_pct"] < 0
    assert data["required_improvement_pct"] == 3.0
    assert len(data["lineage"]) >= 2


def np_isnan(val):
    import numpy as np
    return np.isnan(val)


if __name__ == "__main__":
    print("Running API tests...")
    test_health_endpoint()
    print("Health test passed!")
    test_model_endpoint()
    print("Model test passed!")
    test_predict_endpoint_valid()
    print("Predict valid test passed!")
    test_predict_endpoint_invalid_zone()
    print("Predict invalid zone test passed!")
    test_predict_endpoint_invalid_timestamp()
    print("Predict invalid timestamp test passed!")
    test_monitoring_endpoint()
    print("Monitoring test passed!")
    test_drift_endpoint()
    print("Drift test passed!")
    test_governance_endpoint()
    print("Governance test passed!")
    print("\nAll API tests passed successfully!")
