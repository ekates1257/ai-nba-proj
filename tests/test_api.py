from fastapi.testclient import TestClient

from nba_predictor.main import app


def test_health_endpoint_returns_status() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "artifacts_loaded" in payload
