from fastapi.testclient import TestClient

from nba_predictor.main import app


def test_health_endpoint_returns_status() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "artifacts_loaded" in payload


def test_predict_success() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "GP_r": 82,
            "MIN_r": 10,
            "FG_PCT_r": 0.45,
            "REB_r": 5,
            "AST_r": 3,
            "PTS_r": 8,
            "TOV_r": 3,
        },
    )

    assert response.status_code == 200

def test_predict_rejects_invalid_field_goal_percentage_scale() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "GP_r": 82,
            "MIN_r": 10,
            "FG_PCT_r": 40,
            "REB_r": 5,
            "AST_r": 3,
            "PTS_r": 8,
            "TOV_r": 3,
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"][0]["loc"] == ["body", "FG_PCT_r"]


def test_predict_rejects_values_outside_historical_bounds() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "GP_r": 90,
            "MIN_r": 50,
            "FG_PCT_r": 0.45,
            "REB_r": 13,
            "AST_r": 11,
            "PTS_r": 26,
            "TOV_r": 6,
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    invalid_fields = {item["loc"][-1] for item in detail}
    assert invalid_fields == {"GP_r", "MIN_r", "REB_r", "AST_r", "PTS_r", "TOV_r"}
