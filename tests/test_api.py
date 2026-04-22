from fastapi.testclient import TestClient

from nba_predictor.inference import PlayerNotFoundError
from nba_predictor.main import app, service


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


def test_predict_by_player_name_returns_prediction(monkeypatch) -> None:
    def fake_predict_by_player_name(player_name: str) -> dict:
        assert player_name == "Jayson Tatum"
        return {
            "player_name": "Jayson Tatum",
            "rookie_stats": {
                "GP_r": 80.0,
                "MIN_r": 30.5,
                "FG_PCT_r": 0.475,
                "REB_r": 5.0,
                "AST_r": 1.6,
                "PTS_r": 13.9,
                "TOV_r": 1.4,
            },
            "predictions": {"PTS_s": 16.8, "REB_s": 5.4, "AST_s": 2.0},
            "confidence_interval_90": {
                "PTS_s": 5.8,
                "REB_s": 2.4,
                "AST_s": 1.6,
            },
            "model_version": "v0.1.0",
        }

    monkeypatch.setattr(service, "predict_by_player_name", fake_predict_by_player_name)

    client = TestClient(app)
    response = client.get("/predict/player/Jayson Tatum")

    assert response.status_code == 200
    payload = response.json()
    assert payload["player_name"] == "Jayson Tatum"
    assert payload["predictions"]["PTS_s"] == 16.8


def test_predict_by_player_name_returns_404_for_unknown_player(monkeypatch) -> None:
    def fake_predict_by_player_name(_: str) -> dict:
        raise PlayerNotFoundError("No rookie player data found.")

    monkeypatch.setattr(service, "predict_by_player_name", fake_predict_by_player_name)

    client = TestClient(app)
    response = client.get("/predict/player/Not A Real Rookie")

    assert response.status_code == 404
