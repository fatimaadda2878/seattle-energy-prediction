"""Tests fonctionnels des endpoints de l'API FastAPI."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)

VALID_INPUT = {
    "PropertyGFATotal": 88434,
    "PropertyGFABuilding": 88434,
    "YearBuilt": 1927,
    "NumberofFloors": 12,
    "NumberofBuildings": 1,
    "has_electricity": 1,
    "has_gas": 1,
    "has_steam": 1,
    "Neighborhood": "DOWNTOWN",
    "BuildingType": "NonResidential",
    "ZipCode": 98101,
    "PrimaryPropertyType": "Hotel",
    "tranche_age": "avant 1940",
    "tranche_surface": "moyen"
}


class TestHealthEndpoint:
    """Tests du endpoint /health."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_has_model_loaded_field(self):
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data


class TestPredictEndpoint:
    """Tests du endpoint /predict."""

    @patch("app.main.predict")
    def test_predict_returns_200(self, mock_predict):
        mock_predict.return_value = {
            "prediction_log": 13.5,
            "prediction_kbtu": 735095.0,
            "input_data": VALID_INPUT
        }
        response = client.post("/predict", json=VALID_INPUT)
        assert response.status_code == 200

    @patch("app.main.predict")
    def test_predict_returns_correct_fields(self, mock_predict):
        mock_predict.return_value = {
            "prediction_log": 13.5,
            "prediction_kbtu": 735095.0,
            "input_data": VALID_INPUT
        }
        response = client.post("/predict", json=VALID_INPUT)
        data = response.json()
        assert "prediction_kbtu" in data
        assert "prediction_log" in data
        assert "input_data" in data

    def test_predict_invalid_year_built(self):
        invalid_input = VALID_INPUT.copy()
        invalid_input["YearBuilt"] = 1800
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_negative_surface(self):
        invalid_input = VALID_INPUT.copy()
        invalid_input["PropertyGFATotal"] = -100
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_missing_field(self):
        invalid_input = VALID_INPUT.copy()
        del invalid_input["PrimaryPropertyType"]
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_invalid_tranche_age(self):
        invalid_input = VALID_INPUT.copy()
        invalid_input["tranche_age"] = "invalide"
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422
