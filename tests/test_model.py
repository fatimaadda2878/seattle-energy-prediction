"""Tests unitaires du modèle de machine learning et du preprocessing."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from app.predict import prepare_features, predict, is_model_loaded


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


class TestPrepareFeatures:
    """Tests de la fonction de préparation des features."""

    def test_returns_dataframe(self):
        result = prepare_features(VALID_INPUT)
        assert isinstance(result, pd.DataFrame)

    def test_returns_one_row(self):
        result = prepare_features(VALID_INPUT)
        assert len(result) == 1

    def test_log_surface_totale_correct(self):
        result = prepare_features(VALID_INPUT)
        expected = np.log1p(88434)
        assert abs(result["log_surface_totale"].values[0] - expected) < 1e-6

    def test_building_age_correct(self):
        result = prepare_features(VALID_INPUT)
        assert result["building_age"].values[0] == 2016 - 1927

    def test_nb_sources_energie_correct(self):
        result = prepare_features(VALID_INPUT)
        assert result["nb_sources_energie"].values[0] == 3

    def test_ratio_surface_correct(self):
        result = prepare_features(VALID_INPUT)
        assert abs(result["ratio_surface"].values[0] - 1.0) < 1e-6

    def test_contains_all_required_columns(self):
        result = prepare_features(VALID_INPUT)
        required_cols = [
            "log_surface_totale", "log_surface_batiment", "ratio_surface",
            "building_age", "NumberofFloors", "NumberofBuildings",
            "has_electricity", "has_gas", "has_steam", "nb_sources_energie",
            "Neighborhood_enc", "BuildingType_enc", "ZipCode_enc",
            "PrimaryPropertyType", "tranche_age", "tranche_surface"
        ]
        for col in required_cols:
            assert col in result.columns, f"Colonne manquante : {col}"


class TestPredict:
    """Tests de la fonction de prédiction."""

    @patch("app.predict.load_model")
    def test_predict_returns_dict(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([13.5])
        mock_load.return_value = mock_model

        result = predict(VALID_INPUT)
        assert isinstance(result, dict)

    @patch("app.predict.load_model")
    def test_predict_has_correct_keys(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([13.5])
        mock_load.return_value = mock_model

        result = predict(VALID_INPUT)
        assert "prediction_log" in result
        assert "prediction_kbtu" in result
        assert "input_data" in result

    @patch("app.predict.load_model")
    def test_predict_kbtu_is_expm1_of_log(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([13.5])
        mock_load.return_value = mock_model

        result = predict(VALID_INPUT)
        expected_kbtu = round(float(np.expm1(13.5)), 2)
        assert abs(result["prediction_kbtu"] - expected_kbtu) < 1.0

    @patch("app.predict.load_model")
    def test_predict_kbtu_positive(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([13.5])
        mock_load.return_value = mock_model

        result = predict(VALID_INPUT)
        assert result["prediction_kbtu"] > 0


class TestIsModelLoaded:
    """Tests de la fonction de vérification du modèle."""

    @patch("app.predict.load_model")
    def test_returns_true_when_model_exists(self, mock_load):
        mock_load.return_value = MagicMock()
        assert is_model_loaded() is True

    @patch("app.predict.load_model")
    def test_returns_false_when_model_missing(self, mock_load):
        mock_load.side_effect = FileNotFoundError("Modèle introuvable")
        assert is_model_loaded() is False
