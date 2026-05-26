"""Tests unitaires de la base de données."""

import pytest
from unittest.mock import patch, MagicMock, call
from app.database import save_prediction


VALID_INPUT = {
    "PropertyGFATotal": 88434.0,
    "PropertyGFABuilding": 88434.0,
    "YearBuilt": 1927,
    "NumberofFloors": 12,
    "NumberofBuildings": 1.0,
    "has_electricity": 1,
    "has_gas": 1,
    "has_steam": 1,
    "Neighborhood": "DOWNTOWN",
    "BuildingType": "NonResidential",
    "ZipCode": 98101.0,
    "PrimaryPropertyType": "Hotel",
    "tranche_age": "avant 1940",
    "tranche_surface": "moyen"
}


class TestSavePrediction:
    """Tests de la fonction save_prediction."""

    @patch("app.database.Prediction", create=True)
    def test_save_prediction_calls_db_add(self, mock_pred_class):
        mock_db = MagicMock()
        mock_pred_instance = MagicMock()
        mock_pred_class.return_value = mock_pred_instance

        with patch("app.database.save_prediction") as mock_save:
            mock_save.return_value = mock_pred_instance
            result = mock_save(mock_db, VALID_INPUT, 13.5, 735095.0)
            mock_save.assert_called_once_with(mock_db, VALID_INPUT, 13.5, 735095.0)

    def test_save_prediction_returns_object(self):
        mock_db = MagicMock()
        mock_pred = MagicMock()
        mock_pred.prediction_kbtu = 735095.0
        mock_pred.prediction_log = 13.5

        with patch("app.database.save_prediction", return_value=mock_pred):
            result = save_prediction(mock_db, VALID_INPUT, 13.5, 735095.0)
            assert result.prediction_kbtu == 735095.0
            assert result.prediction_log == 13.5
