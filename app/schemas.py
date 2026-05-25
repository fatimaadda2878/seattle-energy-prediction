from pydantic import BaseModel, Field, field_validator
from typing import Literal
import numpy as np


class BuildingInput(BaseModel):
    """Données d'entrée pour la prédiction de consommation énergétique."""

    # Features numériques
    PropertyGFATotal: float = Field(..., gt=0, description="Surface totale du bâtiment (pieds carrés)")
    PropertyGFABuilding: float = Field(..., gt=0, description="Surface construite (pieds carrés)")
    YearBuilt: int = Field(..., ge=1900, le=2016, description="Année de construction")
    NumberofFloors: int = Field(..., ge=0, description="Nombre d'étages")
    NumberofBuildings: float = Field(..., ge=0, description="Nombre de bâtiments sur la parcelle")
    has_electricity: int = Field(..., ge=0, le=1, description="Utilise l'électricité (0 ou 1)")
    has_gas: int = Field(..., ge=0, le=1, description="Utilise le gaz (0 ou 1)")
    has_steam: int = Field(..., ge=0, le=1, description="Utilise la vapeur (0 ou 1)")
    Neighborhood: str = Field(..., description="Quartier du bâtiment")
    BuildingType: str = Field(..., description="Type de bâtiment")
    ZipCode: float = Field(..., description="Code postal")

    # Features catégorielles
    PrimaryPropertyType: str = Field(..., description="Type d'usage principal")
    tranche_age: Literal[
        "avant 1940", "1940-1959", "1960-1979", "1980-1999", "2000+"
    ] = Field(..., description="Tranche d'âge du bâtiment")
    tranche_surface: Literal[
        "petit", "moyen", "grand", "tres grand"
    ] = Field(..., description="Tranche de surface")

    @field_validator("PropertyGFABuilding")
    @classmethod
    def building_smaller_than_total(cls, v, info):
        if "PropertyGFATotal" in info.data and v > info.data["PropertyGFATotal"]:
            raise ValueError("La surface construite ne peut pas dépasser la surface totale")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionOutput(BaseModel):
    """Résultat de la prédiction."""
    prediction_kbtu: float = Field(..., description="Consommation prédite en kBtu")
    prediction_log: float = Field(..., description="Consommation prédite en log(kBtu)")
    input_data: dict = Field(..., description="Données d'entrée utilisées")


class HealthResponse(BaseModel):
    """Réponse du endpoint de santé."""
    status: str
    model_loaded: bool
