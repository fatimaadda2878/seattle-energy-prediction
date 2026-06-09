from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal


class BuildingInput(BaseModel):
    """
    Données d'entrée pour la prédiction de consommation énergétique.

    Toutes les features sont issues du dataset Seattle 2016 Building Energy Benchmarking.
    Les valeurs doivent correspondre à des bâtiments commerciaux de Seattle.
    """

    # ── Features de surface ──────────────────────────────────────────────────
    PropertyGFATotal: float = Field(
        ...,
        gt=0,
        description="Surface totale du bâtiment en pieds carrés (Gross Floor Area). Inclut toutes les surfaces couvertes.",
        examples=[88434],
    )
    PropertyGFABuilding: float = Field(
        ...,
        gt=0,
        description="Surface construite du bâtiment en pieds carrés. Doit être inférieure ou égale à PropertyGFATotal.",
        examples=[88434],
    )

    # ── Features temporelles ─────────────────────────────────────────────────
    YearBuilt: int = Field(
        ...,
        ge=1900,
        le=2016,
        description="Année de construction du bâtiment. Doit être comprise entre 1900 et 2016.",
        examples=[1927],
    )

    # ── Features structurelles ───────────────────────────────────────────────
    NumberofFloors: int = Field(
        ...,
        ge=0,
        description="Nombre d'étages du bâtiment.",
        examples=[12],
    )
    NumberofBuildings: float = Field(
        ...,
        ge=0,
        description="Nombre de bâtiments sur la parcelle.",
        examples=[1],
    )

    # ── Sources d'énergie (0 = non utilisée, 1 = utilisée) ──────────────────
    has_electricity: int = Field(
        ...,
        ge=0,
        le=1,
        description="Le bâtiment utilise-t-il l'électricité ? (0 = non, 1 = oui)",
        examples=[1],
    )
    has_gas: int = Field(
        ...,
        ge=0,
        le=1,
        description="Le bâtiment utilise-t-il le gaz naturel ? (0 = non, 1 = oui)",
        examples=[1],
    )
    has_steam: int = Field(
        ...,
        ge=0,
        le=1,
        description="Le bâtiment utilise-t-il la vapeur ? (0 = non, 1 = oui)",
        examples=[1],
    )

    # ── Localisation ─────────────────────────────────────────────────────────
    Neighborhood: str = Field(
        ...,
        description="Quartier de Seattle où se situe le bâtiment (ex: DOWNTOWN, CAPITOL HILL, BALLARD).",
        examples=["DOWNTOWN"],
    )
    BuildingType: str = Field(
        ...,
        description="Type de bâtiment (NonResidential, Multifamily HR, Multifamily MR, Multifamily LR, Campus).",
        examples=["NonResidential"],
    )
    ZipCode: float = Field(
        ...,
        description="Code postal du bâtiment à Seattle.",
        examples=[98101],
    )

    # ── Features catégorielles ───────────────────────────────────────────────
    PrimaryPropertyType: str = Field(
        ...,
        description="Type d'usage principal du bâtiment (ex: Hotel, Office, Retail Store, Hospital).",
        examples=["Hotel"],
    )
    tranche_age: Literal[
        "avant 1940", "1940-1959", "1960-1979", "1980-1999", "2000+"
    ] = Field(
        ...,
        description="Tranche d'âge du bâtiment calculée depuis l'année de construction.",
        examples=["avant 1940"],
    )
    tranche_surface: Literal[
        "petit", "moyen", "grand", "tres grand"
    ] = Field(
        ...,
        description="Tranche de surface du bâtiment. petit < 10 000 ft², moyen < 50 000, grand < 200 000, tres grand ≥ 200 000.",
        examples=["moyen"],
    )

    # ── Validations croisées ─────────────────────────────────────────────────
    @field_validator("PropertyGFABuilding")
    @classmethod
    def building_not_greater_than_total(cls, v, info):
        if "PropertyGFATotal" in info.data and v > info.data["PropertyGFATotal"]:
            raise ValueError(
                "PropertyGFABuilding ne peut pas dépasser PropertyGFATotal."
            )
        return v

    @field_validator("has_electricity", "has_gas", "has_steam")
    @classmethod
    def energy_must_be_binary(cls, v):
        if v not in (0, 1):
            raise ValueError("Les champs has_* doivent être 0 ou 1.")
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
                "tranche_surface": "moyen",
            }
        }
    }


class PredictionOutput(BaseModel):
    """Résultat retourné par le modèle après prédiction."""

    prediction_kbtu: float = Field(
        ...,
        description="Consommation énergétique prédite en kBtu (valeur réelle, après transformation inverse du log).",
        examples=[735095.42],
    )
    prediction_log: float = Field(
        ...,
        description="Consommation prédite en log1p(kBtu), valeur brute retournée par le modèle.",
        examples=[13.508],
    )
    input_data: dict = Field(
        ...,
        description="Données d'entrée utilisées pour la prédiction, retournées pour traçabilité.",
    )


class HealthResponse(BaseModel):
    """Réponse du endpoint de vérification de l'état de l'API."""

    status: str = Field(
        ...,
        description="État de l'API. Vaut 'ok' si l'API fonctionne correctement.",
        examples=["ok"],
    )
    model_loaded: bool = Field(
        ...,
        description="Indique si le modèle ML est correctement chargé en mémoire.",
        examples=[True],
    )
