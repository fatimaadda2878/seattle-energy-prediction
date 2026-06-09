from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import BuildingInput, PredictionOutput, HealthResponse
from app.predict import predict, is_model_loaded

app = FastAPI(
    title="API Prédiction Consommation Énergétique — Seattle",
    description="""
## Description

API de prédiction de la consommation énergétique des bâtiments commerciaux de Seattle.

Le modèle est entraîné sur le dataset **2016 Seattle Building Energy Benchmarking**
et utilise un **GradientBoostingRegressor** optimisé par GridSearchCV (R² = 0.715).

## Endpoints disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET`   | `/health` | Vérifie l'état de l'API et du modèle |
| `POST`  | `/predict` | Prédit la consommation en kBtu |

## Fonctionnement

1. Envoyer les features du bâtiment en JSON via `POST /predict`
2. Le modèle calcule les features dérivées (log surface, âge, ratio...)
3. La prédiction est retournée en **kBtu** (valeur réelle)
4. Chaque prédiction est tracée en base de données PostgreSQL

## Validation des données

Toutes les données d'entrée sont validées par **Pydantic** :
- Les surfaces doivent être positives
- L'année de construction doit être entre 1900 et 2016
- Les champs binaires (has_electricity, etc.) doivent valoir 0 ou 1
- La surface construite ne peut pas dépasser la surface totale

En cas de données invalides, l'API retourne une erreur **422 Unprocessable Entity**.
    """,
    version="1.0.0",
    contact={
        "name": "Fatima Adda-Rezig",
        "url": "https://github.com/fatimaadda2878/seattle-energy-prediction",
    },
    license_info={
        "name": "MIT",
    },
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Santé"],
    summary="Vérification de l'état de l'API",
    response_description="État de l'API et du modèle ML",
)
def health_check():
    """
    Vérifie que l'API est opérationnelle et que le modèle ML est correctement chargé.

    Retourne :
    - **status** : `ok` si l'API fonctionne
    - **model_loaded** : `true` si le fichier `model.pkl` est chargé en mémoire

    Ce endpoint est utilisé par le pipeline CI/CD pour valider le déploiement.
    """
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prédiction"],
    summary="Prédiction de la consommation énergétique",
    response_description="Consommation prédite en kBtu et en log(kBtu)",
    responses={
        200: {
            "description": "Prédiction réussie",
            "content": {
                "application/json": {
                    "example": {
                        "prediction_kbtu": 735095.42,
                        "prediction_log": 13.508,
                        "input_data": {
                            "PropertyGFATotal": 88434,
                            "YearBuilt": 1927,
                            "PrimaryPropertyType": "Hotel",
                        },
                    }
                }
            },
        },
        422: {"description": "Données d'entrée invalides (erreur de validation Pydantic)"},
        503: {"description": "Modèle ML non disponible (fichier model.pkl manquant)"},
        500: {"description": "Erreur interne lors de la prédiction"},
    },
)
def predict_energy(building: BuildingInput):
    """
    Prédit la consommation énergétique annuelle d'un bâtiment de Seattle en kBtu.

    ### Traitement des données

    Les features brutes sont transformées avant d'être envoyées au modèle :
    - `log_surface_totale` = log1p(PropertyGFATotal)
    - `building_age` = 2016 - YearBuilt
    - `ratio_surface` = PropertyGFABuilding / PropertyGFATotal
    - `nb_sources_energie` = has_electricity + has_gas + has_steam

    ### Modèle utilisé

    **GradientBoostingRegressor** (scikit-learn) avec Pipeline sklearn :
    - `StandardScaler` pour les features numériques
    - `OneHotEncoder` pour les features catégorielles
    - Optimisé par GridSearchCV (3-fold CV, scoring=R²)

    ### Interprétation du résultat

    La valeur `prediction_kbtu` est la consommation annuelle prédite en **kBtu**.
    Pour convertir : 1 kBtu ≈ 0.293 kWh.
    """
    try:
        result = predict(building.model_dump())
        return PredictionOutput(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Modèle non disponible : {str(e)}. Vérifiez que models/model.pkl existe.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}",
        )
