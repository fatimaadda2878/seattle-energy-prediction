from fastapi import FastAPI, HTTPException
from app.schemas import BuildingInput, PredictionOutput, HealthResponse
from app.predict import predict, is_model_loaded

app = FastAPI(
    title="API Prédiction Consommation Énergétique",
    description="""
API de prédiction de la consommation énergétique des bâtiments de Seattle.

## Endpoints

- **GET /health** — Vérifie que l'API et le modèle sont opérationnels
- **POST /predict** — Prédit la consommation énergétique d'un bâtiment en kBtu
    """,
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, tags=["Santé"])
def health_check():
    """Vérifie l'état de l'API et du modèle."""
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prédiction"])
def predict_energy(building: BuildingInput):
    """
    Prédit la consommation énergétique d'un bâtiment.

    Retourne la consommation en kBtu (valeur réelle) et en log(kBtu).
    """
    try:
        result = predict(building.model_dump())
        return PredictionOutput(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Modèle non disponible : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")
