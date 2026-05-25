import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"

_model = None


def load_model():
    """Charge le modèle depuis le fichier pkl."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def is_model_loaded() -> bool:
    """Vérifie si le modèle est chargé."""
    try:
        load_model()
        return True
    except Exception:
        return False


def prepare_features(data: dict) -> pd.DataFrame:
    """Transforme les données brutes en features pour le modèle."""

    # Features dérivées
    log_surface_totale = np.log1p(data["PropertyGFATotal"])
    log_surface_batiment = np.log1p(data["PropertyGFABuilding"])
    ratio_surface = data["PropertyGFABuilding"] / data["PropertyGFATotal"]
    building_age = 2016 - data["YearBuilt"]
    nb_sources_energie = data["has_electricity"] + data["has_gas"] + data["has_steam"]

    # Label encoding (valeurs fixes apprises sur le dataset)
    neighborhood_map = {
        'BALLARD': 0, 'BEACON HILL': 1, 'BELTOWN': 2, 'CAPITOL HILL': 3,
        'CENTRAL': 4, 'DOWNTOWN': 6, 'FIRST HILL': 7, 'FREMONT': 8,
        'INDUSTRIAL': 9, 'MADISON': 10, 'NORTHGATE': 11, 'QUEEN ANNE': 12,
        'RAINIER VALLEY': 13, 'SEWARD PARK': 14, 'SOUTH LAKE UNION': 15,
        'UNIVERSITY': 16, 'WALLINGFORD': 17, 'WEST SEATTLE': 18
    }
    building_type_map = {
        'Campus': 0, 'Multifamily LR': 1, 'Multifamily MR': 2,
        'Multifamily HR': 3, 'NonResidential': 4
    }

    neighborhood_enc = neighborhood_map.get(data["Neighborhood"].upper(), 5)
    building_type_enc = building_type_map.get(data["BuildingType"], 4)

    # ZipCode encoding (index dans la liste triée des zipcodes du dataset)
    zipcodes = sorted([
        98006, 98007, 98008, 98101, 98102, 98103, 98104, 98105,
        98106, 98107, 98108, 98109, 98112, 98115, 98116, 98117,
        98118, 98119, 98121, 98122, 98125, 98126, 98133, 98134,
        98136, 98144, 98146, 98154, 98164, 98174, 98177, 98178,
        98195, 98199
    ])
    zipcode_enc = zipcodes.index(int(data["ZipCode"])) if int(data["ZipCode"]) in zipcodes else 0

    features = {
        # Numériques
        "log_surface_totale": log_surface_totale,
        "log_surface_batiment": log_surface_batiment,
        "ratio_surface": ratio_surface,
        "building_age": building_age,
        "NumberofFloors": data["NumberofFloors"],
        "NumberofBuildings": data["NumberofBuildings"],
        "has_electricity": data["has_electricity"],
        "has_gas": data["has_gas"],
        "has_steam": data["has_steam"],
        "nb_sources_energie": nb_sources_energie,
        "Neighborhood_enc": neighborhood_enc,
        "BuildingType_enc": building_type_enc,
        "ZipCode_enc": zipcode_enc,
        # Catégorielles (OHE dans la pipeline)
        "PrimaryPropertyType": data["PrimaryPropertyType"],
        "tranche_age": data["tranche_age"],
        "tranche_surface": data["tranche_surface"],
    }

    return pd.DataFrame([features])


def predict(data: dict) -> dict:
    """Lance la prédiction et retourne le résultat."""
    model = load_model()
    df = prepare_features(data)
    log_pred = float(model.predict(df)[0])
    kbtu_pred = float(np.expm1(log_pred))

    return {
        "prediction_log": round(log_pred, 4),
        "prediction_kbtu": round(kbtu_pred, 2),
        "input_data": data,
    }
