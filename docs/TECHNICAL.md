# Documentation Technique

## Architecture du projet

```
seattle-energy-prediction/
├── app/                    # API FastAPI
│   ├── __init__.py
│   ├── main.py             # Endpoints
│   ├── schemas.py          # Modèles Pydantic
│   ├── predict.py          # Inférence du modèle
│   └── database.py         # Connexion PostgreSQL
├── data/
│   ├── raw/                # Dataset brut
│   └── processed/          # Données transformées
├── models/                 # Modèle entraîné (model.pkl)
├── scripts/
│   ├── create_db.py        # Création des tables
│   └── insert_data.py      # Insertion du dataset
├── tests/
│   ├── test_api.py         # Tests endpoints
│   ├── test_model.py       # Tests modèle ML
│   └── test_db.py          # Tests base de données
├── notebooks/              # Notebook d'exploration
├── .github/workflows/      # CI/CD GitHub Actions
├── Dockerfile              # Conteneur pour HF Spaces
├── pytest.ini              # Configuration tests
├── requirements.txt
└── README.md
```

---

## Modèle de Machine Learning

### Algorithme

**GradientBoostingRegressor** (scikit-learn) encapsulé dans une Pipeline sklearn :

```
Pipeline
├── prep (ColumnTransformer)
│   ├── StandardScaler      → features numériques
│   └── OneHotEncoder       → features catégorielles
└── modele (GradientBoostingRegressor)
```

### Hyperparamètres optimaux (GridSearchCV, 3-fold CV)

| Paramètre | Valeur |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `min_samples_leaf` | 5 |
| `random_state` | 42 |

### Performances

| Métrique | Valeur |
|---|---|
| R² (test) | 0.715 |
| MAE log (test) | 0.48 |

### Variable cible

La cible est `log_conso = log1p(SiteEnergyUse(kBtu))`. La prédiction est reconvertie en kBtu via `np.expm1()`.

---

## API FastAPI

### Endpoints

#### `GET /health`
Vérifie l'état de l'API et du modèle.

**Réponse :**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### `POST /predict`
Prédit la consommation énergétique d'un bâtiment.

**Corps de la requête :**
```json
{
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
```

**Réponse :**
```json
{
  "prediction_kbtu": 735095.0,
  "prediction_log": 13.5,
  "input_data": { ... }
}
```

### Validation Pydantic

| Champ | Contrainte |
|---|---|
| `PropertyGFATotal` | > 0 |
| `PropertyGFABuilding` | > 0 et ≤ PropertyGFATotal |
| `YearBuilt` | entre 1900 et 2016 |
| `tranche_age` | valeur parmi 5 catégories définies |
| `tranche_surface` | valeur parmi 4 catégories définies |

---

## Base de données PostgreSQL

### Schéma

```
buildings
├── id (PK)
├── OSEBuildingID (unique)
├── PropertyName, Address, City, ZipCode
├── Neighborhood, BuildingType, PrimaryPropertyType
├── YearBuilt, NumberofFloors, NumberofBuildings
├── PropertyGFATotal, PropertyGFABuilding
├── has_electricity, has_gas, has_steam, nb_sources_energie
├── SiteEnergyUse, log_conso
├── tranche_age, tranche_surface
└── created_at

predictions
├── id (PK)
├── [toutes les features d'entrée]
├── prediction_log
├── prediction_kbtu
└── created_at
```

### Connexion

La connexion utilise SQLAlchemy avec l'URL définie dans `.env` :
```
DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/seattle_energy
```

---

## CI/CD

### Pipeline GitHub Actions

Le fichier `.github/workflows/ci.yml` définit deux jobs :

**1. Tests automatiques** (toutes les branches)
- Installation Python 3.11
- Installation des dépendances
- Exécution de pytest avec rapport de couverture

**2. Déploiement Hugging Face Spaces** (branche `main` uniquement)
- Push automatique vers HF Spaces après succès des tests

### Secrets GitHub requis

| Secret | Description |
|---|---|
| `HF_TOKEN` | Token d'accès Hugging Face (write) |

---

## Tests

### Lancer les tests

```bash
pytest
```

### Rapport de couverture

```bash
pytest --cov=app --cov-report=html
```

Le rapport HTML est généré dans `htmlcov/index.html`.

### Résultats actuels

| Fichier | Couverture |
|---|---|
| `app/schemas.py` | 97% |
| `app/predict.py` | 95% |
| `app/database.py` | 80% |
| `app/main.py` | 75% |
| **TOTAL** | **90%** |

---

## Choix techniques

| Choix | Justification |
|---|---|
| GradientBoosting vs XGBoost | Meilleur R² sur ce dataset après GridSearchCV |
| FastAPI vs Flask | Validation automatique Pydantic, Swagger intégré |
| PostgreSQL vs SQLite | Robustesse, scalabilité, support production |
| SQLAlchemy ORM | Abstraction DB, migration facilitée |
| log1p transform | Normalise la distribution très asymétrique de la cible |

---

## Mise à jour du modèle

1. Réentraîner le modèle dans le notebook
2. Exporter avec `joblib.dump(model, 'models/model.pkl')`
3. Remplacer `models/model.pkl` en local
4. Relancer les tests : `pytest`
5. Commiter et pusher sur `main`
