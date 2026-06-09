# Prédiction de la consommation énergétique des bâtiments de Seattle

Projet de machine learning visant à prédire la consommation énergétique (`SiteEnergyUse` en kBtu) de bâtiments commerciaux à Seattle, à partir du dataset public *2016 Building Energy Benchmarking*.

\---

## Contexte

La ville de Seattle impose aux bâtiments de plus de 20 000 pieds carrés de déclarer annuellement leur consommation énergétique. Ce projet exploite ces données pour construire un modèle de régression permettant d'estimer la consommation à partir de caractéristiques structurelles du bâtiment, sans avoir besoin des relevés de compteurs.

\---

## Architecture du projet


!\[Architecture](docs/architecture.svg)

## Modèle

\* \*\*Algorithme :\*\* `GradientBoostingRegressor` (scikit-learn)
\* \*\*Pipeline :\*\* StandardScaler (numériques) + OneHotEncoder (catégorielles) + GBR
\* \*\*Optimisation :\*\* GridSearchCV (5-fold CV)
\* \*\*Meilleurs hyperparamètres :\*\* `n\\\_estimators=300`, `max\\\_depth=3`, `learning\\\_rate=0.05`, `min\\\_samples\\\_leaf=5`
\* \*\*Performances :\*\* R² = 0.715 | MAE (log) = 0.48 sur le jeu de test

### Features d'entrée

|Feature|Type|Description|
|-|-|-|
|`log\\\_surface\\\_totale`|Numérique|Log de la surface totale (GFA)|
|`log\\\_surface\\\_batiment`|Numérique|Log de la surface construite|
|`ratio\\\_surface`|Numérique|Ratio surface bâtiment / surface totale|
|`building\\\_age`|Numérique|Âge du bâtiment (2016 - YearBuilt)|
|`NumberofFloors`|Numérique|Nombre d'étages|
|`NumberofBuildings`|Numérique|Nombre de bâtiments sur la parcelle|
|`has\\\_electricity`|Binaire|Utilise l'électricité|
|`has\\\_gas`|Binaire|Utilise le gaz naturel|
|`has\\\_steam`|Binaire|Utilise la vapeur|
|`nb\\\_sources\\\_energie`|Numérique|Nombre de sources d'énergie|
|`Neighborhood\\\_enc`|Encodée|Quartier (label encoding)|
|`BuildingType\\\_enc`|Encodée|Type de bâtiment (label encoding)|
|`ZipCode\\\_enc`|Encodée|Code postal (label encoding)|
|`PrimaryPropertyType`|Catégorielle|Type d'usage principal (OHE)|
|`tranche\\\_age`|Catégorielle|Tranche d'âge du bâtiment (OHE)|
|`tranche\\\_surface`|Catégorielle|Tranche de surface (OHE)|

### Cible

\* `log\\\_conso` : log1p de `SiteEnergyUse(kBtu)` — reconverti avec `np.expm1` dans la réponse API

\\---

## Installation

### Prérequis

\* Python 3.11+
\* PostgreSQL 15+
\* Git

### Cloner le projet

```bash
git clone https://github.com/<ton-username>/seattle-energy-prediction.git
cd seattle-energy-prediction
```

### Créer l'environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\\\\Scripts\\\\activate           # Windows
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```

### Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditer .env avec tes identifiants PostgreSQL
```

### Placer les fichiers nécessaires

```
data/raw/2016\\\_Building\\\_Energy\\\_Benchmarking.csv
models/model.pkl   # modèle entraîné exporté depuis le notebook
```

\---

## Lancer l'API

```bash
uvicorn app.main:app --reload
```

L'API est disponible sur `http://localhost:8000`.  
La documentation Swagger est accessible sur `http://localhost:8000/docs`.

\---

## Lancer les tests

```bash
pytest --cov=app tests/ -v
```

\---

## Déploiement

Le projet est déployé sur Hugging Face Spaces via le pipeline CI/CD GitHub Actions. Chaque push sur la branche `main` déclenche automatiquement :

1. Les tests unitaires et fonctionnels
2. Le déploiement sur Hugging Face Spaces (si les tests passent)

\---

## Conventions de branches

|Branche|Usage|
|-|-|
|`main`|Code stable, déployé en production|
|`develop`|Intégration des features avant merge sur main|
|`feature/<nom>`|Développement d'une nouvelle fonctionnalité|
|`fix/<nom>`|Correction de bug|

\---

## Auteur

Fatima Adda-Rezig

