"""
Script d'insertion du dataset dans la base de données PostgreSQL.
Usage : python scripts/insert_data.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/seattle_energy")
DATA_PATH = os.getenv("DATA_PATH", "data/raw/2016_Building_Energy_Benchmarking.csv")


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare le dataset brut pour l'insertion."""

    # Nettoyage de base
    df = df.copy()
    df = df.dropna(subset=["SiteEnergyUse(kBtu)", "PropertyGFATotal"])
    df = df[df["SiteEnergyUse(kBtu)"] > 0]
    df = df[df["PropertyGFATotal"] > 0]

    # Features dérivées
    df["has_electricity"] = df["Electricity(kBtu)"].notna().astype(int) if "Electricity(kBtu)" in df.columns else 0
    df["has_gas"] = df["NaturalGas(kBtu)"].notna().astype(int) if "NaturalGas(kBtu)" in df.columns else 0
    df["has_steam"] = df["SteamUse(kBtu)"].notna().astype(int) if "SteamUse(kBtu)" in df.columns else 0
    df["nb_sources_energie"] = df["has_electricity"] + df["has_gas"] + df["has_steam"]
    df["log_conso"] = np.log1p(df["SiteEnergyUse(kBtu)"])

    # Tranches d'âge
    df["building_age"] = 2016 - df["YearBuilt"].fillna(1980)
    bins_age = [0, 56, 76, 96, 116, float("inf")]
    labels_age = ["2000+", "1980-1999", "1960-1979", "1940-1959", "avant 1940"]
    df["tranche_age"] = pd.cut(df["building_age"], bins=bins_age, labels=labels_age)

    # Tranches de surface
    bins_surf = [0, 10000, 50000, 200000, float("inf")]
    labels_surf = ["petit", "moyen", "grand", "tres grand"]
    df["tranche_surface"] = pd.cut(df["PropertyGFATotal"], bins=bins_surf, labels=labels_surf)

    # Sélection des colonnes
    cols = [
        "OSEBuildingID", "PropertyName", "Address", "City", "ZipCode",
        "Neighborhood", "BuildingType", "PrimaryPropertyType",
        "YearBuilt", "NumberofFloors", "NumberofBuildings",
        "PropertyGFATotal", "PropertyGFABuilding",
        "has_electricity", "has_gas", "has_steam", "nb_sources_energie",
        "SiteEnergyUse(kBtu)", "log_conso", "tranche_age", "tranche_surface"
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].rename(columns={"SiteEnergyUse(kBtu)": "SiteEnergyUse"})

    return df


def insert_data():
    print(f"📂 Chargement du dataset depuis {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"   {len(df)} lignes chargées.")

    df = prepare_dataset(df)
    print(f"   {len(df)} lignes après nettoyage.")

    engine = create_engine(DATABASE_URL)

    # Vider la table avant insertion
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE buildings RESTART IDENTITY CASCADE"))
        conn.commit()

    df.to_sql("buildings", engine, if_exists="append", index=False)
    print(f"✅ {len(df)} bâtiments insérés dans la table 'buildings' !")


if __name__ == "__main__":
    insert_data()
