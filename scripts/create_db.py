"""
Script de création de la base de données PostgreSQL.
Usage : python scripts/create_db.py
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/seattle_energy")

Base = declarative_base()


class Building(Base):
    """Table des bâtiments du dataset Seattle."""
    __tablename__ = "buildings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    OSEBuildingID = Column(String, unique=True, nullable=False)
    PropertyName = Column(String)
    Address = Column(String)
    City = Column(String)
    ZipCode = Column(Float)
    Neighborhood = Column(String)
    BuildingType = Column(String)
    PrimaryPropertyType = Column(String)
    YearBuilt = Column(Integer)
    NumberofFloors = Column(Integer)
    NumberofBuildings = Column(Float)
    PropertyGFATotal = Column(Float)
    PropertyGFABuilding = Column(Float)
    has_electricity = Column(Integer)
    has_gas = Column(Integer)
    has_steam = Column(Integer)
    nb_sources_energie = Column(Integer)
    SiteEnergyUse = Column(Float)
    log_conso = Column(Float)
    tranche_age = Column(String)
    tranche_surface = Column(String)
    created_at = Column(DateTime, server_default=func.now())


class Prediction(Base):
    """Table des prédictions faites via l'API."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Inputs
    PropertyGFATotal = Column(Float)
    PropertyGFABuilding = Column(Float)
    YearBuilt = Column(Integer)
    NumberofFloors = Column(Integer)
    NumberofBuildings = Column(Float)
    has_electricity = Column(Integer)
    has_gas = Column(Integer)
    has_steam = Column(Integer)
    Neighborhood = Column(String)
    BuildingType = Column(String)
    ZipCode = Column(Float)
    PrimaryPropertyType = Column(String)
    tranche_age = Column(String)
    tranche_surface = Column(String)
    # Outputs
    prediction_log = Column(Float)
    prediction_kbtu = Column(Float)
    # Metadata
    created_at = Column(DateTime, server_default=func.now())


def create_tables():
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    print("✅ Tables créées avec succès !")
    print("   - buildings")
    print("   - predictions")
    return engine


if __name__ == "__main__":
    create_tables()
