"""Module de connexion et interaction avec la base de données."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/seattle_energy")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Générateur de session pour les endpoints FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(db, input_data: dict, prediction_log: float, prediction_kbtu: float):
    """Enregistre une prédiction dans la base de données."""
    from scripts.create_db import Prediction

    pred = Prediction(
        **{k: v for k, v in input_data.items()},
        prediction_log=prediction_log,
        prediction_kbtu=prediction_kbtu,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred
