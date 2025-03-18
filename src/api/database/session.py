"""
Gestion de la session de base de données pour l'API EVIL2ROOT.

Ce module fournit les fonctions et classes nécessaires pour
interagir avec la base de données via SQLAlchemy.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import AsyncGenerator
from fastapi import Depends

from src.api.database.config import db_settings

# Création du moteur SQLAlchemy asynchrone
engine = create_async_engine(
    db_settings.get_connection_url(),
    echo=db_settings.DB_ECHO,
    future=True,
    pool_pre_ping=True,  # Vérifie la connexion avant utilisation
)

# Création de la session asynchrone
SessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Création de la classe de base pour les modèles
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dépendance FastAPI pour obtenir une session de base de données.
    
    Yields:
        AsyncSession: Session de base de données asynchrone.
    """
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Gestion des migrations
def run_migrations():
    """
    Exécute les migrations de base de données avec Alembic.
    Cette fonction est appelée au démarrage de l'application.
    """
    import subprocess
    import os
    from pathlib import Path
    
    # Chemin vers les migrations Alembic
    migrations_dir = Path(__file__).parent / "migrations"
    
    # Vérifier si le répertoire existe
    if not migrations_dir.exists():
        print("Répertoire de migrations non trouvé. Initialisation des migrations...")
        # Initialiser les migrations Alembic
        subprocess.run(["alembic", "init", "migrations"], check=True)
    
    # Exécuter les migrations
    print("Exécution des migrations...")
    subprocess.run(["alembic", "upgrade", "head"], check=True)
    
    print("Migrations terminées avec succès.") 