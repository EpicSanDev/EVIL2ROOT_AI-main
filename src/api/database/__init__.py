"""
Module de gestion de la base de données pour l'API EVIL2ROOT.

Ce module fournit les fonctionnalités d'accès et de manipulation
de la base de données pour toutes les entités du système.
"""

from src.api.database.session import engine, SessionLocal, get_db
from src.api.database.models import Base

__all__ = ["engine", "SessionLocal", "get_db", "Base"]

# Fonction pour initialiser la base de données
async def init_db():
    """Initialise la base de données et crée les tables si nécessaire."""
    from src.api.database.models import Base
    from src.api.database.session import engine
    
    # Créer les tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("Base de données initialisée avec succès.") 