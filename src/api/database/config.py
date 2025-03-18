"""
Configuration de la base de données pour l'API EVIL2ROOT.

Ce module gère les paramètres de connexion à la base de données
en utilisant les variables d'environnement ou des valeurs par défaut.
"""

import os
from pydantic import BaseSettings, Field
from typing import Optional


class DatabaseSettings(BaseSettings):
    """Paramètres de configuration de la base de données."""
    
    # Type de base de données (postgresql, mysql, sqlite)
    DB_TYPE: str = Field(
        default="postgresql",
        env="DB_TYPE"
    )
    
    # Informations de connexion
    DB_USER: str = Field(
        default="postgres",
        env="DB_USER"
    )
    DB_PASSWORD: str = Field(
        default="postgres",
        env="DB_PASSWORD"
    )
    DB_HOST: str = Field(
        default="localhost",
        env="DB_HOST"
    )
    DB_PORT: int = Field(
        default=5432,
        env="DB_PORT"
    )
    DB_NAME: str = Field(
        default="evil2root",
        env="DB_NAME"
    )
    
    # URL de connexion complète (optionnelle)
    DATABASE_URL: Optional[str] = Field(
        default=None,
        env="DATABASE_URL"
    )
    
    # Mode de développement
    DB_ECHO: bool = Field(
        default=False,
        env="DB_ECHO"
    )
    
    class Config:
        """Configuration Pydantic."""
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_connection_url(self) -> str:
        """
        Construit et retourne l'URL de connexion à la base de données.
        
        Si DATABASE_URL est défini directement, il sera utilisé en priorité.
        Sinon, l'URL sera construite à partir des autres paramètres.
        
        Returns:
            str: URL de connexion à la base de données.
        """
        if self.DATABASE_URL:
            return self.DATABASE_URL
        
        # Construction de l'URL selon le type de base de données
        if self.DB_TYPE == "sqlite":
            return f"sqlite:///{self.DB_NAME}.db"
        elif self.DB_TYPE == "postgresql":
            return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        elif self.DB_TYPE == "mysql":
            return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        else:
            raise ValueError(f"Type de base de données non supporté: {self.DB_TYPE}")


# Instance des paramètres de base de données
db_settings = DatabaseSettings() 