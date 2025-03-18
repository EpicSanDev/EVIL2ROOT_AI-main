"""
Modèles de base pour SQLAlchemy.

Ce module définit les classes et mixins de base pour les modèles SQLAlchemy
utilisés dans toute l'application.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, String, Boolean
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import declarative_mixin

from src.api.database.session import Base


@declarative_mixin
class TimestampMixin:
    """Mixin pour ajouter des timestamps de création et de mise à jour."""
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow, 
        nullable=False
    )


@declarative_mixin
class UUIDMixin:
    """Mixin pour utiliser un UUID comme clé primaire."""
    
    id = Column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4()),
        index=True
    )


@declarative_mixin
class SoftDeleteMixin:
    """Mixin pour la suppression logique."""
    
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)
    
    def soft_delete(self):
        """Marque l'enregistrement comme supprimé."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()


@declarative_mixin
class TableNameMixin:
    """Mixin pour générer automatiquement le nom de la table."""
    
    @declared_attr
    def __tablename__(cls):
        """Génère le nom de la table à partir du nom de la classe."""
        return cls.__name__.lower() + 's'


class BaseModel(UUIDMixin, TimestampMixin, TableNameMixin, Base):
    """Classe de base pour tous les modèles avec UUID et timestamps."""
    
    __abstract__ = True


class SoftDeleteBaseModel(BaseModel, SoftDeleteMixin):
    """Classe de base avec UUID, timestamps et suppression logique."""
    
    __abstract__ = True 