"""
Modèles SQLAlchemy pour les utilisateurs.

Ce module définit les modèles de base de données pour les utilisateurs
et les entités associées (préférences, tokens de réinitialisation, etc.).
"""

import uuid
from datetime import datetime, timedelta
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from pydantic import EmailStr

from src.api.database.models.base import SoftDeleteBaseModel, BaseModel


class User(SoftDeleteBaseModel):
    """Modèle SQLAlchemy pour les utilisateurs."""
    
    # Informations de base
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Statut du compte
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_token = Column(String(100), nullable=True)
    
    # Abonnement
    subscription_tier = Column(String(50), default="free", nullable=False)
    
    # Dates importantes
    last_login = Column(DateTime, nullable=True)
    
    # Relations
    preferences = relationship("UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")
    reset_tokens = relationship("PasswordResetToken", back_populates="user", cascade="all, delete-orphan")
    subscriptions = relationship("UserSubscription", back_populates="user", cascade="all, delete-orphan")
    trading_accounts = relationship("TradingAccount", back_populates="user", cascade="all, delete-orphan")
    strategies = relationship("TradingStrategy", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
    
    def to_dict(self):
        """Convertit l'utilisateur en dictionnaire."""
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "subscription_tier": self.subscription_tier,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_login": self.last_login,
            "preferences": self.preferences.to_dict() if self.preferences else None
        }
    
    def to_response(self):
        """
        Convertit l'utilisateur en modèle de réponse Pydantic.
        Compatible avec le modèle UserResponse existant.
        """
        from src.api.models.user import UserResponse
        
        return UserResponse(
            id=self.id,
            email=self.email,
            full_name=self.full_name,
            is_active=self.is_active,
            subscription_tier=self.subscription_tier,
            created_at=self.created_at,
            last_login=self.last_login
        )


class UserPreference(BaseModel):
    """Modèle SQLAlchemy pour les préférences utilisateur."""
    
    # Relation avec l'utilisateur
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    user = relationship("User", back_populates="preferences")
    
    # Préférences d'interface
    theme = Column(String(20), default="light", nullable=False)
    language = Column(String(10), default="fr", nullable=False)
    timezone = Column(String(50), default="Europe/Paris", nullable=False)
    
    # Préférences de notification
    notifications_enabled = Column(Boolean, default=True, nullable=False)
    email_notifications = Column(Boolean, default=True, nullable=False)
    telegram_notifications = Column(Boolean, default=False, nullable=False)
    
    # Préférences commerciales
    display_currency = Column(String(10), default="EUR", nullable=False)
    
    # Widgets du tableau de bord (stockés en JSON)
    dashboard_widgets = Column(JSON, default=dict, nullable=True)
    
    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id})>"
    
    def to_dict(self):
        """Convertit les préférences en dictionnaire."""
        return {
            "theme": self.theme,
            "language": self.language,
            "timezone": self.timezone,
            "notifications_enabled": self.notifications_enabled,
            "email_notifications": self.email_notifications,
            "telegram_notifications": self.telegram_notifications,
            "display_currency": self.display_currency,
            "dashboard_widgets": self.dashboard_widgets
        }


class PasswordResetToken(BaseModel):
    """Modèle SQLAlchemy pour les tokens de réinitialisation de mot de passe."""
    
    # Relation avec l'utilisateur
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    user = relationship("User", back_populates="reset_tokens")
    
    # Token et validité
    token = Column(String(100), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False)
    is_used = Column(Boolean, default=False, nullable=False)
    
    def __repr__(self):
        return f"<PasswordResetToken(user_id={self.user_id})>"
    
    @property
    def is_valid(self):
        """Vérifie si le token est valide (non expiré et non utilisé)."""
        return not self.is_used and datetime.utcnow() < self.expires_at
    
    @classmethod
    def generate(cls, user_id):
        """
        Génère un nouveau token de réinitialisation.
        
        Args:
            user_id (str): ID de l'utilisateur
            
        Returns:
            PasswordResetToken: Instance du modèle avec le token généré
        """
        token = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        return cls(
            user_id=user_id,
            token=token,
            expires_at=expires_at
        )
    
    def invalidate(self):
        """Invalide le token en le marquant comme utilisé."""
        self.is_used = True 