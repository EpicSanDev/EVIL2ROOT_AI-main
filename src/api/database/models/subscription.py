"""
Modèles SQLAlchemy pour les abonnements.

Ce module définit les modèles de base de données pour les abonnements,
les paiements et les souscriptions utilisateurs.
"""

from datetime import datetime
from decimal import Decimal
from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey, 
    Integer, Numeric, JSON, Text, Enum, CheckConstraint
)
from sqlalchemy.orm import relationship
import enum

from src.api.database.models.base import BaseModel


class SubscriptionTierEnum(str, enum.Enum):
    """Énumération des niveaux d'abonnement disponibles."""
    
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class PaymentStatusEnum(str, enum.Enum):
    """Énumération des statuts de paiement possibles."""
    
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class PaymentMethodEnum(str, enum.Enum):
    """Énumération des méthodes de paiement supportées."""
    
    CREDIT_CARD = "credit_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"


class Subscription(BaseModel):
    """Modèle SQLAlchemy pour les types d'abonnements disponibles."""
    
    # Informations de base
    name = Column(String(50), nullable=False, unique=True)
    tier = Column(Enum(SubscriptionTierEnum), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Tarification
    price_monthly = Column(Numeric(10, 2), nullable=False)
    price_yearly = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default="EUR", nullable=False)
    
    # Caractéristiques
    features = Column(JSON, nullable=False)
    
    # Statut
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relations
    user_subscriptions = relationship("UserSubscription", back_populates="subscription")
    
    def __repr__(self):
        return f"<Subscription(id={self.id}, name={self.name}, tier={self.tier})>"
    
    def to_dict(self):
        """Convertit l'abonnement en dictionnaire."""
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier,
            "description": self.description,
            "price_monthly": float(self.price_monthly),
            "price_yearly": float(self.price_yearly),
            "currency": self.currency,
            "features": self.features,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class UserSubscription(BaseModel):
    """Modèle SQLAlchemy pour les abonnements utilisateur."""
    
    # Relations
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    user = relationship("User", back_populates="subscriptions")
    
    subscription_id = Column(String(36), ForeignKey("subscriptions.id"), nullable=False)
    subscription = relationship("Subscription", back_populates="user_subscriptions")
    
    # Période d'abonnement
    start_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=False)
    
    # Type d'abonnement
    is_yearly = Column(Boolean, default=False, nullable=False)
    auto_renew = Column(Boolean, default=True, nullable=False)
    
    # Statut
    is_active = Column(Boolean, default=True, nullable=False)
    is_trial = Column(Boolean, default=False, nullable=False)
    
    # Propriétés calculées pour la facturation
    amount_paid = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default="EUR", nullable=False)
    
    # Relations
    payments = relationship("Payment", back_populates="user_subscription", cascade="all, delete-orphan")
    
    __table_args__ = (
        # S'assurer que la date de fin est postérieure à la date de début
        CheckConstraint('end_date > start_date', name='check_subscription_dates'),
    )
    
    def __repr__(self):
        return f"<UserSubscription(user_id={self.user_id}, subscription_id={self.subscription_id})>"
    
    @property
    def is_expired(self):
        """Vérifie si l'abonnement est expiré."""
        return datetime.utcnow() > self.end_date
    
    def to_dict(self):
        """Convertit l'abonnement utilisateur en dictionnaire."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "subscription_id": self.subscription_id,
            "subscription_name": self.subscription.name if self.subscription else None,
            "subscription_tier": self.subscription.tier if self.subscription else None,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "is_yearly": self.is_yearly,
            "auto_renew": self.auto_renew,
            "is_active": self.is_active,
            "is_trial": self.is_trial,
            "is_expired": self.is_expired,
            "amount_paid": float(self.amount_paid),
            "currency": self.currency,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class Payment(BaseModel):
    """Modèle SQLAlchemy pour les paiements d'abonnement."""
    
    # Relations
    user_subscription_id = Column(String(36), ForeignKey("usersubscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    user_subscription = relationship("UserSubscription", back_populates="payments")
    
    # Informations de paiement
    amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default="EUR", nullable=False)
    payment_method = Column(Enum(PaymentMethodEnum), nullable=False)
    payment_status = Column(Enum(PaymentStatusEnum), default=PaymentStatusEnum.PENDING, nullable=False)
    
    # Identifiants externes
    transaction_id = Column(String(100), nullable=True, unique=True)
    payment_provider = Column(String(50), nullable=True)
    
    # Dates importantes
    payment_date = Column(DateTime, nullable=True)
    
    # Métadonnées
    payment_metadata = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<Payment(id={self.id}, amount={self.amount}, status={self.payment_status})>"
    
    def to_dict(self):
        """Convertit le paiement en dictionnaire."""
        return {
            "id": self.id,
            "user_subscription_id": self.user_subscription_id,
            "amount": float(self.amount),
            "currency": self.currency,
            "payment_method": self.payment_method,
            "payment_status": self.payment_status,
            "transaction_id": self.transaction_id,
            "payment_provider": self.payment_provider,
            "payment_date": self.payment_date,
            "metadata": self.payment_metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        } 