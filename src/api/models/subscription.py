"""
Modèles de données pour les abonnements et la facturation.

Ce module définit les modèles Pydantic pour les abonnements, plans et factures.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class SubscriptionTier(str, Enum):
    """Niveaux d'abonnement disponibles."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class BillingPeriod(str, Enum):
    """Périodes de facturation."""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class PaymentMethod(str, Enum):
    """Méthodes de paiement supportées."""
    CREDIT_CARD = "credit_card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"
    CRYPTO = "crypto"

class SubscriptionStatus(str, Enum):
    """Statuts possibles pour un abonnement."""
    ACTIVE = "active"
    PENDING = "pending"
    CANCELED = "canceled"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    TRIAL = "trial"

class InvoiceStatus(str, Enum):
    """Statuts possibles pour une facture."""
    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"
    CANCELED = "canceled"
    REFUNDED = "refunded"

class PricingPlan(BaseModel):
    """Modèle pour un plan tarifaire."""
    
    id: str
    name: str
    tier: SubscriptionTier
    price_monthly: float
    price_quarterly: float
    price_annually: float
    description: str
    features: List[str]
    max_symbols: int
    max_strategies: int
    max_backtests_per_day: int
    trading_modes: List[str]
    api_rate_limit: int
    premium_indicators: bool
    ai_suggestions: bool
    telegram_notifications: bool
    phone_support: bool
    is_active: bool = True
    
    class Config:
        orm_mode = True

class SubscriptionCreate(BaseModel):
    """Modèle pour la création d'un abonnement."""
    
    user_id: str
    plan_id: str
    billing_period: BillingPeriod
    payment_method: PaymentMethod
    coupon_code: Optional[str] = None
    auto_renew: bool = True

class SubscriptionResponse(BaseModel):
    """Modèle de réponse pour un abonnement."""
    
    id: str
    user_id: str
    plan_id: str
    tier: SubscriptionTier
    status: SubscriptionStatus
    start_date: datetime
    end_date: datetime
    billing_period: BillingPeriod
    payment_method: PaymentMethod
    auto_renew: bool
    next_billing_date: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class SubscriptionUpdate(BaseModel):
    """Modèle pour la mise à jour d'un abonnement."""
    
    plan_id: Optional[str] = None
    billing_period: Optional[BillingPeriod] = None
    payment_method: Optional[PaymentMethod] = None
    auto_renew: Optional[bool] = None
    status: Optional[SubscriptionStatus] = None

class InvoiceResponse(BaseModel):
    """Modèle de réponse pour une facture."""
    
    id: str
    user_id: str
    subscription_id: str
    amount: float
    tax_amount: float
    total_amount: float
    currency: str = "EUR"
    status: InvoiceStatus
    due_date: datetime
    paid_at: Optional[datetime] = None
    payment_method: PaymentMethod
    payment_details: Dict[str, Any] = {}
    invoice_number: str
    items: List[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        orm_mode = True

class CouponResponse(BaseModel):
    """Modèle de réponse pour un coupon de réduction."""
    
    id: str
    code: str
    description: str
    discount_percentage: float
    discount_amount: float
    valid_from: datetime
    valid_until: datetime
    max_uses: Optional[int] = None
    current_uses: int = 0
    is_active: bool = True
    created_at: datetime
    
    class Config:
        orm_mode = True

class UserSubscriptionDetails(BaseModel):
    """Modèle pour les détails d'abonnement d'un utilisateur."""
    
    user_id: str
    subscription: Optional[SubscriptionResponse] = None
    plan: Optional[PricingPlan] = None
    usage_stats: Dict[str, Any] = {
        "symbols_used": 0,
        "strategies_used": 0,
        "backtests_today": 0,
        "api_calls_today": 0
    }
    invoices: List[InvoiceResponse] = []
    
    class Config:
        orm_mode = True

class PaymentResponse(BaseModel):
    """Modèle de réponse pour un paiement."""
    
    id: str
    user_id: str
    invoice_id: str
    amount: float
    currency: str = "EUR"
    payment_method: PaymentMethod
    payment_details: Dict[str, Any]
    status: str
    transaction_id: Optional[str] = None
    created_at: datetime
    
    class Config:
        orm_mode = True 