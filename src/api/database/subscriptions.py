"""
Opérations CRUD pour les abonnements.

Ce module fournit les fonctions d'accès à la base de données
pour créer, lire, mettre à jour et supprimer des abonnements et paiements.
"""

from sqlalchemy import select, update, delete, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
import uuid
import logging
from typing import List, Dict, Optional, Any

from src.api.database.models.subscription import (
    Subscription, UserSubscription, Payment,
    SubscriptionTierEnum, PaymentStatusEnum, PaymentMethodEnum
)
from src.api.database.models.user import User
from src.api.database.session import get_db

logger = logging.getLogger("api.database.subscriptions")


# --- ABONNEMENTS ---

async def create_subscription(
    name: str,
    tier: SubscriptionTierEnum,
    price_monthly: float,
    price_yearly: float,
    features: List[Dict[str, Any]],
    description: str = None,
    currency: str = "EUR",
    is_active: bool = True,
    db: AsyncSession = None
) -> Subscription:
    """
    Crée un nouvel abonnement dans la base de données.
    
    Args:
        name: Nom de l'abonnement
        tier: Niveau d'abonnement
        price_monthly: Prix mensuel
        price_yearly: Prix annuel
        features: Liste des fonctionnalités
        description: Description
        currency: Devise
        is_active: Statut d'activation
        db: Session de base de données
        
    Returns:
        Subscription: L'abonnement créé
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    subscription = Subscription(
        id=str(uuid.uuid4()),
        name=name,
        tier=tier,
        price_monthly=price_monthly,
        price_yearly=price_yearly,
        features=features,
        description=description,
        currency=currency,
        is_active=is_active
    )
    
    db.add(subscription)
    await db.commit()
    await db.refresh(subscription)
    
    logger.info(f"Abonnement créé: {name}")
    
    return subscription


async def get_subscription(
    subscription_id: str,
    db: AsyncSession = None
) -> Subscription:
    """
    Récupère un abonnement par son ID.
    
    Args:
        subscription_id: ID de l'abonnement
        db: Session de base de données
    
    Returns:
        Subscription ou None: L'abonnement trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Subscription).where(Subscription.id == subscription_id)
    result = await db.execute(query)
    subscription = result.scalars().first()
    
    return subscription


async def get_subscription_by_tier(
    tier: SubscriptionTierEnum,
    db: AsyncSession = None
) -> Subscription:
    """
    Récupère un abonnement par son niveau.
    
    Args:
        tier: Niveau d'abonnement
        db: Session de base de données
    
    Returns:
        Subscription ou None: L'abonnement trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Subscription).where(Subscription.tier == tier, Subscription.is_active == True)
    result = await db.execute(query)
    subscription = result.scalars().first()
    
    return subscription


async def get_all_subscriptions(
    active_only: bool = True,
    db: AsyncSession = None
) -> List[Subscription]:
    """
    Récupère tous les abonnements.
    
    Args:
        active_only: Si True, récupère uniquement les abonnements actifs
        db: Session de base de données
    
    Returns:
        List[Subscription]: Liste des abonnements
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Subscription)
    if active_only:
        query = query.where(Subscription.is_active == True)
    
    query = query.order_by(asc(Subscription.price_monthly))
    
    result = await db.execute(query)
    subscriptions = result.scalars().all()
    
    return list(subscriptions)


# --- ABONNEMENTS UTILISATEUR ---

async def create_user_subscription(
    user_id: str,
    subscription_id: str,
    is_yearly: bool = False,
    auto_renew: bool = True,
    is_trial: bool = False,
    amount_paid: float = 0.0,
    currency: str = "EUR",
    duration_months: int = None,
    db: AsyncSession = None
) -> UserSubscription:
    """
    Crée un nouvel abonnement utilisateur dans la base de données.
    
    Args:
        user_id: ID de l'utilisateur
        subscription_id: ID de l'abonnement
        is_yearly: Si True, abonnement annuel, sinon mensuel
        auto_renew: Si True, renouvellement automatique
        is_trial: Si True, période d'essai
        amount_paid: Montant payé
        currency: Devise
        duration_months: Durée en mois (par défaut 1 ou 12 selon is_yearly)
        db: Session de base de données
        
    Returns:
        UserSubscription: L'abonnement utilisateur créé
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    # Déterminer la durée de l'abonnement
    if duration_months is None:
        duration_months = 12 if is_yearly else 1
    
    # Calculer les dates de début et de fin
    start_date = datetime.utcnow()
    end_date = start_date + timedelta(days=30 * duration_months)
    
    # Créer l'abonnement utilisateur
    user_subscription = UserSubscription(
        id=str(uuid.uuid4()),
        user_id=user_id,
        subscription_id=subscription_id,
        start_date=start_date,
        end_date=end_date,
        is_yearly=is_yearly,
        auto_renew=auto_renew,
        is_active=True,
        is_trial=is_trial,
        amount_paid=amount_paid,
        currency=currency
    )
    
    db.add(user_subscription)
    await db.commit()
    await db.refresh(user_subscription)
    
    # Mettre à jour le niveau d'abonnement de l'utilisateur
    subscription = await get_subscription(subscription_id, db)
    if subscription:
        user = await db.get(User, user_id)
        if user:
            user.subscription_tier = subscription.tier
            await db.commit()
    
    logger.info(f"Abonnement utilisateur créé: {user_id} - {subscription_id}")
    
    return user_subscription


async def get_user_subscription(
    user_subscription_id: str,
    db: AsyncSession = None
) -> UserSubscription:
    """
    Récupère un abonnement utilisateur par son ID.
    
    Args:
        user_subscription_id: ID de l'abonnement utilisateur
        db: Session de base de données
    
    Returns:
        UserSubscription ou None: L'abonnement utilisateur trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(UserSubscription).where(UserSubscription.id == user_subscription_id)
    result = await db.execute(query)
    user_subscription = result.scalars().first()
    
    return user_subscription


async def get_active_user_subscription(
    user_id: str,
    db: AsyncSession = None
) -> UserSubscription:
    """
    Récupère l'abonnement actif d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        db: Session de base de données
    
    Returns:
        UserSubscription ou None: L'abonnement utilisateur actif ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    now = datetime.utcnow()
    
    query = select(UserSubscription).where(
        UserSubscription.user_id == user_id,
        UserSubscription.is_active == True,
        UserSubscription.start_date <= now,
        UserSubscription.end_date > now
    ).order_by(desc(UserSubscription.end_date))
    
    result = await db.execute(query)
    user_subscription = result.scalars().first()
    
    return user_subscription


async def get_user_subscription_history(
    user_id: str,
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = None
) -> List[UserSubscription]:
    """
    Récupère l'historique des abonnements d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        limit: Nombre maximum de résultats
        offset: Décalage pour la pagination
        db: Session de base de données
    
    Returns:
        List[UserSubscription]: Liste des abonnements utilisateur
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(UserSubscription).where(
        UserSubscription.user_id == user_id
    ).order_by(desc(UserSubscription.start_date))
    
    query = query.limit(limit).offset(offset)
    
    result = await db.execute(query)
    user_subscriptions = result.scalars().all()
    
    return list(user_subscriptions)


async def cancel_user_subscription(
    user_subscription_id: str,
    db: AsyncSession = None
) -> UserSubscription:
    """
    Annule le renouvellement automatique d'un abonnement utilisateur.
    
    Args:
        user_subscription_id: ID de l'abonnement utilisateur
        db: Session de base de données
    
    Returns:
        UserSubscription: L'abonnement utilisateur mis à jour
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    user_subscription = await get_user_subscription(user_subscription_id, db)
    if not user_subscription:
        return None
    
    user_subscription.auto_renew = False
    user_subscription.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(user_subscription)
    
    logger.info(f"Abonnement utilisateur annulé: {user_subscription_id}")
    
    return user_subscription


# --- PAIEMENTS ---

async def create_payment(
    user_subscription_id: str,
    amount: float,
    payment_method: PaymentMethodEnum,
    currency: str = "EUR",
    transaction_id: str = None,
    payment_provider: str = None,
    payment_status: PaymentStatusEnum = PaymentStatusEnum.PENDING,
    metadata: Dict[str, Any] = None,
    db: AsyncSession = None
) -> Payment:
    """
    Crée un nouveau paiement dans la base de données.
    
    Args:
        user_subscription_id: ID de l'abonnement utilisateur
        amount: Montant
        payment_method: Méthode de paiement
        currency: Devise
        transaction_id: ID de transaction externe
        payment_provider: Fournisseur de paiement
        payment_status: Statut du paiement
        metadata: Métadonnées
        db: Session de base de données
        
    Returns:
        Payment: Le paiement créé
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    payment = Payment(
        id=str(uuid.uuid4()),
        user_subscription_id=user_subscription_id,
        amount=amount,
        currency=currency,
        payment_method=payment_method,
        payment_status=payment_status,
        transaction_id=transaction_id,
        payment_provider=payment_provider,
        metadata=metadata or {},
        payment_date=datetime.utcnow() if payment_status == PaymentStatusEnum.COMPLETED else None
    )
    
    db.add(payment)
    await db.commit()
    await db.refresh(payment)
    
    logger.info(f"Paiement créé: {payment.id} pour l'abonnement {user_subscription_id}")
    
    return payment


async def get_payment(
    payment_id: str,
    db: AsyncSession = None
) -> Payment:
    """
    Récupère un paiement par son ID.
    
    Args:
        payment_id: ID du paiement
        db: Session de base de données
    
    Returns:
        Payment ou None: Le paiement trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Payment).where(Payment.id == payment_id)
    result = await db.execute(query)
    payment = result.scalars().first()
    
    return payment


async def get_payment_by_transaction_id(
    transaction_id: str,
    db: AsyncSession = None
) -> Payment:
    """
    Récupère un paiement par son ID de transaction externe.
    
    Args:
        transaction_id: ID de transaction externe
        db: Session de base de données
    
    Returns:
        Payment ou None: Le paiement trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Payment).where(Payment.transaction_id == transaction_id)
    result = await db.execute(query)
    payment = result.scalars().first()
    
    return payment


async def update_payment_status(
    payment_id: str,
    payment_status: PaymentStatusEnum,
    transaction_id: str = None,
    metadata: Dict[str, Any] = None,
    db: AsyncSession = None
) -> Payment:
    """
    Met à jour le statut d'un paiement.
    
    Args:
        payment_id: ID du paiement
        payment_status: Nouveau statut
        transaction_id: ID de transaction externe
        metadata: Métadonnées supplémentaires
        db: Session de base de données
    
    Returns:
        Payment: Le paiement mis à jour
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    payment = await get_payment(payment_id, db)
    if not payment:
        return None
    
    payment.payment_status = payment_status
    
    if transaction_id:
        payment.transaction_id = transaction_id
    
    if metadata:
        if not payment.metadata:
            payment.metadata = {}
        payment.metadata.update(metadata)
    
    if payment_status == PaymentStatusEnum.COMPLETED and not payment.payment_date:
        payment.payment_date = datetime.utcnow()
    
    payment.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(payment)
    
    logger.info(f"Statut du paiement {payment_id} mis à jour: {payment_status}")
    
    return payment


async def get_subscription_payments(
    user_subscription_id: str,
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = None
) -> List[Payment]:
    """
    Récupère les paiements d'un abonnement utilisateur.
    
    Args:
        user_subscription_id: ID de l'abonnement utilisateur
        limit: Nombre maximum de résultats
        offset: Décalage pour la pagination
        db: Session de base de données
    
    Returns:
        List[Payment]: Liste des paiements
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Payment).where(
        Payment.user_subscription_id == user_subscription_id
    ).order_by(desc(Payment.created_at))
    
    query = query.limit(limit).offset(offset)
    
    result = await db.execute(query)
    payments = result.scalars().all()
    
    return list(payments)


async def get_user_payments(
    user_id: str,
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = None
) -> List[Payment]:
    """
    Récupère tous les paiements d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        limit: Nombre maximum de résultats
        offset: Décalage pour la pagination
        db: Session de base de données
    
    Returns:
        List[Payment]: Liste des paiements
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Payment).join(
        UserSubscription,
        Payment.user_subscription_id == UserSubscription.id
    ).where(
        UserSubscription.user_id == user_id
    ).order_by(desc(Payment.created_at))
    
    query = query.limit(limit).offset(offset)
    
    result = await db.execute(query)
    payments = result.scalars().all()
    
    return list(payments) 