"""
Routes de gestion des abonnements pour l'API du bot de trading.

Ce module gère les abonnements, plans tarifaires et factures.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime, timedelta

from src.api.models.subscription import (
    PricingPlan,
    SubscriptionCreate,
    SubscriptionResponse,
    SubscriptionUpdate,
    InvoiceResponse,
    CouponResponse,
    UserSubscriptionDetails,
    PaymentResponse,
    BillingPeriod,
    SubscriptionStatus
)
from src.api.middleware.authentication import get_current_user
from src.api.database.subscriptions import (
    get_pricing_plans,
    get_plan_by_id,
    get_user_subscription,
    create_subscription,
    update_subscription,
    cancel_subscription,
    get_user_invoices,
    get_invoice_by_id,
    create_invoice,
    update_invoice_status,
    validate_coupon,
    get_user_subscription_details,
    create_payment
)
from src.api.services.payment_service import (
    process_payment,
    calculate_subscription_price,
    calculate_next_billing_date
)
from src.api.utils.email import (
    send_subscription_confirmation_email,
    send_invoice_email
)

# Configuration du logger
logger = logging.getLogger("api.subscriptions")

# Configuration du router
router = APIRouter()

@router.get("/plans", response_model=List[PricingPlan])
async def get_plans():
    """
    Récupère tous les plans tarifaires disponibles.
    
    Returns:
        Liste des plans tarifaires
    """
    plans = await get_pricing_plans()
    return plans

@router.get("/my-subscription", response_model=UserSubscriptionDetails)
async def get_my_subscription(request: Request):
    """
    Récupère les détails d'abonnement de l'utilisateur actuel.
    
    Args:
        request: La requête HTTP
        
    Returns:
        Les détails d'abonnement
    """
    user = get_current_user(request)
    subscription_details = await get_user_subscription_details(user.id)
    return subscription_details

@router.post("/subscribe", response_model=SubscriptionResponse)
async def subscribe(
    subscription_data: SubscriptionCreate,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Souscrit à un plan tarifaire.
    
    Args:
        subscription_data: Les données d'abonnement
        request: La requête HTTP
        background_tasks: Tâches à exécuter en arrière-plan
        
    Returns:
        L'abonnement créé
        
    Raises:
        HTTPException: Si les données d'abonnement sont invalides
    """
    user = get_current_user(request)
    
    # Remplacer l'ID utilisateur par celui de l'utilisateur authentifié
    subscription_data.user_id = user.id
    
    # Vérifier si l'utilisateur a déjà un abonnement actif
    existing_subscription = await get_user_subscription(user.id)
    if existing_subscription and existing_subscription.status in [SubscriptionStatus.ACTIVE.value, SubscriptionStatus.TRIAL.value]:
        raise HTTPException(
            status_code=400,
            detail="Vous avez déjà un abonnement actif. Veuillez d'abord annuler votre abonnement actuel."
        )
    
    # Récupérer le plan
    plan = await get_plan_by_id(subscription_data.plan_id)
    if not plan:
        raise HTTPException(
            status_code=404,
            detail="Plan tarifaire non trouvé."
        )
    
    # Vérifier le coupon si fourni
    discount_percentage = 0
    coupon_code = None
    if subscription_data.coupon_code:
        coupon = await validate_coupon(subscription_data.coupon_code)
        if coupon:
            discount_percentage = coupon.discount_percentage
            coupon_code = coupon.code
        else:
            raise HTTPException(
                status_code=400,
                detail="Code promo invalide ou expiré."
            )
    
    # Calculer les dates d'abonnement
    now = datetime.now()
    start_date = now
    
    # Durée en fonction de la période de facturation
    duration = None
    if subscription_data.billing_period == BillingPeriod.MONTHLY:
        duration = timedelta(days=30)
    elif subscription_data.billing_period == BillingPeriod.QUARTERLY:
        duration = timedelta(days=90)
    elif subscription_data.billing_period == BillingPeriod.ANNUALLY:
        duration = timedelta(days=365)
    
    end_date = start_date + duration
    next_billing_date = end_date if subscription_data.auto_renew else None
    
    # Créer l'abonnement
    subscription_id = str(uuid.uuid4())
    subscription = await create_subscription(
        id=subscription_id,
        user_id=user.id,
        plan_id=plan.id,
        tier=plan.tier,
        status=SubscriptionStatus.PENDING.value,
        start_date=start_date,
        end_date=end_date,
        billing_period=subscription_data.billing_period.value,
        payment_method=subscription_data.payment_method.value,
        auto_renew=subscription_data.auto_renew,
        next_billing_date=next_billing_date
    )
    
    # Calculer le prix de l'abonnement
    price = calculate_subscription_price(
        plan=plan,
        billing_period=subscription_data.billing_period,
        discount_percentage=discount_percentage
    )
    
    # Créer la facture
    invoice_id = str(uuid.uuid4())
    invoice_number = f"INV-{now.strftime('%Y%m%d')}-{invoice_id[:8].upper()}"
    invoice = await create_invoice(
        id=invoice_id,
        user_id=user.id,
        subscription_id=subscription_id,
        amount=price,
        tax_amount=price * 0.2,  # TVA 20%
        total_amount=price * 1.2,
        due_date=now + timedelta(days=7),
        payment_method=subscription_data.payment_method.value,
        invoice_number=invoice_number,
        items=[{
            "description": f"Abonnement {plan.name}",
            "period": subscription_data.billing_period.value,
            "quantity": 1,
            "unit_price": price,
            "discount": discount_percentage,
            "coupon_code": coupon_code
        }]
    )
    
    # Traiter le paiement
    try:
        payment_result = await process_payment(
            invoice=invoice,
            payment_method=subscription_data.payment_method,
            user=user
        )
        
        # Enregistrer le paiement
        payment_id = str(uuid.uuid4())
        payment = await create_payment(
            id=payment_id,
            user_id=user.id,
            invoice_id=invoice_id,
            amount=invoice.total_amount,
            payment_method=subscription_data.payment_method.value,
            payment_details=payment_result.get("details", {}),
            status=payment_result.get("status", "pending"),
            transaction_id=payment_result.get("transaction_id")
        )
        
        # Si le paiement est réussi, activer l'abonnement
        if payment_result.get("status") == "succeeded":
            subscription = await update_subscription(
                subscription_id=subscription_id,
                status=SubscriptionStatus.ACTIVE.value
            )
            
            # Mettre à jour le statut de la facture
            await update_invoice_status(
                invoice_id=invoice_id,
                status="paid",
                paid_at=datetime.now()
            )
            
            # Mettre à jour le niveau d'abonnement de l'utilisateur
            user.subscription_tier = plan.tier
            # Code pour mettre à jour l'utilisateur
            
            # Envoyer un email de confirmation en arrière-plan
            background_tasks.add_task(
                send_subscription_confirmation_email,
                email=user.email,
                name=user.full_name,
                plan_name=plan.name,
                subscription_end=end_date,
                auto_renew=subscription_data.auto_renew
            )
            
            # Envoyer la facture par email
            background_tasks.add_task(
                send_invoice_email,
                email=user.email,
                name=user.full_name,
                invoice_number=invoice_number,
                invoice_date=now,
                total_amount=invoice.total_amount
            )
            
            logger.info(f"Nouvel abonnement activé pour {user.email}: {plan.name}")
            
        else:
            # Si le paiement est en attente
            logger.info(f"Paiement en attente pour {user.email}: {plan.name}")
            
    except Exception as e:
        logger.error(f"Erreur lors du traitement du paiement: {str(e)}")
        
        # En cas d'erreur, mettre à jour le statut de l'abonnement
        subscription = await update_subscription(
            subscription_id=subscription_id,
            status=SubscriptionStatus.PENDING.value
        )
        
        raise HTTPException(
            status_code=400,
            detail=f"Erreur lors du traitement du paiement: {str(e)}"
        )
    
    return subscription

@router.put("/my-subscription", response_model=SubscriptionResponse)
async def update_my_subscription(
    subscription_data: SubscriptionUpdate,
    request: Request
):
    """
    Met à jour l'abonnement de l'utilisateur actuel.
    
    Args:
        subscription_data: Les données de mise à jour
        request: La requête HTTP
        
    Returns:
        L'abonnement mis à jour
        
    Raises:
        HTTPException: Si l'utilisateur n'a pas d'abonnement actif
    """
    user = get_current_user(request)
    
    # Récupérer l'abonnement actuel
    subscription = await get_user_subscription(user.id)
    if not subscription:
        raise HTTPException(
            status_code=404,
            detail="Vous n'avez pas d'abonnement actif."
        )
    
    # Mise à jour de l'abonnement
    updated_subscription = await update_subscription(
        subscription_id=subscription.id,
        plan_id=subscription_data.plan_id,
        billing_period=subscription_data.billing_period.value if subscription_data.billing_period else None,
        payment_method=subscription_data.payment_method.value if subscription_data.payment_method else None,
        auto_renew=subscription_data.auto_renew,
        status=subscription_data.status.value if subscription_data.status else None
    )
    
    # Si le plan a changé, recalculer la prochaine date de facturation
    if subscription_data.plan_id and subscription_data.plan_id != subscription.plan_id:
        new_plan = await get_plan_by_id(subscription_data.plan_id)
        if new_plan:
            # Mettre à jour le niveau d'abonnement de l'utilisateur
            user.subscription_tier = new_plan.tier
            # Code pour mettre à jour l'utilisateur
    
    logger.info(f"Abonnement mis à jour pour {user.email}")
    
    return updated_subscription

@router.delete("/my-subscription", response_model=SubscriptionResponse)
async def cancel_my_subscription(request: Request):
    """
    Annule l'abonnement de l'utilisateur actuel.
    
    Args:
        request: La requête HTTP
        
    Returns:
        L'abonnement annulé
        
    Raises:
        HTTPException: Si l'utilisateur n'a pas d'abonnement actif
    """
    user = get_current_user(request)
    
    # Récupérer l'abonnement actuel
    subscription = await get_user_subscription(user.id)
    if not subscription:
        raise HTTPException(
            status_code=404,
            detail="Vous n'avez pas d'abonnement actif."
        )
    
    # Annuler l'abonnement
    canceled_subscription = await cancel_subscription(subscription.id)
    
    # Désactiver le renouvellement automatique
    canceled_subscription = await update_subscription(
        subscription_id=subscription.id,
        auto_renew=False,
        status=SubscriptionStatus.CANCELED.value
    )
    
    logger.info(f"Abonnement annulé pour {user.email}")
    
    return canceled_subscription

@router.get("/invoices", response_model=List[InvoiceResponse])
async def get_invoices(request: Request):
    """
    Récupère les factures de l'utilisateur actuel.
    
    Args:
        request: La requête HTTP
        
    Returns:
        Liste des factures
    """
    user = get_current_user(request)
    invoices = await get_user_invoices(user.id)
    return invoices

@router.get("/invoices/{invoice_id}", response_model=InvoiceResponse)
async def get_invoice(
    invoice_id: str,
    request: Request
):
    """
    Récupère une facture spécifique.
    
    Args:
        invoice_id: ID de la facture
        request: La requête HTTP
        
    Returns:
        La facture demandée
        
    Raises:
        HTTPException: Si la facture n'existe pas
    """
    user = get_current_user(request)
    
    # Récupérer la facture
    invoice = await get_invoice_by_id(invoice_id)
    
    if not invoice or invoice.user_id != user.id:
        raise HTTPException(
            status_code=404,
            detail="Facture non trouvée."
        )
    
    return invoice 