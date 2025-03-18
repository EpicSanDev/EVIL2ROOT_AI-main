"""
Routes de trading pour l'API du bot de trading.

Ce module gère les opérations de trading, les signaux et les positions.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks, Query
from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime

from src.api.models.trading import (
    OrderCreate, 
    OrderResponse, 
    PositionResponse, 
    SignalResponse,
    TradingBotStatus,
    OrderType,
    OrderSide,
    OrderStatus,
    TradingMode,
    PositionStatus,
    SignalType
)
from src.api.middleware.authentication import get_current_user
from src.core.trading import TradingSystem
from src.api.database.trading import (
    create_order,
    get_order_by_id,
    get_user_orders,
    update_order_status,
    get_user_positions,
    get_position_by_id,
    create_position,
    update_position,
    close_position,
    get_user_signals,
    get_bot_status
)
from src.api.services.trading_service import (
    execute_market_order,
    validate_order_request,
    calculate_position_pnl,
    update_positions_prices
)

# Configuration du logger
logger = logging.getLogger("api.trading")

# Configuration du router
router = APIRouter()

# Instance partagée du système de trading
trading_system = None

def get_trading_system() -> TradingSystem:
    """
    Récupère l'instance du système de trading.
    
    Returns:
        L'instance du système de trading
    """
    global trading_system
    if trading_system is None:
        # Créer une nouvelle instance en mode paper par défaut
        trading_system = TradingSystem(mode="paper")
    return trading_system

@router.get("/status", response_model=TradingBotStatus)
async def get_status(request: Request):
    """
    Obtient l'état actuel du bot de trading.
    
    Args:
        request: La requête HTTP
        
    Returns:
        L'état du bot de trading
    """
    user = get_current_user(request)
    
    # Récupérer l'état du bot depuis la base de données
    status = await get_bot_status(user.id)
    
    return status

@router.post("/start", response_model=TradingBotStatus)
async def start_bot(
    request: Request,
    mode: TradingMode,
    symbols: List[str],
    force_train: bool = False
):
    """
    Démarre le bot de trading.
    
    Args:
        request: La requête HTTP
        mode: Le mode de trading
        symbols: Les symboles à trader
        force_train: Forcer l'entraînement des modèles
        
    Returns:
        L'état du bot de trading
    """
    user = get_current_user(request)
    
    # Vérifier que l'utilisateur a accès au mode demandé
    if mode == TradingMode.LIVE and user.subscription_tier not in ["pro", "enterprise"]:
        raise HTTPException(
            status_code=403,
            detail="Votre abonnement ne permet pas d'utiliser le mode de trading réel."
        )
    
    # Récupérer le système de trading
    trading_sys = get_trading_system()
    
    # Démarrer le système de trading
    try:
        trading_sys.start(
            mode=mode.value,
            user_id=user.id,
            symbols=symbols,
            force_train=force_train
        )
        
        logger.info(f"Bot de trading démarré par {user.email} en mode {mode.value}")
        
        # Récupérer l'état mis à jour
        status = await get_bot_status(user.id)
        return status
        
    except Exception as e:
        logger.error(f"Erreur au démarrage du bot: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur au démarrage du bot de trading: {str(e)}"
        )

@router.post("/stop", response_model=TradingBotStatus)
async def stop_bot(request: Request):
    """
    Arrête le bot de trading.
    
    Args:
        request: La requête HTTP
        
    Returns:
        L'état du bot de trading
    """
    user = get_current_user(request)
    
    # Récupérer le système de trading
    trading_sys = get_trading_system()
    
    # Arrêter le système de trading
    try:
        trading_sys.stop(user_id=user.id)
        
        logger.info(f"Bot de trading arrêté par {user.email}")
        
        # Récupérer l'état mis à jour
        status = await get_bot_status(user.id)
        return status
        
    except Exception as e:
        logger.error(f"Erreur à l'arrêt du bot: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur à l'arrêt du bot de trading: {str(e)}"
        )

@router.get("/signals", response_model=List[SignalResponse])
async def get_signals(
    request: Request,
    symbol: Optional[str] = None,
    signal_type: Optional[SignalType] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """
    Récupère les signaux de trading.
    
    Args:
        request: La requête HTTP
        symbol: Filtrer par symbole
        signal_type: Filtrer par type de signal
        from_date: Date de début (YYYY-MM-DD)
        to_date: Date de fin (YYYY-MM-DD)
        limit: Nombre maximum de résultats
        
    Returns:
        Liste des signaux de trading
    """
    user = get_current_user(request)
    
    # Convertir les dates si fournies
    from_datetime = None
    to_datetime = None
    
    if from_date:
        try:
            from_datetime = datetime.strptime(from_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Le format de date de début doit être YYYY-MM-DD."
            )
            
    if to_date:
        try:
            to_datetime = datetime.strptime(to_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Le format de date de fin doit être YYYY-MM-DD."
            )
    
    # Récupérer les signaux
    signals = await get_user_signals(
        user_id=user.id,
        symbol=symbol,
        signal_type=signal_type.value if signal_type else None,
        from_date=from_datetime,
        to_date=to_datetime,
        limit=limit
    )
    
    return signals

@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(
    request: Request,
    symbol: Optional[str] = None,
    status: Optional[PositionStatus] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """
    Récupère les positions de trading.
    
    Args:
        request: La requête HTTP
        symbol: Filtrer par symbole
        status: Filtrer par statut
        limit: Nombre maximum de résultats
        
    Returns:
        Liste des positions de trading
    """
    user = get_current_user(request)
    
    # Récupérer les positions
    positions = await get_user_positions(
        user_id=user.id,
        symbol=symbol,
        status=status.value if status else None,
        limit=limit
    )
    
    # Mettre à jour les prix actuels et les PnL
    await update_positions_prices(positions)
    
    return positions

@router.get("/positions/{position_id}", response_model=PositionResponse)
async def get_position(
    position_id: str,
    request: Request
):
    """
    Récupère une position spécifique.
    
    Args:
        position_id: ID de la position
        request: La requête HTTP
        
    Returns:
        La position demandée
        
    Raises:
        HTTPException: Si la position n'existe pas
    """
    user = get_current_user(request)
    
    # Récupérer la position
    position = await get_position_by_id(position_id)
    
    if not position or position.user_id != user.id:
        raise HTTPException(
            status_code=404,
            detail="Position non trouvée."
        )
    
    # Mettre à jour le prix actuel et le PnL
    positions = [position]
    await update_positions_prices(positions)
    
    return position

@router.post("/positions/{position_id}/close", response_model=PositionResponse)
async def close_position_endpoint(
    position_id: str,
    request: Request
):
    """
    Ferme une position existante.
    
    Args:
        position_id: ID de la position
        request: La requête HTTP
        
    Returns:
        La position fermée
        
    Raises:
        HTTPException: Si la position n'existe pas ou est déjà fermée
    """
    user = get_current_user(request)
    
    # Récupérer la position
    position = await get_position_by_id(position_id)
    
    if not position or position.user_id != user.id:
        raise HTTPException(
            status_code=404,
            detail="Position non trouvée."
        )
    
    if position.status != PositionStatus.OPEN.value:
        raise HTTPException(
            status_code=400,
            detail="Cette position est déjà fermée."
        )
    
    # Récupérer le système de trading
    trading_sys = get_trading_system()
    
    # Fermer la position
    try:
        closed_position = await close_position(
            position_id=position_id,
            close_price=trading_sys.get_current_price(position.symbol),
            close_date=datetime.now(),
            reason="Fermeture manuelle"
        )
        
        logger.info(f"Position {position_id} fermée par {user.email}")
        
        return closed_position
        
    except Exception as e:
        logger.error(f"Erreur lors de la fermeture de la position: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la fermeture de la position: {str(e)}"
        )

@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    request: Request,
    symbol: Optional[str] = None,
    status: Optional[OrderStatus] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """
    Récupère les ordres de trading.
    
    Args:
        request: La requête HTTP
        symbol: Filtrer par symbole
        status: Filtrer par statut
        limit: Nombre maximum de résultats
        
    Returns:
        Liste des ordres de trading
    """
    user = get_current_user(request)
    
    # Récupérer les ordres
    orders = await get_user_orders(
        user_id=user.id,
        symbol=symbol,
        status=status.value if status else None,
        limit=limit
    )
    
    return orders

@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    request: Request
):
    """
    Récupère un ordre spécifique.
    
    Args:
        order_id: ID de l'ordre
        request: La requête HTTP
        
    Returns:
        L'ordre demandé
        
    Raises:
        HTTPException: Si l'ordre n'existe pas
    """
    user = get_current_user(request)
    
    # Récupérer l'ordre
    order = await get_order_by_id(order_id)
    
    if not order or order.user_id != user.id:
        raise HTTPException(
            status_code=404,
            detail="Ordre non trouvé."
        )
    
    return order

@router.post("/orders", response_model=OrderResponse)
async def create_order_endpoint(
    order_data: OrderCreate,
    request: Request
):
    """
    Crée un nouvel ordre de trading.
    
    Args:
        order_data: Les données de l'ordre
        request: La requête HTTP
        
    Returns:
        L'ordre créé
        
    Raises:
        HTTPException: Si les données de l'ordre sont invalides
    """
    user = get_current_user(request)
    
    # Remplacer l'ID utilisateur par celui de l'utilisateur authentifié
    order_data.user_id = user.id
    
    # Récupérer le système de trading
    trading_sys = get_trading_system()
    
    # Valider la demande d'ordre
    try:
        validate_order_request(order_data, user.subscription_tier, trading_sys)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    # Créer l'ordre
    try:
        order_id = str(uuid.uuid4())
        order = await create_order(
            id=order_id,
            symbol=order_data.symbol,
            order_type=order_data.order_type.value,
            side=order_data.side.value,
            quantity=order_data.quantity,
            price=order_data.price,
            stop_price=order_data.stop_price,
            time_in_force=order_data.time_in_force.value,
            status=OrderStatus.PENDING.value,
            trading_mode=order_data.trading_mode.value,
            user_id=user.id
        )
        
        # Si c'est un ordre au marché, l'exécuter immédiatement
        if order_data.order_type == OrderType.MARKET:
            filled_order, position = await execute_market_order(order, trading_sys)
            return filled_order
        
        # Sinon, retourner l'ordre en attente
        return order
        
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'ordre: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la création de l'ordre: {str(e)}"
        )

@router.delete("/orders/{order_id}", response_model=OrderResponse)
async def cancel_order(
    order_id: str,
    request: Request
):
    """
    Annule un ordre en attente.
    
    Args:
        order_id: ID de l'ordre
        request: La requête HTTP
        
    Returns:
        L'ordre annulé
        
    Raises:
        HTTPException: Si l'ordre n'existe pas ou ne peut pas être annulé
    """
    user = get_current_user(request)
    
    # Récupérer l'ordre
    order = await get_order_by_id(order_id)
    
    if not order or order.user_id != user.id:
        raise HTTPException(
            status_code=404,
            detail="Ordre non trouvé."
        )
    
    if order.status not in [OrderStatus.PENDING.value, OrderStatus.OPEN.value]:
        raise HTTPException(
            status_code=400,
            detail="Cet ordre ne peut pas être annulé."
        )
    
    # Annuler l'ordre
    try:
        canceled_order = await update_order_status(
            order_id=order_id,
            new_status=OrderStatus.CANCELED.value
        )
        
        logger.info(f"Ordre {order_id} annulé par {user.email}")
        
        return canceled_order
        
    except Exception as e:
        logger.error(f"Erreur lors de l'annulation de l'ordre: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'annulation de l'ordre: {str(e)}"
        ) 