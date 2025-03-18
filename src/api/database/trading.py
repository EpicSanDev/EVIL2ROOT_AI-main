"""
Opérations CRUD pour le trading.

Ce module fournit les fonctions d'accès à la base de données
pour créer, lire, mettre à jour et supprimer des entités liées au trading.
"""

from sqlalchemy import select, update, delete, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import uuid
import logging
from typing import List, Dict, Optional, Any, Tuple

from src.api.database.models.trading import (
    TradingStrategy, TradingAccount, Trade, Exchange, Symbol, BacktestResult,
    TradeStatusEnum, TradeTypeEnum, OrderTypeEnum, TradingStrategyTypeEnum, AccountTypeEnum
)
from src.api.database.session import get_db

logger = logging.getLogger("api.database.trading")


# --- ÉCHANGES ---

async def create_exchange(
    name: str,
    logo_url: str = None,
    website_url: str = None,
    description: str = None,
    is_active: bool = True,
    api_base_url: str = None,
    supports_websocket: bool = False,
    supported_currencies: List[str] = None,
    maker_fee: float = 0.0,
    taker_fee: float = 0.0,
    db: AsyncSession = None
) -> Exchange:
    """
    Crée une nouvelle plateforme d'échange dans la base de données.
    
    Args:
        name: Nom de l'échange
        logo_url: URL du logo
        website_url: URL du site web
        description: Description
        is_active: Statut d'activation
        api_base_url: URL de base de l'API
        supports_websocket: Indique si l'échange supporte les websockets
        supported_currencies: Liste des devises supportées
        maker_fee: Frais maker
        taker_fee: Frais taker
        db: Session de base de données
        
    Returns:
        Exchange: L'échange créé
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    exchange = Exchange(
        id=str(uuid.uuid4()),
        name=name,
        logo_url=logo_url,
        website_url=website_url,
        description=description,
        is_active=is_active,
        api_base_url=api_base_url,
        supports_websocket=supports_websocket,
        supported_currencies=supported_currencies or [],
        maker_fee=maker_fee,
        taker_fee=taker_fee
    )
    
    db.add(exchange)
    await db.commit()
    await db.refresh(exchange)
    
    logger.info(f"Échange créé: {name}")
    
    return exchange


async def get_exchange(
    exchange_id: str,
    db: AsyncSession = None
) -> Exchange:
    """
    Récupère un échange par son ID.
    
    Args:
        exchange_id: ID de l'échange
        db: Session de base de données
    
    Returns:
        Exchange ou None: L'échange trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Exchange).where(Exchange.id == exchange_id)
    result = await db.execute(query)
    exchange = result.scalars().first()
    
    return exchange


async def get_exchange_by_name(
    name: str,
    db: AsyncSession = None
) -> Exchange:
    """
    Récupère un échange par son nom.
    
    Args:
        name: Nom de l'échange
        db: Session de base de données
    
    Returns:
        Exchange ou None: L'échange trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Exchange).where(Exchange.name == name)
    result = await db.execute(query)
    exchange = result.scalars().first()
    
    return exchange


async def get_all_exchanges(
    active_only: bool = False,
    db: AsyncSession = None
) -> List[Exchange]:
    """
    Récupère tous les échanges.
    
    Args:
        active_only: Si True, récupère uniquement les échanges actifs
        db: Session de base de données
    
    Returns:
        List[Exchange]: Liste des échanges
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Exchange)
    if active_only:
        query = query.where(Exchange.is_active == True)
    
    result = await db.execute(query)
    exchanges = result.scalars().all()
    
    return list(exchanges)


# --- SYMBOLES ---

async def create_symbol(
    name: str,
    exchange_id: str,
    base_asset: str,
    quote_asset: str,
    min_amount: float = None,
    max_amount: float = None,
    price_precision: int = 8,
    amount_precision: int = 8,
    is_active: bool = True,
    db: AsyncSession = None
) -> Symbol:
    """
    Crée un nouveau symbole (paire de trading) dans la base de données.
    
    Args:
        name: Nom du symbole (ex: BTC/USD)
        exchange_id: ID de l'échange
        base_asset: Actif de base (ex: BTC)
        quote_asset: Actif de cotation (ex: USD)
        min_amount: Montant minimum
        max_amount: Montant maximum
        price_precision: Précision du prix
        amount_precision: Précision du montant
        is_active: Statut d'activation
        db: Session de base de données
        
    Returns:
        Symbol: Le symbole créé
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    symbol = Symbol(
        id=str(uuid.uuid4()),
        name=name,
        exchange_id=exchange_id,
        base_asset=base_asset,
        quote_asset=quote_asset,
        min_amount=min_amount,
        max_amount=max_amount,
        price_precision=price_precision,
        amount_precision=amount_precision,
        is_active=is_active
    )
    
    db.add(symbol)
    await db.commit()
    await db.refresh(symbol)
    
    logger.info(f"Symbole créé: {name} sur l'échange {exchange_id}")
    
    return symbol


async def get_symbol(
    symbol_id: str,
    db: AsyncSession = None
) -> Symbol:
    """
    Récupère un symbole par son ID.
    
    Args:
        symbol_id: ID du symbole
        db: Session de base de données
    
    Returns:
        Symbol ou None: Le symbole trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Symbol).where(Symbol.id == symbol_id)
    result = await db.execute(query)
    symbol = result.scalars().first()
    
    return symbol


async def get_symbol_by_name(
    name: str,
    exchange_id: str,
    db: AsyncSession = None
) -> Symbol:
    """
    Récupère un symbole par son nom et l'ID de l'échange.
    
    Args:
        name: Nom du symbole
        exchange_id: ID de l'échange
        db: Session de base de données
    
    Returns:
        Symbol ou None: Le symbole trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Symbol).where(Symbol.name == name, Symbol.exchange_id == exchange_id)
    result = await db.execute(query)
    symbol = result.scalars().first()
    
    return symbol


async def get_symbols_by_exchange(
    exchange_id: str,
    active_only: bool = False,
    db: AsyncSession = None
) -> List[Symbol]:
    """
    Récupère tous les symboles d'un échange.
    
    Args:
        exchange_id: ID de l'échange
        active_only: Si True, récupère uniquement les symboles actifs
        db: Session de base de données
    
    Returns:
        List[Symbol]: Liste des symboles
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Symbol).where(Symbol.exchange_id == exchange_id)
    if active_only:
        query = query.where(Symbol.is_active == True)
    
    result = await db.execute(query)
    symbols = result.scalars().all()
    
    return list(symbols)


# --- COMPTES DE TRADING ---

async def create_trading_account(
    user_id: str,
    exchange_id: str,
    name: str,
    account_type: AccountTypeEnum = AccountTypeEnum.SPOT,
    api_key: str = None,
    api_secret: str = None,
    balance: Dict[str, float] = None,
    is_active: bool = True,
    is_demo: bool = False,
    db: AsyncSession = None
) -> TradingAccount:
    """
    Crée un nouveau compte de trading dans la base de données.
    
    Args:
        user_id: ID de l'utilisateur
        exchange_id: ID de l'échange
        name: Nom du compte
        account_type: Type de compte
        api_key: Clé API
        api_secret: Secret API
        balance: Solde du compte
        is_active: Statut d'activation
        is_demo: Indique s'il s'agit d'un compte de démonstration
        db: Session de base de données
        
    Returns:
        TradingAccount: Le compte de trading créé
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    account = TradingAccount(
        id=str(uuid.uuid4()),
        user_id=user_id,
        exchange_id=exchange_id,
        name=name,
        account_type=account_type,
        api_key=api_key,
        api_secret=api_secret,
        balance=balance or {},
        is_active=is_active,
        is_demo=is_demo
    )
    
    db.add(account)
    await db.commit()
    await db.refresh(account)
    
    logger.info(f"Compte de trading créé: {name} pour l'utilisateur {user_id}")
    
    return account


async def get_trading_account(
    account_id: str,
    db: AsyncSession = None
) -> TradingAccount:
    """
    Récupère un compte de trading par son ID.
    
    Args:
        account_id: ID du compte
        db: Session de base de données
    
    Returns:
        TradingAccount ou None: Le compte trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(TradingAccount).where(
        TradingAccount.id == account_id,
        TradingAccount.is_deleted == False
    )
    result = await db.execute(query)
    account = result.scalars().first()
    
    return account


async def get_user_trading_accounts(
    user_id: str,
    active_only: bool = False,
    db: AsyncSession = None
) -> List[TradingAccount]:
    """
    Récupère tous les comptes de trading d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        active_only: Si True, récupère uniquement les comptes actifs
        db: Session de base de données
    
    Returns:
        List[TradingAccount]: Liste des comptes de trading
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(TradingAccount).where(
        TradingAccount.user_id == user_id,
        TradingAccount.is_deleted == False
    )
    if active_only:
        query = query.where(TradingAccount.is_active == True)
    
    result = await db.execute(query)
    accounts = result.scalars().all()
    
    return list(accounts)


async def update_trading_account_balance(
    account_id: str,
    balance: Dict[str, float],
    db: AsyncSession = None
) -> TradingAccount:
    """
    Met à jour le solde d'un compte de trading.
    
    Args:
        account_id: ID du compte
        balance: Nouveau solde
        db: Session de base de données
    
    Returns:
        TradingAccount: Le compte mis à jour
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    account = await get_trading_account(account_id, db)
    if not account:
        return None
    
    account.balance = balance
    account.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(account)
    
    logger.info(f"Solde mis à jour pour le compte {account_id}")
    
    return account


# --- STRATÉGIES DE TRADING ---

async def create_trading_strategy(
    user_id: str,
    name: str,
    description: str = None,
    strategy_type: TradingStrategyTypeEnum = TradingStrategyTypeEnum.MANUAL,
    parameters: Dict[str, Any] = None,
    risk_level: int = 5,
    code: str = None,
    is_backtested: bool = False,
    is_active: bool = True,
    account_ids: List[str] = None,
    db: AsyncSession = None
) -> TradingStrategy:
    """
    Crée une nouvelle stratégie de trading dans la base de données.
    
    Args:
        user_id: ID de l'utilisateur
        name: Nom de la stratégie
        description: Description
        strategy_type: Type de stratégie
        parameters: Paramètres de la stratégie
        risk_level: Niveau de risque (1-10)
        code: Code de la stratégie (pour les stratégies algorithmiques)
        is_backtested: Indique si la stratégie a été backtestée
        is_active: Statut d'activation
        account_ids: Liste des IDs de comptes associés
        db: Session de base de données
        
    Returns:
        TradingStrategy: La stratégie créée
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    strategy = TradingStrategy(
        id=str(uuid.uuid4()),
        user_id=user_id,
        name=name,
        description=description,
        strategy_type=strategy_type,
        parameters=parameters or {},
        risk_level=risk_level,
        code=code,
        is_backtested=is_backtested,
        is_active=is_active
    )
    
    db.add(strategy)
    await db.commit()
    await db.refresh(strategy)
    
    # Associer des comptes si spécifiés
    if account_ids:
        accounts = []
        for account_id in account_ids:
            account = await get_trading_account(account_id, db)
            if account:
                accounts.append(account)
        
        strategy.accounts = accounts
        await db.commit()
        await db.refresh(strategy)
    
    logger.info(f"Stratégie de trading créée: {name} pour l'utilisateur {user_id}")
    
    return strategy


async def get_trading_strategy(
    strategy_id: str,
    db: AsyncSession = None
) -> TradingStrategy:
    """
    Récupère une stratégie de trading par son ID.
    
    Args:
        strategy_id: ID de la stratégie
        db: Session de base de données
    
    Returns:
        TradingStrategy ou None: La stratégie trouvée ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(TradingStrategy).where(TradingStrategy.id == strategy_id)
    result = await db.execute(query)
    strategy = result.scalars().first()
    
    return strategy


async def get_user_trading_strategies(
    user_id: str,
    active_only: bool = False,
    db: AsyncSession = None
) -> List[TradingStrategy]:
    """
    Récupère toutes les stratégies de trading d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        active_only: Si True, récupère uniquement les stratégies actives
        db: Session de base de données
    
    Returns:
        List[TradingStrategy]: Liste des stratégies de trading
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(TradingStrategy).where(TradingStrategy.user_id == user_id)
    if active_only:
        query = query.where(TradingStrategy.is_active == True)
    
    result = await db.execute(query)
    strategies = result.scalars().all()
    
    return list(strategies)


# --- TRADES ---

async def create_trade(
    trading_account_id: str,
    symbol_id: str,
    trade_type: TradeTypeEnum,
    order_type: OrderTypeEnum,
    amount: float,
    price: float = None,
    stop_loss: float = None,
    take_profit: float = None,
    strategy_id: str = None,
    exchange_order_id: str = None,
    notes: str = None,
    metadata: Dict[str, Any] = None,
    status: TradeStatusEnum = TradeStatusEnum.PENDING,
    db: AsyncSession = None
) -> Trade:
    """
    Crée un nouveau trade dans la base de données.
    
    Args:
        trading_account_id: ID du compte de trading
        symbol_id: ID du symbole
        trade_type: Type de trade
        order_type: Type d'ordre
        amount: Montant
        price: Prix
        stop_loss: Stop loss
        take_profit: Take profit
        strategy_id: ID de la stratégie
        exchange_order_id: ID de l'ordre sur l'échange
        notes: Notes
        metadata: Métadonnées
        status: Statut du trade
        db: Session de base de données
        
    Returns:
        Trade: Le trade créé
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    trade = Trade(
        id=str(uuid.uuid4()),
        trading_account_id=trading_account_id,
        symbol_id=symbol_id,
        strategy_id=strategy_id,
        trade_type=trade_type,
        order_type=order_type,
        amount=amount,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        exchange_order_id=exchange_order_id,
        notes=notes,
        metadata=metadata,
        status=status,
        opened_at=datetime.utcnow()
    )
    
    db.add(trade)
    await db.commit()
    await db.refresh(trade)
    
    logger.info(f"Trade créé: {trade.id} pour le compte {trading_account_id}")
    
    return trade


async def get_trade(
    trade_id: str,
    db: AsyncSession = None
) -> Trade:
    """
    Récupère un trade par son ID.
    
    Args:
        trade_id: ID du trade
        db: Session de base de données
    
    Returns:
        Trade ou None: Le trade trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Trade).where(Trade.id == trade_id)
    result = await db.execute(query)
    trade = result.scalars().first()
    
    return trade


async def update_trade_status(
    trade_id: str,
    status: TradeStatusEnum,
    executed_price: float = None,
    profit_loss: float = None,
    profit_loss_percentage: float = None,
    fees: float = None,
    closed_at: datetime = None,
    db: AsyncSession = None
) -> Trade:
    """
    Met à jour le statut d'un trade.
    
    Args:
        trade_id: ID du trade
        status: Nouveau statut
        executed_price: Prix d'exécution
        profit_loss: Profit/perte
        profit_loss_percentage: Profit/perte en pourcentage
        fees: Frais
        closed_at: Date de clôture
        db: Session de base de données
    
    Returns:
        Trade: Le trade mis à jour
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    trade = await get_trade(trade_id, db)
    if not trade:
        return None
    
    trade.status = status
    
    if executed_price is not None:
        trade.executed_price = executed_price
    
    if profit_loss is not None:
        trade.profit_loss = profit_loss
    
    if profit_loss_percentage is not None:
        trade.profit_loss_percentage = profit_loss_percentage
    
    if fees is not None:
        trade.fees = fees
    
    if status == TradeStatusEnum.CLOSED and closed_at is None:
        trade.closed_at = datetime.utcnow()
    elif closed_at is not None:
        trade.closed_at = closed_at
    
    trade.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(trade)
    
    logger.info(f"Statut du trade {trade_id} mis à jour: {status}")
    
    return trade


async def get_account_trades(
    account_id: str,
    status: TradeStatusEnum = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = None
) -> List[Trade]:
    """
    Récupère les trades d'un compte.
    
    Args:
        account_id: ID du compte
        status: Filtrer par statut
        limit: Nombre maximum de résultats
        offset: Décalage pour la pagination
        db: Session de base de données
    
    Returns:
        List[Trade]: Liste des trades
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(Trade).where(Trade.trading_account_id == account_id)
    
    if status:
        query = query.where(Trade.status == status)
    
    query = query.order_by(desc(Trade.opened_at))
    query = query.limit(limit).offset(offset)
    
    result = await db.execute(query)
    trades = result.scalars().all()
    
    return list(trades)


# --- BACKTEST ---

async def create_backtest_result(
    strategy_id: str,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    initial_balance: float,
    final_balance: float,
    profit_loss: float,
    profit_loss_percentage: float,
    max_drawdown: float = None,
    win_rate: float = None,
    sharpe_ratio: float = None,
    total_trades: int = 0,
    winning_trades: int = 0,
    losing_trades: int = 0,
    trades_data: List[Dict[str, Any]] = None,
    equity_curve: List[Dict[str, Any]] = None,
    db: AsyncSession = None
) -> BacktestResult:
    """
    Crée un nouveau résultat de backtest dans la base de données.
    
    Args:
        strategy_id: ID de la stratégie
        symbol: Symbole testé
        timeframe: Timeframe utilisé
        start_date: Date de début
        end_date: Date de fin
        initial_balance: Solde initial
        final_balance: Solde final
        profit_loss: Profit/perte
        profit_loss_percentage: Profit/perte en pourcentage
        max_drawdown: Drawdown maximum
        win_rate: Taux de réussite
        sharpe_ratio: Ratio de Sharpe
        total_trades: Nombre total de trades
        winning_trades: Nombre de trades gagnants
        losing_trades: Nombre de trades perdants
        trades_data: Données détaillées des trades
        equity_curve: Points de la courbe d'équité
        db: Session de base de données
        
    Returns:
        BacktestResult: Le résultat de backtest créé
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    result = BacktestResult(
        id=str(uuid.uuid4()),
        strategy_id=strategy_id,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        final_balance=final_balance,
        profit_loss=profit_loss,
        profit_loss_percentage=profit_loss_percentage,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        sharpe_ratio=sharpe_ratio,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        trades_data=trades_data,
        equity_curve=equity_curve
    )
    
    db.add(result)
    await db.commit()
    await db.refresh(result)
    
    # Mettre à jour le statut de backtest de la stratégie
    strategy = await get_trading_strategy(strategy_id, db)
    if strategy:
        strategy.is_backtested = True
        await db.commit()
    
    logger.info(f"Résultat de backtest créé pour la stratégie {strategy_id}")
    
    return result


async def get_strategy_backtest_results(
    strategy_id: str,
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = None
) -> List[BacktestResult]:
    """
    Récupère les résultats de backtest d'une stratégie.
    
    Args:
        strategy_id: ID de la stratégie
        limit: Nombre maximum de résultats
        offset: Décalage pour la pagination
        db: Session de base de données
    
    Returns:
        List[BacktestResult]: Liste des résultats de backtest
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(BacktestResult).where(BacktestResult.strategy_id == strategy_id)
    query = query.order_by(desc(BacktestResult.created_at))
    query = query.limit(limit).offset(offset)
    
    result = await db.execute(query)
    backtest_results = result.scalars().all()
    
    return list(backtest_results) 