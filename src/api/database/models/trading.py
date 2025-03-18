"""
Modèles SQLAlchemy pour le trading.

Ce module définit les modèles de base de données pour les comptes de trading,
les stratégies, les trades, les échanges et les résultats de backtest.
"""

from datetime import datetime
from decimal import Decimal
from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey, 
    Integer, Numeric, JSON, Text, Enum, Float, Table
)
from sqlalchemy.orm import relationship
import enum

from src.api.database.models.base import BaseModel, SoftDeleteBaseModel


class TradeStatusEnum(str, enum.Enum):
    """Énumération des statuts de trade possibles."""
    
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    PENDING = "pending"
    PARTIAL = "partial"


class TradeTypeEnum(str, enum.Enum):
    """Énumération des types de trade."""
    
    BUY = "buy"
    SELL = "sell"
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderTypeEnum(str, enum.Enum):
    """Énumération des types d'ordre."""
    
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class TradingStrategyTypeEnum(str, enum.Enum):
    """Énumération des types de stratégie de trading."""
    
    MANUAL = "manual"
    ALGORITHMIC = "algorithmic"
    COPY_TRADING = "copy_trading"
    HYBRID = "hybrid"


class AccountTypeEnum(str, enum.Enum):
    """Énumération des types de compte."""
    
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"
    DEMO = "demo"


# Table d'association entre stratégies et comptes
strategy_account_association = Table(
    'strategy_account_association',
    BaseModel.metadata,
    Column('strategy_id', String(36), ForeignKey('tradingstrategys.id', ondelete="CASCADE")),
    Column('account_id', String(36), ForeignKey('tradingaccounts.id', ondelete="CASCADE")),
)


class Exchange(BaseModel):
    """Modèle SQLAlchemy pour les plateformes d'échange."""
    
    # Informations de base
    name = Column(String(100), nullable=False, unique=True)
    logo_url = Column(String(255), nullable=True)
    website_url = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    # Statut
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Configuration
    api_base_url = Column(String(255), nullable=True)
    supports_websocket = Column(Boolean, default=False, nullable=False)
    supported_currencies = Column(JSON, default=list, nullable=True)
    maker_fee = Column(Float, default=0.0, nullable=False)
    taker_fee = Column(Float, default=0.0, nullable=False)
    
    # Relations
    symbols = relationship("Symbol", back_populates="exchange", cascade="all, delete-orphan")
    trading_accounts = relationship("TradingAccount", back_populates="exchange")
    
    def __repr__(self):
        return f"<Exchange(id={self.id}, name={self.name})>"
    
    def to_dict(self):
        """Convertit l'échange en dictionnaire."""
        return {
            "id": self.id,
            "name": self.name,
            "logo_url": self.logo_url,
            "website_url": self.website_url,
            "description": self.description,
            "is_active": self.is_active,
            "api_base_url": self.api_base_url,
            "supports_websocket": self.supports_websocket,
            "supported_currencies": self.supported_currencies,
            "maker_fee": self.maker_fee,
            "taker_fee": self.taker_fee,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class Symbol(BaseModel):
    """Modèle SQLAlchemy pour les paires de trading."""
    
    # Informations de base
    name = Column(String(50), nullable=False, index=True)
    base_asset = Column(String(20), nullable=False)
    quote_asset = Column(String(20), nullable=False)
    
    # Relation avec l'échange
    exchange_id = Column(String(36), ForeignKey("exchanges.id", ondelete="CASCADE"), nullable=False, index=True)
    exchange = relationship("Exchange", back_populates="symbols")
    
    # Propriétés de trading
    min_amount = Column(Float, nullable=True)
    max_amount = Column(Float, nullable=True)
    price_precision = Column(Integer, default=8, nullable=False)
    amount_precision = Column(Integer, default=8, nullable=False)
    
    # Statut
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relations
    trades = relationship("Trade", back_populates="symbol")
    
    __table_args__ = (
        # Contrainte d'unicité pour le nom sur un échange donné
        # (un même symbole peut exister sur différents échanges)
        {"UniqueConstraint": ("name", "exchange_id")}
    )
    
    def __repr__(self):
        return f"<Symbol(id={self.id}, name={self.name}, exchange={self.exchange.name if self.exchange else None})>"
    
    def to_dict(self):
        """Convertit le symbole en dictionnaire."""
        return {
            "id": self.id,
            "name": self.name,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "exchange_id": self.exchange_id,
            "exchange_name": self.exchange.name if self.exchange else None,
            "min_amount": self.min_amount,
            "max_amount": self.max_amount,
            "price_precision": self.price_precision,
            "amount_precision": self.amount_precision,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class TradingAccount(SoftDeleteBaseModel):
    """Modèle SQLAlchemy pour les comptes de trading."""
    
    # Relation avec l'utilisateur
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    user = relationship("User", back_populates="trading_accounts")
    
    # Relation avec l'échange
    exchange_id = Column(String(36), ForeignKey("exchanges.id"), nullable=False, index=True)
    exchange = relationship("Exchange", back_populates="trading_accounts")
    
    # Informations du compte
    name = Column(String(100), nullable=False)
    account_type = Column(Enum(AccountTypeEnum), default=AccountTypeEnum.SPOT, nullable=False)
    api_key = Column(String(255), nullable=True)
    api_secret = Column(String(255), nullable=True)
    
    # Solde et allocation
    balance = Column(JSON, default=dict, nullable=True)  # Structure: {"BTC": 0.5, "ETH": 2.0, ...}
    
    # Statut
    is_active = Column(Boolean, default=True, nullable=False)
    is_demo = Column(Boolean, default=False, nullable=False)
    
    # Relations
    trades = relationship("Trade", back_populates="trading_account", cascade="all, delete-orphan")
    strategies = relationship(
        "TradingStrategy",
        secondary=strategy_account_association,
        back_populates="accounts"
    )
    
    def __repr__(self):
        return f"<TradingAccount(id={self.id}, user_id={self.user_id}, exchange={self.exchange.name if self.exchange else None})>"
    
    def to_dict(self):
        """Convertit le compte de trading en dictionnaire."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "exchange_id": self.exchange_id,
            "exchange_name": self.exchange.name if self.exchange else None,
            "name": self.name,
            "account_type": self.account_type,
            "balance": self.balance,
            "is_active": self.is_active,
            "is_demo": self.is_demo,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class TradingStrategy(BaseModel):
    """Modèle SQLAlchemy pour les stratégies de trading."""
    
    # Relation avec l'utilisateur
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    user = relationship("User", back_populates="strategies")
    
    # Informations de base
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    strategy_type = Column(Enum(TradingStrategyTypeEnum), default=TradingStrategyTypeEnum.MANUAL, nullable=False)
    
    # Configuration
    parameters = Column(JSON, default=dict, nullable=True)
    risk_level = Column(Integer, default=5, nullable=False)  # 1-10
    
    # Pour les stratégies algorithmiques
    code = Column(Text, nullable=True)
    is_backtested = Column(Boolean, default=False, nullable=False)
    
    # Statut
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relations
    accounts = relationship(
        "TradingAccount",
        secondary=strategy_account_association,
        back_populates="strategies"
    )
    trades = relationship("Trade", back_populates="strategy")
    backtest_results = relationship("BacktestResult", back_populates="strategy", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<TradingStrategy(id={self.id}, name={self.name}, user_id={self.user_id})>"
    
    def to_dict(self):
        """Convertit la stratégie de trading en dictionnaire."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type,
            "parameters": self.parameters,
            "risk_level": self.risk_level,
            "code": self.code,
            "is_backtested": self.is_backtested,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class Trade(BaseModel):
    """Modèle SQLAlchemy pour les trades."""
    
    # Relations
    trading_account_id = Column(String(36), ForeignKey("tradingaccounts.id", ondelete="CASCADE"), nullable=False, index=True)
    trading_account = relationship("TradingAccount", back_populates="trades")
    
    strategy_id = Column(String(36), ForeignKey("tradingstrategys.id"), nullable=True, index=True)
    strategy = relationship("TradingStrategy", back_populates="trades")
    
    symbol_id = Column(String(36), ForeignKey("symbols.id"), nullable=False, index=True)
    symbol = relationship("Symbol", back_populates="trades")
    
    # Informations de base
    trade_type = Column(Enum(TradeTypeEnum), nullable=False)
    order_type = Column(Enum(OrderTypeEnum), nullable=False)
    status = Column(Enum(TradeStatusEnum), default=TradeStatusEnum.PENDING, nullable=False)
    
    # Valeurs de l'ordre
    amount = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=True)  # Peut être null pour les ordres au marché
    executed_price = Column(Numeric(20, 8), nullable=True)
    
    # Stop-loss et take-profit
    stop_loss = Column(Numeric(20, 8), nullable=True)
    take_profit = Column(Numeric(20, 8), nullable=True)
    
    # Identifiants externes
    exchange_order_id = Column(String(100), nullable=True, unique=True)
    
    # Dates importantes
    opened_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    closed_at = Column(DateTime, nullable=True)
    
    # Calculs de performance
    profit_loss = Column(Numeric(20, 8), nullable=True)
    profit_loss_percentage = Column(Float, nullable=True)
    fees = Column(Numeric(20, 8), default=0, nullable=False)
    
    # Notes et métadonnées
    notes = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<Trade(id={self.id}, type={self.trade_type}, status={self.status})>"
    
    def to_dict(self):
        """Convertit le trade en dictionnaire."""
        return {
            "id": self.id,
            "trading_account_id": self.trading_account_id,
            "strategy_id": self.strategy_id,
            "symbol_id": self.symbol_id,
            "symbol_name": self.symbol.name if self.symbol else None,
            "trade_type": self.trade_type,
            "order_type": self.order_type,
            "status": self.status,
            "amount": float(self.amount),
            "price": float(self.price) if self.price else None,
            "executed_price": float(self.executed_price) if self.executed_price else None,
            "stop_loss": float(self.stop_loss) if self.stop_loss else None,
            "take_profit": float(self.take_profit) if self.take_profit else None,
            "exchange_order_id": self.exchange_order_id,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "profit_loss": float(self.profit_loss) if self.profit_loss else None,
            "profit_loss_percentage": self.profit_loss_percentage,
            "fees": float(self.fees),
            "notes": self.notes,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class BacktestResult(BaseModel):
    """Modèle SQLAlchemy pour les résultats de backtest des stratégies."""
    
    # Relation avec la stratégie
    strategy_id = Column(String(36), ForeignKey("tradingstrategys.id", ondelete="CASCADE"), nullable=False, index=True)
    strategy = relationship("TradingStrategy", back_populates="backtest_results")
    
    # Paramètres du backtest
    symbol = Column(String(50), nullable=False)
    timeframe = Column(String(20), nullable=False)  # 1m, 5m, 15m, 1h, 4h, 1d, etc.
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_balance = Column(Numeric(20, 8), nullable=False)
    
    # Résultats
    final_balance = Column(Numeric(20, 8), nullable=False)
    profit_loss = Column(Numeric(20, 8), nullable=False)
    profit_loss_percentage = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    
    # Statistiques détaillées
    total_trades = Column(Integer, default=0, nullable=False)
    winning_trades = Column(Integer, default=0, nullable=False)
    losing_trades = Column(Integer, default=0, nullable=False)
    
    # Données détaillées
    trades_data = Column(JSON, nullable=True)  # Liste détaillée des trades simulés
    equity_curve = Column(JSON, nullable=True)  # Points de l'equity curve pour graphique
    
    def __repr__(self):
        return f"<BacktestResult(id={self.id}, strategy_id={self.strategy_id}, profit_loss_percentage={self.profit_loss_percentage})>"
    
    def to_dict(self):
        """Convertit le résultat de backtest en dictionnaire."""
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy.name if self.strategy else None,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_balance": float(self.initial_balance),
            "final_balance": float(self.final_balance),
            "profit_loss": float(self.profit_loss),
            "profit_loss_percentage": self.profit_loss_percentage,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        } 