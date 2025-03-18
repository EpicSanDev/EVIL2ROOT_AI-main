"""
Modèles de données pour les opérations de trading.

Ce module définit les modèles Pydantic pour les transactions, stratégies et résultats de trading.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

class OrderType(str, Enum):
    """Types d'ordres supportés."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    """Côtés des ordres (achat/vente)."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    """Statuts possibles pour un ordre."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TimeInForce(str, Enum):
    """Durée de validité des ordres."""
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    DAY = "day"  # Day Order

class TradingMode(str, Enum):
    """Modes de trading disponibles."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    ANALYSIS = "analysis"

class PositionStatus(str, Enum):
    """Statuts possibles pour une position."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"

class SignalType(str, Enum):
    """Types de signaux de trading."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class OrderCreate(BaseModel):
    """Modèle pour la création d'un ordre."""
    
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None  # Requis pour les ordres limit et stop_limit
    stop_price: Optional[float] = None  # Requis pour les ordres stop et stop_limit
    time_in_force: TimeInForce = TimeInForce.GTC
    trading_mode: TradingMode
    user_id: str
    
    @validator('price')
    def validate_price(cls, v, values):
        """Valide que le prix est présent pour les ordres limit et stop_limit."""
        order_type = values.get('order_type')
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError("Le prix est requis pour les ordres limit et stop_limit")
        return v
    
    @validator('stop_price')
    def validate_stop_price(cls, v, values):
        """Valide que le prix stop est présent pour les ordres stop et stop_limit."""
        order_type = values.get('order_type')
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and v is None:
            raise ValueError("Le prix stop est requis pour les ordres stop et stop_limit")
        return v

class OrderResponse(BaseModel):
    """Modèle de réponse pour un ordre."""
    
    id: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce
    status: OrderStatus
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    trading_mode: TradingMode
    user_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class PositionResponse(BaseModel):
    """Modèle de réponse pour une position."""
    
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    status: PositionStatus
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    trading_mode: TradingMode
    user_id: str
    opened_at: datetime
    closed_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class SignalResponse(BaseModel):
    """Modèle de réponse pour un signal de trading."""
    
    id: str
    symbol: str
    signal_type: SignalType
    price: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    timeframe: str
    indicators: Dict[str, Any] = {}
    created_at: datetime
    validated: bool = False
    validation_confidence: float = 0.0
    
    class Config:
        orm_mode = True

class BacktestRequest(BaseModel):
    """Modèle de requête pour lancer un backtest."""
    
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    strategy_id: Optional[str] = None
    strategy_params: Optional[Dict[str, Any]] = None
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        """Valide le format de date (YYYY-MM-DD)."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("La date doit être au format YYYY-MM-DD")
    
    @validator('end_date')
    def validate_end_date(cls, v, values):
        """Valide que la date de fin est postérieure à la date de début."""
        if 'start_date' in values and v < values['start_date']:
            raise ValueError("La date de fin doit être postérieure à la date de début")
        return v

class BacktestResponse(BaseModel):
    """Modèle de réponse pour un backtest."""
    
    id: str
    user_id: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    strategy_id: Optional[str] = None
    strategy_params: Optional[Dict[str, Any]] = None
    trade_history: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        orm_mode = True

class TradingBotStatus(BaseModel):
    """Modèle pour l'état du bot de trading."""
    
    status: str
    mode: TradingMode
    active_symbols: List[str]
    running_since: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    total_positions: int
    open_positions: int
    today_pnl: float
    total_pnl: float
    cpu_usage: float
    memory_usage: float
    
    class Config:
        orm_mode = True

class StrategyBase(BaseModel):
    """Modèle de base pour une stratégie de trading."""
    
    name: str
    description: str
    parameters: Dict[str, Any] = {}
    
class StrategyCreate(StrategyBase):
    """Modèle pour la création d'une stratégie."""
    
    user_id: str

class StrategyResponse(StrategyBase):
    """Modèle de réponse pour une stratégie."""
    
    id: str
    user_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    performance: Optional[Dict[str, Any]] = None
    
    class Config:
        orm_mode = True 