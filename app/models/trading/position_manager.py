import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import uuid
import json

logger = logging.getLogger(__name__)

class Position:
    """Class representing a trading position with risk management capabilities."""
    
    def __init__(
        self,
        symbol: str,
        direction: str,  # 'long' or 'short'
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        entry_time: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ):
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.direction = direction.lower()
        self.entry_price = entry_price
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = entry_time or datetime.now()
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.status = "open"
        self.metadata = metadata or {}
        
        # Validate position
        if self.direction not in ['long', 'short']:
            raise ValueError("Direction must be either 'long' or 'short'")
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if self.size <= 0:
            raise ValueError("Position size must be positive")
        
        logger.info(f"Created new {direction} position for {symbol} with size {size} at price {entry_price}")
    
    def calculate_current_value(self, current_price: float) -> float:
        """Calculate the current value of the position."""
        return self.size * current_price
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate the unrealized profit/loss of the position."""
        if self.direction == 'long':
            return self.size * (current_price - self.entry_price)
        else:  # short
            return self.size * (self.entry_price - current_price)
    
    def calculate_unrealized_pnl_percentage(self, current_price: float) -> float:
        """Calculate the unrealized profit/loss percentage of the position."""
        if self.direction == 'long':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            return ((self.entry_price - current_price) / self.entry_price) * 100
    
    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if the position should be closed based on current price.
        Returns (should_close, reason)
        """
        if self.direction == 'long':
            # Check stop loss
            if self.stop_loss and current_price <= self.stop_loss:
                return True, "stop_loss"
            
            # Check take profit
            if self.take_profit and current_price >= self.take_profit:
                return True, "take_profit"
        
        else:  # short
            # Check stop loss
            if self.stop_loss and current_price >= self.stop_loss:
                return True, "stop_loss"
            
            # Check take profit
            if self.take_profit and current_price <= self.take_profit:
                return True, "take_profit"
        
        return False, None
    
    def close(self, exit_price: float, exit_time: Optional[datetime] = None) -> float:
        """Close the position and return the realized PnL."""
        if self.status != "open":
            raise ValueError(f"Cannot close position that is already {self.status}")
        
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        
        # Calculate realized PnL
        if self.direction == 'long':
            self.pnl = self.size * (self.exit_price - self.entry_price)
        else:  # short
            self.pnl = self.size * (self.entry_price - self.exit_price)
        
        self.status = "closed"
        
        logger.info(f"Closed {self.direction} position for {self.symbol} with PnL {self.pnl:.2f}")
        return self.pnl
    
    def update_stop_loss(self, new_stop_loss: float) -> None:
        """Update the stop loss level."""
        if self.status != "open":
            raise ValueError(f"Cannot update stop loss for position that is {self.status}")
        
        # Validate the new stop loss
        if self.direction == 'long' and new_stop_loss > self.entry_price:
            raise ValueError("Stop loss for long position must be below entry price")
        if self.direction == 'short' and new_stop_loss < self.entry_price:
            raise ValueError("Stop loss for short position must be above entry price")
        
        self.stop_loss = new_stop_loss
        logger.info(f"Updated stop loss for {self.symbol} to {new_stop_loss}")
    
    def update_take_profit(self, new_take_profit: float) -> None:
        """Update the take profit level."""
        if self.status != "open":
            raise ValueError(f"Cannot update take profit for position that is {self.status}")
        
        # Validate the new take profit
        if self.direction == 'long' and new_take_profit < self.entry_price:
            raise ValueError("Take profit for long position must be above entry price")
        if self.direction == 'short' and new_take_profit > self.entry_price:
            raise ValueError("Take profit for short position must be below entry price")
        
        self.take_profit = new_take_profit
        logger.info(f"Updated take profit for {self.symbol} to {new_take_profit}")
    
    def to_dict(self) -> Dict:
        """Convert the position to a dictionary."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'status': self.status,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """Create a position from a dictionary."""
        position = cls(
            symbol=data['symbol'],
            direction=data['direction'],
            entry_price=data['entry_price'],
            size=data['size'],
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            entry_time=datetime.fromisoformat(data['entry_time']) if data.get('entry_time') else None,
            metadata=data.get('metadata', {})
        )
        
        # Restore other fields
        position.id = data['id']
        position.status = data['status']
        
        if data.get('exit_time'):
            position.exit_time = datetime.fromisoformat(data['exit_time'])
        
        position.exit_price = data.get('exit_price')
        position.pnl = data.get('pnl', 0.0)
        
        return position


class PositionManager:
    """Manages a portfolio of trading positions with advanced risk management."""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}  # Map from position ID to Position
        self.closed_positions: List[Position] = []
        self.position_history: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        # Record initial equity point
        self._record_equity_point()
        
        logger.info(f"Position manager initialized with ${initial_balance} balance")
    
    def _record_equity_point(self, timestamp: Optional[datetime] = None) -> None:
        """Record a point in the equity curve."""
        timestamp = timestamp or datetime.now()
        
        # Calculate equity (balance + unrealized PnL)
        unrealized_pnl = sum(position.calculate_unrealized_pnl(position.entry_price) 
                            for position in self.positions.values())
        
        equity = self.current_balance + unrealized_pnl
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'balance': self.current_balance,
            'equity': equity,
            'unrealized_pnl': unrealized_pnl,
            'num_positions': len(self.positions)
        })
    
    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> Position:
        """Open a new position."""
        # Calculate position cost
        position_cost = entry_price * size
        
        # Check if we have enough balance
        if position_cost > self.current_balance:
            raise ValueError(f"Insufficient balance (${self.current_balance}) to open position costing ${position_cost}")
        
        # Create the position
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata
        )
        
        # Deduct the cost from the balance
        self.current_balance -= position_cost
        
        # Store the position
        self.positions[position.id] = position
        
        # Record the transaction
        self.position_history.append({
            'action': 'open',
            'position_id': position.id,
            'symbol': symbol,
            'direction': direction,
            'price': entry_price,
            'size': size,
            'timestamp': position.entry_time.isoformat(),
            'balance_after': self.current_balance
        })
        
        # Update equity curve
        self._record_equity_point(position.entry_time)
        
        logger.info(f"Opened {direction} position for {symbol} with size {size} at price {entry_price}")
        return position
    
    def close_position(self, position_id: str, exit_price: float) -> float:
        """Close an existing position."""
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")
        
        position = self.positions[position_id]
        
        # Close the position and get the PnL
        pnl = position.close(exit_price)
        
        # Add the position value plus PnL back to the balance
        position_value = position.size * exit_price
        self.current_balance += position_value
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[position_id]
        
        # Record the transaction
        self.position_history.append({
            'action': 'close',
            'position_id': position.id,
            'symbol': position.symbol,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': pnl,
            'timestamp': position.exit_time.isoformat(),
            'balance_after': self.current_balance
        })
        
        # Update equity curve
        self._record_equity_point(position.exit_time)
        
        logger.info(f"Closed {position.direction} position for {position.symbol} with PnL {pnl:.2f}")
        return pnl
    
    def update_position_stops(self, position_id: str, new_stop_loss: Optional[float] = None, 
                             new_take_profit: Optional[float] = None) -> None:
        """Update the stop loss and take profit levels for a position."""
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")
        
        position = self.positions[position_id]
        
        if new_stop_loss is not None:
            position.update_stop_loss(new_stop_loss)
        
        if new_take_profit is not None:
            position.update_take_profit(new_take_profit)
        
        logger.info(f"Updated stops for {position.symbol} position: SL={new_stop_loss}, TP={new_take_profit}")
    
    def check_stops(self, symbol: str, current_price: float) -> List[str]:
        """
        Check if any positions for the given symbol should be closed based on their stops.
        Returns a list of position IDs that were closed.
        """
        closed_position_ids = []
        
        for position_id, position in list(self.positions.items()):
            if position.symbol != symbol:
                continue
            
            should_close, reason = position.should_close(current_price)
            
            if should_close:
                self.close_position(position_id, current_price)
                closed_position_ids.append(position_id)
                logger.info(f"Automatically closed {position.symbol} position due to {reason} at price {current_price}")
        
        return closed_position_ids
    
    def get_position_exposure(self, symbol: Optional[str] = None) -> float:
        """Get the total exposure for a symbol or all symbols."""
        if symbol:
            return sum(p.calculate_current_value(p.entry_price) 
                      for p in self.positions.values() 
                      if p.symbol == symbol)
        else:
            return sum(p.calculate_current_value(p.entry_price) 
                      for p in self.positions.values())
    
    def get_total_unrealized_pnl(self, market_prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate the total unrealized PnL across all positions."""
        total_pnl = 0.0
        
        for position in self.positions.values():
            # Use provided market price if available, otherwise use entry price (no PnL)
            price = position.entry_price
            if market_prices and position.symbol in market_prices:
                price = market_prices[position.symbol]
            
            total_pnl += position.calculate_unrealized_pnl(price)
        
        return total_pnl
    
    def get_position_count(self, symbol: Optional[str] = None) -> int:
        """Get the number of open positions for a symbol or all symbols."""
        if symbol:
            return sum(1 for p in self.positions.values() if p.symbol == symbol)
        else:
            return len(self.positions)
    
    def get_portfolio_stats(self, market_prices: Optional[Dict[str, float]] = None) -> Dict:
        """Get comprehensive portfolio statistics."""
        stats = {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'symbols_held': len(set(p.symbol for p in self.positions.values())),
            'total_exposure': self.get_position_exposure(),
            'unrealized_pnl': self.get_total_unrealized_pnl(market_prices),
            'realized_pnl': sum(p.pnl for p in self.closed_positions),
            'total_pnl': 0.0,  # Will be calculated below
            'win_rate': 0.0,  # Will be calculated below
            'profit_factor': 0.0,  # Will be calculated below
            'positions_by_symbol': {},
            'exposure_by_symbol': {}
        }
        
        # Calculate total PnL
        stats['total_pnl'] = stats['unrealized_pnl'] + stats['realized_pnl']
        
        # Calculate win metrics
        if self.closed_positions:
            winning_trades = sum(1 for p in self.closed_positions if p.pnl > 0)
            total_trades = len(self.closed_positions)
            stats['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0.0
            
            gross_profit = sum(p.pnl for p in self.closed_positions if p.pnl > 0)
            gross_loss = abs(sum(p.pnl for p in self.closed_positions if p.pnl < 0))
            stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate positions and exposure by symbol
        for symbol in set(p.symbol for p in self.positions.values()):
            stats['positions_by_symbol'][symbol] = sum(1 for p in self.positions.values() if p.symbol == symbol)
            stats['exposure_by_symbol'][symbol] = self.get_position_exposure(symbol)
        
        return stats
    
    def save_to_file(self, filepath: str) -> None:
        """Save the portfolio state to a file."""
        data = {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'positions': [p.to_dict() for p in self.positions.values()],
            'closed_positions': [p.to_dict() for p in self.closed_positions],
            'position_history': self.position_history,
            'equity_curve': [{
                'timestamp': point['timestamp'].isoformat() if isinstance(point['timestamp'], datetime) else point['timestamp'],
                'balance': point['balance'],
                'equity': point['equity'],
                'unrealized_pnl': point['unrealized_pnl'],
                'num_positions': point['num_positions']
            } for point in self.equity_curve]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved portfolio state to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """Load the portfolio state from a file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.initial_balance = data['initial_balance']
        self.current_balance = data['current_balance']
        
        # Load positions
        self.positions = {}
        for pos_data in data['positions']:
            position = Position.from_dict(pos_data)
            self.positions[position.id] = position
        
        # Load closed positions
        self.closed_positions = [Position.from_dict(pos_data) for pos_data in data['closed_positions']]
        
        # Load history
        self.position_history = data['position_history']
        
        # Load equity curve
        self.equity_curve = [{
            'timestamp': datetime.fromisoformat(point['timestamp']) if isinstance(point['timestamp'], str) else point['timestamp'],
            'balance': point['balance'],
            'equity': point['equity'],
            'unrealized_pnl': point['unrealized_pnl'],
            'num_positions': point['num_positions']
        } for point in data['equity_curve']]
        
        logger.info(f"Loaded portfolio state from {filepath}") 