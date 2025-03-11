import pytest
import datetime
from app.models.position_manager import PositionManager, Position

class TestPosition:
    """Test suite for the Position class"""
    
    def test_position_initialization(self):
        """Test that a position is correctly initialized"""
        position = Position(
            symbol="AAPL",
            direction="long",
            entry_price=150.0,
            size=10.0,
            stop_loss=145.0,
            take_profit=160.0
        )
        
        assert position.symbol == "AAPL"
        assert position.direction == "long"
        assert position.entry_price == 150.0
        assert position.size == 10.0
        assert position.stop_loss == 145.0
        assert position.take_profit == 160.0
        assert position.status == "open"
        assert position.entry_time is not None
        assert position.exit_time is None
        assert position.exit_price is None
        assert position.realized_pnl is None
        assert position.id is not None
    
    def test_position_pnl_calculation(self):
        """Test PnL calculations for long and short positions"""
        long_position = Position(
            symbol="AAPL",
            direction="long",
            entry_price=150.0,
            size=10.0
        )
        
        short_position = Position(
            symbol="MSFT",
            direction="short",
            entry_price=300.0,
            size=5.0
        )
        
        # Test long position PnL
        assert long_position.calculate_unrealized_pnl(155.0) == 50.0  # (155-150) * 10
        assert long_position.calculate_unrealized_pnl(145.0) == -50.0  # (145-150) * 10
        
        # Test short position PnL
        assert short_position.calculate_unrealized_pnl(290.0) == 50.0  # (300-290) * 5
        assert short_position.calculate_unrealized_pnl(310.0) == -50.0  # (300-310) * 5
    
    def test_should_close_condition(self):
        """Test stop loss and take profit conditions"""
        position = Position(
            symbol="AAPL",
            direction="long",
            entry_price=150.0,
            size=10.0,
            stop_loss=145.0,
            take_profit=160.0
        )
        
        # Test normal price (no close)
        should_close, reason = position.should_close(152.0)
        assert not should_close
        assert reason == ""
        
        # Test stop loss
        should_close, reason = position.should_close(144.0)
        assert should_close
        assert reason == "Stop Loss"
        
        # Test take profit
        should_close, reason = position.should_close(161.0)
        assert should_close
        assert reason == "Take Profit"

    def test_position_close(self):
        """Test closing a position"""
        position = Position(
            symbol="AAPL",
            direction="long",
            entry_price=150.0,
            size=10.0
        )
        
        # Close the position
        pnl = position.close(155.0)
        
        assert pnl == 50.0  # (155-150) * 10
        assert position.status == "closed"
        assert position.exit_price == 155.0
        assert position.exit_time is not None
        assert position.realized_pnl == 50.0


class TestPositionManager:
    """Test suite for the PositionManager class"""
    
    def test_position_manager_initialization(self):
        """Test that a position manager is correctly initialized"""
        pm = PositionManager(initial_balance=10000.0)
        
        assert pm.initial_balance == 10000.0
        assert pm.current_balance == 10000.0
        assert len(pm.positions) == 0
        assert len(pm.closed_positions) == 0
        assert len(pm.equity_curve) == 1  # Initial equity point
    
    def test_open_position(self):
        """Test opening a position"""
        pm = PositionManager(initial_balance=10000.0)
        
        position = pm.open_position(
            symbol="AAPL",
            direction="long",
            entry_price=150.0,
            size=10.0,
            stop_loss=145.0,
            take_profit=160.0
        )
        
        assert len(pm.positions) == 1
        assert position.id in pm.positions
        assert pm.current_balance == 8500.0  # 10000 - (150 * 10)
        assert len(pm.position_history) == 1
        assert len(pm.equity_curve) == 2  # Initial + after position
    
    def test_close_position(self):
        """Test closing a position"""
        pm = PositionManager(initial_balance=10000.0)
        
        position = pm.open_position(
            symbol="AAPL",
            direction="long",
            entry_price=150.0,
            size=10.0
        )
        
        # Close with profit
        pnl = pm.close_position(position.id, 160.0)
        
        assert pnl == 100.0  # (160-150) * 10
        assert len(pm.positions) == 0
        assert len(pm.closed_positions) == 1
        assert pm.current_balance == 10100.0  # 10000 - (150*10) + (160*10)
        assert len(pm.position_history) == 2  # Open + Close
    
    def test_update_position_stops(self):
        """Test updating stop loss and take profit levels"""
        pm = PositionManager(initial_balance=10000.0)
        
        position = pm.open_position(
            symbol="AAPL",
            direction="long",
            entry_price=150.0,
            size=10.0,
            stop_loss=145.0,
            take_profit=160.0
        )
        
        # Update stops
        pm.update_position_stops(position.id, new_stop_loss=147.0, new_take_profit=165.0)
        
        updated_position = pm.positions[position.id]
        assert updated_position.stop_loss == 147.0
        assert updated_position.take_profit == 165.0
    
    def test_check_stops(self):
        """Test checking stop conditions"""
        pm = PositionManager(initial_balance=10000.0)
        
        position = pm.open_position(
            symbol="AAPL",
            direction="long",
            entry_price=150.0,
            size=10.0,
            stop_loss=145.0,
            take_profit=160.0
        )
        
        # Test with price within range (no close)
        closed_positions = pm.check_stops("AAPL", 152.0)
        assert len(closed_positions) == 0
        assert len(pm.positions) == 1
        
        # Test with stop loss hit
        closed_positions = pm.check_stops("AAPL", 144.0)
        assert len(closed_positions) == 1
        assert len(pm.positions) == 0
        assert len(pm.closed_positions) == 1
        
    def test_portfolio_stats(self):
        """Test portfolio statistics calculation"""
        pm = PositionManager(initial_balance=10000.0)
        
        # Open two positions
        pm.open_position(
            symbol="AAPL",
            direction="long",
            entry_price=150.0,
            size=10.0
        )
        
        pm.open_position(
            symbol="MSFT",
            direction="short",
            entry_price=300.0,
            size=5.0
        )
        
        # Get stats with current prices
        stats = pm.get_portfolio_stats({
            "AAPL": 155.0,
            "MSFT": 290.0
        })
        
        assert "current_balance" in stats
        assert "open_positions" in stats
        assert "total_exposure" in stats
        assert "unrealized_pnl" in stats
        assert "total_equity" in stats
        
        assert stats["open_positions"] == 2
        assert stats["unrealized_pnl"] == 100.0  # (155-150)*10 + (300-290)*5
        assert stats["total_equity"] == pm.current_balance + stats["unrealized_pnl"] 