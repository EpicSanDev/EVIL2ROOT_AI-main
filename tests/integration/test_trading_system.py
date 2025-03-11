import pytest
import pandas as pd
import numpy as np
from app.trading import TradingBot, DataManager
from app.models.price_prediction import PricePredictionModel
from app.models.risk_management import RiskManagementModel

class TestTradingSystem:
    """Integration tests for the trading system components."""
    
    def test_trading_bot_initialization(self, trading_bot):
        """Test that the trading bot initializes correctly."""
        assert trading_bot is not None
        assert trading_bot.initial_balance == 100000
        assert trading_bot.current_balance == 100000
        assert len(trading_bot.positions) == 0
    
    def test_data_manager_provides_data_to_trading_bot(self, data_manager, trading_bot):
        """Test that the data manager correctly provides data to the trading bot."""
        # Get data for a specific symbol
        symbol = data_manager.symbols[0]
        data = data_manager.data.get(symbol)
        
        # Verify the data has been loaded
        assert data is not None
        assert not data.empty
        assert 'Open' in data.columns
        assert 'Close' in data.columns
        
        # Simulate the trading bot processing this data
        trading_bot.process_market_data(symbol, data)
        
        # Check that the bot processed the data without errors
        # This is a basic check - more specific checks would depend on the implementation
        assert trading_bot is not None
    
    def test_trading_signals_generation(self, data_manager, trading_bot):
        """Test that trading signals can be generated from market data."""
        # Get data for a specific symbol
        symbol = data_manager.symbols[0]
        data = data_manager.data.get(symbol)
        
        # Create a price prediction model
        price_model = PricePredictionModel()
        
        # Mock the model's predict method
        def mock_predict(data):
            # Generate some random predictions with an upward bias for testing
            preds = data['Close'].values[-10:] * (1 + np.random.normal(0.001, 0.005, 10))
            return preds.reshape(-1, 1)
        
        price_model.predict = mock_predict
        
        # Register the model with the trading bot
        trading_bot.price_model = price_model
        
        # Generate trading signals
        signals = trading_bot.generate_trading_signals(symbol, data)
        
        # Verify that signals were generated
        assert signals is not None
        assert isinstance(signals, list)
    
    def test_risk_management_integration(self, data_manager, trading_bot):
        """Test that risk management is correctly integrated with trading."""
        # Initialize a risk model
        risk_model = RiskManagementModel()
        
        # Register the model with the trading bot
        trading_bot.risk_model = risk_model
        
        # Simulate a trade decision
        symbol = data_manager.symbols[0]
        price = 150.0
        direction = 'buy'
        
        # Calculate position size using risk management
        position_size = trading_bot.calculate_position_size(
            symbol=symbol,
            price=price,
            risk_percentage=0.02,  # 2% risk per trade
            stop_loss_percentage=0.05  # 5% stop loss
        )
        
        # Verify that a valid position size was calculated
        assert position_size > 0
        # Verify that the position size respects the risk limits
        # (2% of 100,000 with 5% stop loss would be around 400 shares at $150)
        expected_position = (100000 * 0.02) / (150 * 0.05)
        assert abs(position_size - expected_position) < 1  # Allow for small rounding differences 