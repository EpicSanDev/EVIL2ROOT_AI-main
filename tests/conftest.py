import pytest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.trading import TradingBot, DataManager

@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    app = create_app(testing=True)
    
    yield app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """A test CLI runner for the app."""
    return app.test_cli_runner()

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    # Create a date range
    date_range = pd.date_range(start='2020-01-01', end='2022-01-01', freq='D')
    
    # Create sample price data
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, len(date_range)),
        'High': np.random.uniform(100, 200, len(date_range)),
        'Low': np.random.uniform(100, 200, len(date_range)),
        'Close': np.random.uniform(100, 200, len(date_range)),
        'Volume': np.random.uniform(1000000, 5000000, len(date_range))
    }, index=date_range)
    
    # Ensure High is the highest and Low is the lowest for each day
    for i in range(len(data)):
        values = data.iloc[i, 0:4].values
        data.iloc[i, 1] = max(values)  # High
        data.iloc[i, 2] = min(values)  # Low
    
    return data

@pytest.fixture
def data_manager():
    """Create a DataManager instance for testing."""
    # Create a test data manager with sample data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data_manager = DataManager(symbols=symbols)
    
    # Mock the data fetching to use sample data
    for symbol in symbols:
        data_manager.data[symbol] = sample_data()
    
    return data_manager

@pytest.fixture
def position_manager():
    """Create a test position manager with initial positions"""
    from app.models.position_manager import PositionManager
    
    # Initialize with test balance
    position_manager = PositionManager(initial_balance=10000.0)
    
    # Add some test positions
    position_manager.open_position(
        symbol='AAPL',
        direction='long',
        entry_price=150.0,
        size=10.0,
        stop_loss=145.0,
        take_profit=160.0,
        metadata={'test': True}
    )
    
    position_manager.open_position(
        symbol='MSFT',
        direction='short',
        entry_price=300.0,
        size=5.0,
        stop_loss=310.0,
        take_profit=280.0,
        metadata={'test': True}
    )
    
    return position_manager

@pytest.fixture
def trading_bot(data_manager, position_manager):
    """Create a test trading bot with mocked models"""
    from app.trading import TradingBot
    
    # Initialize with test balance and provided position manager
    bot = TradingBot(initial_balance=10000.0, position_manager=position_manager)
    bot.data_manager = data_manager
    
    # Mock the telegram bot to avoid actual message sending
    bot.telegram_bot = MagicMock()
    
    return bot 