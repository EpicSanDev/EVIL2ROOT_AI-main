import os
import sys
import logging
import pandas as pd
import numpy as np
import yfinance as yf

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the models
from app.models.price_prediction import PricePredictionModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('simple_train_test')

def create_synthetic_data(periods=100):
    """Create synthetic price data that resembles stock market data"""
    logger.info(f"Creating synthetic data with {periods} periods")
    
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=periods)
    
    # Generate realistic stock price data
    base_price = 100.0
    daily_returns = np.random.normal(loc=0.0005, scale=0.01, size=periods)
    daily_returns[0] = 0  # Start with no returns
    
    # Cumulative returns
    cum_returns = np.cumprod(1 + daily_returns)
    
    # Generate prices
    close_prices = base_price * cum_returns
    
    # Generate OHLC data
    daily_volatility = 0.015
    high_prices = close_prices * (1 + np.random.uniform(0, daily_volatility, size=periods))
    low_prices = close_prices * (1 - np.random.uniform(0, daily_volatility, size=periods))
    open_prices = low_prices + np.random.uniform(0, 1, size=periods) * (high_prices - low_prices)
    
    # Generate volume
    volume = np.random.randint(1000000, 10000000, size=periods)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume,
        'Adj Close': close_prices  # For simplicity, use same as close
    }, index=dates)
    
    logger.info(f"Created synthetic data with shape {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"First few rows:\n{df.head(3)}")
    
    return df

def test_simple_training():
    """Test simple training using synthetic data"""
    
    symbol = "TEST"
    logger.info(f"Starting simple training test for synthetic {symbol} data")
    
    # Create synthetic data
    data = create_synthetic_data(periods=250)  # About a year of trading days
    
    # Create model instance
    model = PricePredictionModel()
    logger.info(f"Created model instance: {type(model).__name__}")
    
    # Train with named parameters
    try:
        logger.info(f"Training model with synthetic data")
        model.train(data=data, symbol=symbol, optimize=False)
        logger.info(f"Training succeeded!")
        
        # Try prediction
        prediction = model.predict(data, symbol)
        logger.info(f"Prediction for {symbol}: {prediction}")
        
        logger.info(f"Test completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_training() 