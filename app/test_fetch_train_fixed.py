import os
import sys
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

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
logger = logging.getLogger('test_fetch_train')

def fetch_market_data(symbol, period="3mo", interval="1d"):
    """Improved version of fetch_market_data with proper MultiIndex handling"""
    try:
        logger.info(f"Fetching market data for {symbol}")
        
        # Use a single symbol string to avoid MultiIndex complications
        # Explicitly set auto_adjust=False to maintain backward compatibility
        data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        
        # Check if data is empty
        if data.empty:
            logger.warning(f"No data received for {symbol} using period={period}. Attempting with maximum period...")
            # Try again with max period as fallback
            data = yf.download(symbol, period="max", interval=interval, progress=False, auto_adjust=False)
            
            if data.empty:
                raise ValueError(f"No data received for {symbol} even with maximum period")
        
        # Handle the MultiIndex columns that yfinance always returns
        if isinstance(data.columns, pd.MultiIndex):
            logger.info(f"Converting MultiIndex columns to flat columns for {symbol}")
            # The first level contains the price types (Open, High, Low, Close, etc.)
            # The second level contains the ticker symbols
            
            # Extract the ticker specific data if available
            if symbol in data.columns.get_level_values(1):
                # Get data for this specific symbol and drop the ticker level
                data = data.xs(symbol, axis=1, level=1)
                logger.info(f"Extracted data for {symbol} from MultiIndex")
            else:
                # Just drop the second level if we can't find the symbol
                data = data.droplevel(1, axis=1)
                logger.warning(f"Could not find {symbol} in MultiIndex, dropped level instead")
            
            logger.info(f"Columns after MultiIndex processing: {list(data.columns)}")
        
        # Check again if data is empty after processing
        if data.empty:
            raise ValueError(f"No valid data received for {symbol} after processing")
            
        # Print the column names to help with debugging
        logger.info(f"Final DataFrame columns: {data.columns.tolist()}")
        
        # Verify we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
        
        # Log data size for debugging
        logger.info(f"Downloaded {len(data)} data points for {symbol}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise

def add_technical_indicators(data):
    """Add basic technical indicators for testing"""
    logger.info("Adding technical indicators to data")
    
    # Make sure we have basic required columns
    if 'Close' not in data.columns:
        logger.error("Cannot add indicators - Close column missing!")
        return data
    
    # Simple moving averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index) - simplified calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    logger.info(f"Added technical indicators. New columns: {list(data.columns)}")
    return data

def test_fetch_and_train():
    """Test the fetch and train process with our fixes"""
    try:
        # Symbol to test
        symbol = "AAPL"
        logger.info(f"Starting fetch and train test for {symbol}")
        
        # Fetch the data with improved method
        market_data = fetch_market_data(symbol, period="1mo")
        
        # Add technical indicators for model training
        market_data = add_technical_indicators(market_data)
        
        logger.info(f"Final data shape: {market_data.shape}")
        logger.info(f"Final data columns: {list(market_data.columns)}")
        
        # Verify critical columns exist
        if 'Close' not in market_data.columns:
            logger.error("Critical column 'Close' missing from data - cannot proceed with training")
            return
        
        # Create model instance
        model = PricePredictionModel()
        logger.info(f"Created model instance: {type(model).__name__}")
        
        # Train with named parameters
        logger.info(f"Training model with named parameters")
        model.train(data=market_data, symbol=symbol, optimize=False)
        logger.info(f"Training succeeded!")
        
        # Test prediction
        logger.info(f"Testing prediction")
        prediction = model.predict(market_data, symbol)
        logger.info(f"Prediction result: {prediction}")
        
        logger.info(f"Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fetch_and_train() 