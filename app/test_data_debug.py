import os
import sys
import logging
import pandas as pd
import numpy as np
import yfinance as yf

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the feature preparation function from the model
from app.models.price_prediction import PricePredictionModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('test_data_debug')

def fetch_market_data(symbol, period="3mo", interval="1d"):
    """Improved version of fetch_market_data with proper MultiIndex handling"""
    try:
        logger.info(f"Fetching market data for {symbol}")
        
        # Use a single symbol string
        data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        
        # Print the raw data info
        logger.info(f"Raw data info:")
        logger.info(f"Shape: {data.shape}")
        logger.info(f"Columns: {data.columns}")
        logger.info(f"Data types: {data.dtypes}")
        logger.info(f"First few rows: \n{data.head(2)}")
        
        # Check if data is empty
        if data.empty:
            logger.warning(f"No data received for {symbol} using period={period}. Attempting with maximum period...")
            data = yf.download(symbol, period="max", interval=interval, progress=False, auto_adjust=False)
            
            if data.empty:
                raise ValueError(f"No data received for {symbol} even with maximum period")
        
        # Handle the MultiIndex columns that yfinance always returns
        if isinstance(data.columns, pd.MultiIndex):
            logger.info(f"Converting MultiIndex columns to flat columns")
            # The first level contains the price types (Open, High, Low, Close, etc.)
            # The second level contains the ticker symbols
            
            # Extract the ticker specific data if available
            if symbol in data.columns.get_level_values(1):
                # Get data for this specific symbol and drop the ticker level
                data = data.xs(symbol, axis=1, level=1)
                logger.info(f"Extracted data for {symbol} from MultiIndex")
            else:
                # Just drop the first level if we can't find the symbol
                # This results in columns like 'AAPL', 'AAPL', etc. which is not what we want
                data = data.droplevel(0, axis=1)
                logger.warning(f"Could not find {symbol} in MultiIndex, dropped level instead")
            
            logger.info(f"Columns after MultiIndex processing: {list(data.columns)}")
        
        # Additional debugging of data
        logger.info(f"Data info after processing:")
        logger.info(f"Shape: {data.shape}")
        logger.info(f"Any NaN values: {data.isna().any().any()}")
        logger.info(f"NaN count per column: {data.isna().sum()}")
        logger.info(f"Last few rows: \n{data.tail(2)}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise

def debug_feature_preparation():
    """Debug the feature preparation process"""
    
    symbol = "AAPL"
    logger.info(f"Starting feature preparation debug for {symbol}")
    
    # Fetch the data
    data = fetch_market_data(symbol, period="1mo")
    
    # Create a model instance to access its methods
    model = PricePredictionModel()
    
    # Get the _prepare_features method
    prepare_features = model._prepare_features
    
    # Log data shape and columns before feature preparation
    logger.info(f"Data shape before feature preparation: {data.shape}")
    logger.info(f"Data columns before feature preparation: {list(data.columns)}")
    
    # Try to prepare features
    try:
        logger.info("Calling _prepare_features method...")
        feature_data = prepare_features(data)
        
        # Log data shape and columns after feature preparation
        logger.info(f"Data shape after feature preparation: {feature_data.shape}")
        logger.info(f"Data columns after feature preparation: {list(feature_data.columns)}")
        logger.info(f"Any NaN values: {feature_data.isna().any().any()}")
        logger.info(f"NaN count per column: {feature_data.isna().sum()}")
        
        # Check the Close column specifically
        if 'Close' in feature_data.columns:
            logger.info(f"Close column data: {feature_data['Close'].head()}")
            logger.info(f"Close column shape: {feature_data['Close'].shape}")
            logger.info(f"Close column has NaN: {feature_data['Close'].isna().any()}")
        else:
            logger.error("Close column not found in feature data!")
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("Debug complete")

if __name__ == "__main__":
    debug_feature_preparation() 