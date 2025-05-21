import os
import sys
import logging
import pandas as pd
from datetime import datetime
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
logger = logging.getLogger('test_price_model')

def main():
    """Test the price prediction model with different training approaches"""
    logger.info("Starting price prediction model test")
    
    # Symbol to test
    symbol = "AAPL"
    
    # Initialize model
    model = PricePredictionModel()
    logger.info(f"Created model instance: {type(model).__name__}")
    
    # Fetch data
    logger.info(f"Fetching data for {symbol}")
    data = yf.download(symbol, period="2mo", interval="1d", auto_adjust=False)
    logger.info(f"Downloaded {len(data)} data points")
    
    # Test 1: Train with positional arguments
    logger.info("TEST 1: Training with positional arguments")
    try:
        model.train(data, symbol, True)
        logger.info("✅ Success: Training with positional arguments worked")
    except Exception as e:
        logger.error(f"❌ Error: Training with positional arguments failed: {e}")
    
    # Test 2: Train with named arguments
    logger.info("TEST 2: Training with named arguments")
    try:
        model.train(data=data, symbol=symbol, optimize=False)
        logger.info("✅ Success: Training with named arguments worked")
    except Exception as e:
        logger.error(f"❌ Error: Training with named arguments failed: {e}")
    
    # Test 3: Predict after training
    logger.info("TEST 3: Predicting after training")
    try:
        prediction = model.predict(data, symbol)
        logger.info(f"✅ Success: Prediction result = {prediction}")
    except Exception as e:
        logger.error(f"❌ Error: Prediction failed: {e}")
    
    # Create a fresh model to test keyword args
    logger.info("Creating fresh model instance")
    fresh_model = PricePredictionModel()
    
    # Test 4: Train with kwargs
    logger.info("TEST 4: Training with kwargs")
    try:
        kwargs = {"data": data, "symbol": symbol, "optimize": False}
        fresh_model.train(**kwargs)
        logger.info("✅ Success: Training with kwargs worked")
    except Exception as e:
        logger.error(f"❌ Error: Training with kwargs failed: {e}")
    
    logger.info("Tests completed")

if __name__ == "__main__":
    main() 