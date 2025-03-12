import os
import sys
import logging
import yfinance as yf
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the models
from app.models.price_prediction import PricePredictionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('test_price_prediction')

def test_direct_training():
    """Test the direct training approach we implemented"""
    try:
        # Symbol to test
        symbol = "AAPL"
        logger.info(f"Testing direct training approach for {symbol}")
        
        # Fetch market data
        logger.info(f"Fetching market data for {symbol}")
        market_data = yf.download(symbol, period="1mo", interval="1d", auto_adjust=False)
        logger.info(f"Downloaded {len(market_data)} data points for {symbol}")
        
        # Create model instance
        logger.info(f"Creating fresh model instance")
        model = PricePredictionModel()
        
        # Direct approach training - using positional arguments as in train_all_models
        logger.info(f"Calling model.train with positional arguments")
        model.train(market_data, symbol, True)
        logger.info(f"Direct call to train succeeded!")
        
        # Test prediction
        logger.info(f"Testing prediction")
        prediction = model.predict(market_data, symbol)
        logger.info(f"Prediction result: {prediction}")
        
        # Test training with named arguments
        logger.info(f"Testing training with named arguments")
        model2 = PricePredictionModel()
        model2.train(data=market_data, symbol=symbol, optimize=False)
        logger.info(f"Named arguments training succeeded!")
        
        logger.info(f"All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_training() 