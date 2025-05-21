import yfinance as yf
import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('test_yfinance')

def test_yfinance_download():
    """Test basic yfinance downloads with various configurations"""
    
    symbols = ["AAPL", "GOOGL", "MSFT"]
    
    for symbol in symbols:
        logger.info(f"\n=== Testing download for {symbol} ===")
        
        # Test 1: Basic download with auto_adjust=False
        logger.info("Test 1: Basic download with auto_adjust=False")
        data1 = yf.download(symbol, period="1mo", interval="1d", auto_adjust=False)
        logger.info(f"Shape: {data1.shape}")
        logger.info(f"Columns: {data1.columns}")
        logger.info(f"First row:\n{data1.head(1)}")
        
        # Test 2: Download with auto_adjust=True
        logger.info("\nTest 2: Download with auto_adjust=True")
        data2 = yf.download(symbol, period="1mo", interval="1d", auto_adjust=True)
        logger.info(f"Shape: {data2.shape}")
        logger.info(f"Columns: {data2.columns}")
        logger.info(f"First row:\n{data2.head(1)}")
        
        # Test 3: Download as a list
        logger.info("\nTest 3: Download as a list")
        data3 = yf.download([symbol], period="1mo", interval="1d", auto_adjust=False)
        logger.info(f"Shape: {data3.shape}")
        logger.info(f"Columns: {data3.columns}")
        logger.info(f"Is MultiIndex: {isinstance(data3.columns, pd.MultiIndex)}")
        if isinstance(data3.columns, pd.MultiIndex):
            logger.info(f"MultiIndex levels: {data3.columns.levels}")
            # Try flattening
            data3_flat = data3.droplevel(0, axis=1)
            logger.info(f"After flattening - Columns: {data3_flat.columns}")
        logger.info(f"First row:\n{data3.head(1)}")
        
        # Test 4: Download multiple symbols
        logger.info("\nTest 4: Download multiple symbols")
        data4 = yf.download(["AAPL", "MSFT"], period="1mo", interval="1d", auto_adjust=False)
        logger.info(f"Shape: {data4.shape}")
        logger.info(f"Columns: {data4.columns}")
        logger.info(f"Is MultiIndex: {isinstance(data4.columns, pd.MultiIndex)}")
        if isinstance(data4.columns, pd.MultiIndex):
            logger.info(f"MultiIndex levels: {data4.columns.levels}")
            # Check the specific symbol
            symbol_data = data4.xs(symbol, axis=1, level=1, drop_level=True) if symbol in data4.columns.levels[1] else None
            if symbol_data is not None:
                logger.info(f"Symbol specific data - Shape: {symbol_data.shape}")
                logger.info(f"Symbol specific data - Columns: {symbol_data.columns}")
        logger.info(f"First row:\n{data4.head(1)}")

if __name__ == "__main__":
    test_yfinance_download() 