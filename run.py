import os
import logging
from logging.handlers import RotatingFileHandler
import asyncio
import yfinance as yf
import flask
import threading
from datetime import datetime, timedelta
import schedule
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from app import create_app
from app.trading import TradingBot, DataManager
from app.telegram_bot import TelegramBot
from app.model_trainer import ModelTrainer

# Load environment variables
load_dotenv()

# Configuration
class Config:
    # Trading parameters
    INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '100000'))
    RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '5'))
    UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', '5'))  # minutes
    
    # Data parameters
    START_DATE = os.getenv('START_DATE', '2010-01-01')
    DATA_DIR = Path('data')
    MODELS_DIR = Path('saved_models')
    LOG_DIR = Path('logs')
    
    # Create necessary directories
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    # File paths
    MARKET_DATA = DATA_DIR / 'market_data.csv'
    CLEANED_DATA = DATA_DIR / 'market_data_cleaned.csv'
    LOG_FILE = LOG_DIR / 'trading_bot.log'
    
    # Trading symbols
    SYMBOLS = {
        'stocks': [
            "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", 
            "NFLX", "BABA", "AMD", "INTC"
        ],
        'crypto': [
            "BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "DOT-USD", 
            "BNB-USD", "SOL-USD"
        ],
        'forex': [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"
        ]
    }
    
    @classmethod
    def get_all_symbols(cls):
        return (cls.SYMBOLS['stocks'] + 
                cls.SYMBOLS['crypto'] + 
                cls.SYMBOLS['forex'])

# Setup logging
def setup_logging():
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        Config.LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class TradingSystem:
    def __init__(self):
        self.app = create_app()
        self.data_manager = None
        self.trading_bot = None
        self.telegram_bot = None
        self.model_trainer = None
        
    async def initialize(self):
        """Initialize all components of the trading system"""
        try:
            logger.info("Initializing trading system...")
            
            # Initialize components
            self.data_manager = DataManager(
                Config.get_all_symbols(),
                start_date=Config.START_DATE
            )
            
            self.trading_bot = TradingBot(
                initial_balance=Config.INITIAL_BALANCE
            )
            
            self.telegram_bot = TelegramBot()
            self.model_trainer = ModelTrainer(self.trading_bot)
            
            # Ensure market data exists
            await self.ensure_market_data()
            
            logger.info("Trading system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system: {e}")
            return False
    
    async def ensure_market_data(self):
        """Ensure market data is available and up to date"""
        try:
            if not Config.MARKET_DATA.exists():
                logger.info("Downloading market data...")
                data = yf.download(
                    Config.get_all_symbols(),
                    start=Config.START_DATE,
                    end=datetime.now().strftime('%Y-%m-%d')
                )
                data.to_csv(str(Config.MARKET_DATA))
                logger.info("Market data downloaded successfully")
            
            # Clean the data
            from data_cleaner import clean_data
            clean_data(str(Config.MARKET_DATA), str(Config.CLEANED_DATA))
            logger.info("Market data cleaned successfully")
            
        except Exception as e:
            logger.error(f"Error ensuring market data: {e}")
            raise
    
    async def train_models(self):
        """Train all models"""
        try:
            logger.info("Starting model training...")
            
            # Train standard models
            await self.model_trainer.train_all_models(self.data_manager)
            
            # Train RL model
            self.trading_bot.train_rl_model(str(Config.CLEANED_DATA))
            
            # Run backtest
            backtest_results = self.trading_bot.run_backtest(
                str(Config.CLEANED_DATA),
                risk_percentage=Config.RISK_PER_TRADE,
                max_positions=Config.MAX_POSITIONS
            )
            
            logger.info("Model training completed successfully")
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def run_flask_app(self):
        """Run Flask app in a separate thread"""
        try:
            self.app.run(host='0.0.0.0', port=5000)
        except Exception as e:
            logger.error(f"Flask app error: {e}")
    
    async def run(self):
        """Main run loop"""
        try:
            # Initialize system
            if not await self.initialize():
                logger.error("Failed to initialize. Exiting.")
                return
            
            try:
                # Send start message (non-fatal if it fails)
                await self.telegram_bot.send_message(
                    "Trading system starting..."
                )
            except Exception as e:
                logger.warning(f"Failed to send start message (continuing anyway): {e}")
            
            # Train models
            backtest_results = await self.train_models()
            
            # Start real-time trading
            self.trading_bot.start_real_time_scanning(
                self.data_manager,
                interval_seconds=Config.UPDATE_INTERVAL * 60
            )
            
            # Keep the main loop running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            try:
                await self.telegram_bot.send_message(
                    f"Trading system error: {str(e)}"
                )
            except:
                pass  # Ignore Telegram errors
        finally:
            try:
                await self.telegram_bot.send_message(
                    "Trading system shutting down..."
                )
            except:
                pass  # Ignore Telegram errors

async def main():
    """Main entry point"""
    trading_system = TradingSystem()
    
    # Start Flask app in separate thread
    flask_thread = threading.Thread(
        target=trading_system.run_flask_app,
        daemon=True
    )
    flask_thread.start()
    
    # Run trading system
    await trading_system.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
