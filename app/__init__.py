from flask import Flask
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global trading objects to be initialized in create_app()
trading_bot = None
data_manager = None

def create_app():
    app = Flask(__name__)
    
    # Configure application
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-flask')
    
    # Import here to avoid circular imports
    from app.trading import TradingBot, DataManager
    
    # Initialize global objects with environment variables
    symbols = os.environ.get('SYMBOLS', 'AAPL,GOOGL,MSFT,BTC-USD').split(',')
    
    global trading_bot, data_manager
    if trading_bot is None:
        data_manager = DataManager(symbols)
        trading_bot = TradingBot(initial_balance=float(os.environ.get('INITIAL_BALANCE', '100000')))
    
    # Import and register the main blueprint
    from .routes import main_blueprint
    app.register_blueprint(main_blueprint)

    return app
