import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import local modules
from app.filters import register_filters
from app.models.position_manager import PositionManager
from app.monitoring import init_monitoring
from app.api import register_api_routes

# Global objects
trading_bot = None
data_manager = None
position_manager = None

def create_app(testing=False):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config['TESTING'] = testing
    
    # Set up configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key')
    app.config['DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///trading.db')
    
    # Initialize global objects if not in testing mode
    if not testing:
        from app.trading import TradingBot, DataManager
        global trading_bot, data_manager, position_manager
        
        # Initialiser le PositionManager
        position_manager = PositionManager(initial_balance=float(os.environ.get('INITIAL_BALANCE', '100000.0')))
        
        # Initialiser le TradingBot avec notre PositionManager
        trading_bot = TradingBot(initial_balance=float(os.environ.get('INITIAL_BALANCE', '100000.0')), 
                                position_manager=position_manager)
        
        # Initialiser le DataManager avec les symboles
        symbols = os.environ.get('TRADING_SYMBOLS', 'AAPL,MSFT,GOOGL').split(',')
        data_manager = DataManager(symbols=symbols)
        
        # Make global objects available to the app
        app.config['trading_bot'] = trading_bot
        app.config['data_manager'] = data_manager
        app.config['position_manager'] = position_manager
    
    # Register custom filters
    register_filters(app)
    
    # Register blueprints
    from app.routes import main_blueprint
    app.register_blueprint(main_blueprint)
    
    # Register API routes
    register_api_routes(app)
    
    # Initialize monitoring (Prometheus metrics)
    init_monitoring(app, port=int(os.environ.get('PROMETHEUS_PORT', '9090')))
    
    # Set up logging
    setup_logging(app)
    
    # Ensure required directories exist
    for directory in ['data', 'logs', 'saved_models']:
        os.makedirs(directory, exist_ok=True)
    
    return app

def setup_logging(app):
    """Configure logging for the application."""
    log_level = getattr(logging, os.environ.get('LOG_LEVEL', 'INFO'))
    log_file = os.environ.get('LOG_FILE', 'logs/app.log')
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure file handler with rotation
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(log_level)
    
    # Configure app logger
    app.logger.addHandler(file_handler)
    app.logger.setLevel(log_level)
    
    # Log application startup
    app.logger.info('Application starting up...')
