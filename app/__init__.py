import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from dotenv import load_dotenv
from flask_login import LoginManager

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
login_manager = LoginManager()

# Simple user class for Flask-Login
class User:
    def __init__(self, id, username):
        self.id = id
        self.username = username
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
    
    def get_id(self):
        return str(self.id)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    # In this simplified version, we just check if it's the admin user
    if user_id == '1':  # Admin user ID
        return User(1, os.environ.get('ADMIN_USERNAME', 'admin'))
    return None

def create_app(testing=False):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config['TESTING'] = testing
    
    # Set up configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key')
    app.config['DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///trading.db')
    
    # Initialize Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'warning'
    
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
    
    # Enregistrer le blueprint de santé pour les health checks Docker
    from app.routes.health import health_bp
    app.register_blueprint(health_bp)
    
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
