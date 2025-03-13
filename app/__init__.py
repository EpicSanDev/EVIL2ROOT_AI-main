import os
import logging
import datetime
import hashlib
import time
from logging.handlers import RotatingFileHandler
from flask import Flask, request, abort
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

# Dictionary to store failed login attempts
login_attempts = {}
# Lock duration in seconds (15 minutes)
LOCK_DURATION = 15 * 60
# Maximum number of failed attempts before locking
MAX_ATTEMPTS = 5

# Secure user class with password hashing
class User:
    def __init__(self, id, username, password_hash=None, salt=None):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.salt = salt
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
    
    def get_id(self):
        return str(self.id)
    
    @staticmethod
    def hash_password(password, salt=None):
        """Hash a password with a randomly generated salt"""
        if salt is None:
            salt = os.urandom(32).hex()  # Generate a random salt
        
        # Use a secure hash algorithm with key stretching
        key = hashlib.pbkdf2_hmac(
            'sha256',                  # The hash algorithm
            password.encode('utf-8'),  # Convert password to bytes
            salt.encode('utf-8'),      # Salt as bytes
            100000,                    # Number of iterations (key stretching)
            dklen=128                  # Length of the derived key
        ).hex()
        
        return key, salt
    
    def verify_password(self, password):
        """Verify a password against its hash"""
        if not self.password_hash or not self.salt:
            return False
        
        # Hash the provided password with the stored salt
        hash_to_check, _ = self.hash_password(password, self.salt)
        
        # Compare in constant time to prevent timing attacks
        return self.constant_time_compare(hash_to_check, self.password_hash)
    
    @staticmethod
    def constant_time_compare(a, b):
        """Compare two strings in constant time to prevent timing attacks"""
        if len(a) != len(b):
            return False
        
        result = 0
        for x, y in zip(a, b):
            result |= ord(x) ^ ord(y)
        
        return result == 0

# User loader for Flask-Login with improved security
@login_manager.user_loader
def load_user(user_id):
    # Check for admin user
    if user_id == '1':  # Admin user ID
        admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
        admin_password = os.environ.get('ADMIN_PASSWORD', 'admin')
        
        # Create admin user with hashed password if not already done
        admin_user = User(1, admin_username)
        
        # Get or create password hash and salt
        if not hasattr(load_user, 'admin_hash') or not hasattr(load_user, 'admin_salt'):
            # Only hash on first call to avoid repeated hashing
            load_user.admin_hash, load_user.admin_salt = User.hash_password(admin_password)
        
        admin_user.password_hash = load_user.admin_hash
        admin_user.salt = load_user.admin_salt
        
        return admin_user
    
    return None

# Fonction pour vérifier les tentatives de connexion et limiter la force brute
def check_login_attempts(username, success=False):
    """
    Vérifier et mettre à jour les tentatives de connexion
    Retourne True si l'utilisateur peut tenter une connexion, False s'il est bloqué
    """
    current_time = time.time()
    ip_address = request.remote_addr
    
    # Clé unique pour l'utilisateur et l'adresse IP
    key = f"{username}:{ip_address}"
    
    # Si la connexion est réussie, réinitialiser les tentatives
    if success:
        if key in login_attempts:
            del login_attempts[key]
        return True
    
    # Vérifier si l'utilisateur est actuellement bloqué
    if key in login_attempts:
        attempts, last_attempt_time, is_locked_until = login_attempts[key]
        
        # Si l'utilisateur est bloqué, vérifier si la durée du blocage est écoulée
        if is_locked_until and current_time < is_locked_until:
            return False
        
        # Si le blocage est terminé, réinitialiser
        if is_locked_until and current_time >= is_locked_until:
            login_attempts[key] = (0, current_time, None)
        
        # Incrémenter le compteur d'échecs
        attempts += 1
        
        # Bloquer après MAX_ATTEMPTS tentatives échouées
        if attempts >= MAX_ATTEMPTS:
            login_attempts[key] = (attempts, current_time, current_time + LOCK_DURATION)
            return False
        else:
            login_attempts[key] = (attempts, current_time, None)
    else:
        # Première tentative échouée
        login_attempts[key] = (1, current_time, None)
    
    return True

def create_app(testing=False):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config['TESTING'] = testing
    
    # Set up configuration with secure defaults
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(32).hex())
    app.config['DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///trading.db')
    
    # Security settings
    app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(hours=2)
    
    # Initialize Flask-Login with enhanced security
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'
    login_manager.login_message = 'Veuillez vous connecter pour accéder à cette page.'
    login_manager.login_message_category = 'warning'
    login_manager.session_protection = 'strong'
    
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
