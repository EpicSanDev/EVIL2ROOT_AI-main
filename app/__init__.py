import os
import logging
import datetime
import hashlib
import time
from logging.handlers import RotatingFileHandler
from flask import Flask, request, abort
from dotenv import load_dotenv
from flask_login import LoginManager, current_user
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect

# Load environment variables
load_dotenv()

# Import local modules
from app.ui.filters import register_filters
from app.models.trading.position_manager import PositionManager
from app.monitoring import init_monitoring
from app.api import register_api_routes
from app.models.db_user import db, User
from app.routes import register_routes

# Global objects
trading_bot = None
data_manager = None
position_manager = None
login_manager = LoginManager()
csrf = CSRFProtect()
migrate = Migrate()

# Dictionary to store failed login attempts
login_attempts = {}
# Lock duration in seconds (15 minutes)
LOCK_DURATION = 15 * 60
# Maximum number of failed attempts before locking
MAX_ATTEMPTS = 5

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
    
    # Si connexion réussie, réinitialiser les tentatives
    if success:
        if key in login_attempts:
            del login_attempts[key]
        return True
    
    # Vérifier si le compte est verrouillé
    if key in login_attempts:
        attempt_info = login_attempts[key]
        
        # Si le compte est verrouillé, vérifier si la période de verrouillage est terminée
        if attempt_info.get('locked_until') and attempt_info['locked_until'] > current_time:
            return False
        
        # Si la période de verrouillage est terminée, réinitialiser les tentatives
        if attempt_info.get('locked_until') and attempt_info['locked_until'] <= current_time:
            attempt_info['count'] = 0
            attempt_info['locked_until'] = None
        
        # Incrémenter le compteur de tentatives
        attempt_info['count'] += 1
        attempt_info['last_attempt'] = current_time
        
        # Vérifier si le nombre maximum de tentatives est atteint
        if attempt_info['count'] >= MAX_ATTEMPTS:
            attempt_info['locked_until'] = current_time + LOCK_DURATION
            return False
    else:
        # Première tentative pour cette adresse IP et cet utilisateur
        login_attempts[key] = {
            'count': 1,
            'last_attempt': current_time,
            'locked_until': None
        }
    
    return True

@login_manager.user_loader
def load_user(user_id):
    """Charge un utilisateur à partir de son ID"""
    return User.query.get(user_id)

def create_app(testing=False):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config['TESTING'] = testing
    
    # Set up configuration with secure defaults
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(32).hex())
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///trading.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Security settings
    app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(hours=2)
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    migrate.init_app(app, db)
    
    # Configure login manager
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Veuillez vous connecter pour accéder à cette page.'
    login_manager.login_message_category = 'warning'
    login_manager.session_protection = 'strong'
    
    # Initialize global objects if not in testing mode
    if not testing:
        from app.core.trading import TradingBot, DataManager
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
    
    # Register API routes
    register_api_routes(app)
    
    # Register application routes
    register_routes(app)
    
    # Set up logging
    setup_logging(app)
    
    # Initialize monitoring
    if not testing and os.environ.get('ENABLE_MONITORING', 'false').lower() == 'true':
        init_monitoring(app)
    
    # Création du dossier pour les uploads de profil
    upload_dir = os.path.join(app.static_folder, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create all database tables
    with app.app_context():
        db.create_all()
        # Créer l'utilisateur admin par défaut s'il n'existe pas
        create_default_admin()
    
    return app

def create_default_admin():
    """Crée un utilisateur admin par défaut si aucun utilisateur n'existe"""
    admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
    admin_password = os.environ.get('ADMIN_PASSWORD', 'admin')
    admin_email = os.environ.get('ADMIN_EMAIL', 'admin@example.com')
    
    # Vérifier si l'admin existe déjà
    admin = User.query.filter_by(username=admin_username).first()
    if admin is None:
        admin = User(
            id="1",  # ID fixe pour admin
            username=admin_username,
            email=admin_email,
            subscription_type='enterprise',
            is_active=True
        )
        admin.set_password(admin_password)
        db.session.add(admin)
        db.session.commit()
        logging.info(f"Utilisateur administrateur {admin_username} créé avec succès.")

def setup_logging(app):
    """Set up logging for the application."""
    log_level = getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper())
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'app.log')
    
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add a file handler for app.log
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    handler.setLevel(log_level)
    
    # Add handlers to Flask app logger
    app.logger.addHandler(handler)
    app.logger.setLevel(log_level)
    
    # Also log to stderr
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    console_handler.setLevel(log_level)
    app.logger.addHandler(console_handler)
    
    return handler
