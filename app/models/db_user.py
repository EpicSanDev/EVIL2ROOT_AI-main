"""
Modèle SQLAlchemy pour la gestion des utilisateurs
Fournit une implémentation compatible avec Flask-Login
"""

from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
from flask_sqlalchemy import SQLAlchemy

# Initialisation de SQLAlchemy
db = SQLAlchemy()

class User(db.Model, UserMixin):
    """
    Modèle d'utilisateur pour l'authentification et la gestion des profils
    Compatible avec Flask-Login
    """
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(64), nullable=True)
    last_name = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    subscription_type = db.Column(db.String(20), default='free')
    subscription_expiry = db.Column(db.DateTime, nullable=True)
    telegram_id = db.Column(db.String(20), nullable=True, unique=True)
    last_login = db.Column(db.DateTime, nullable=True)
    profile_image = db.Column(db.String(255), nullable=True)
    
    def set_password(self, password):
        """Hache le mot de passe de l'utilisateur pour un stockage sécurisé"""
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        """Vérifie si le mot de passe fourni correspond au hash stocké"""
        return check_password_hash(self.password_hash, password)
    
    def is_subscription_active(self):
        """Vérifie si l'abonnement de l'utilisateur est actif"""
        if self.subscription_type == 'free':
            return True
        if not self.subscription_expiry:
            return False
        return self.subscription_expiry > datetime.utcnow()
    
    def get_subscription_days_left(self):
        """Retourne le nombre de jours restants dans l'abonnement"""
        if self.subscription_type == 'free':
            return float('inf')  # Infini pour les comptes gratuits
        if not self.subscription_expiry:
            return 0
        delta = self.subscription_expiry - datetime.utcnow()
        return max(0, delta.days)
    
    def to_dict(self):
        """Convertit l'utilisateur en dictionnaire pour l'API"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_active': self.is_active,
            'subscription_type': self.subscription_type,
            'subscription_expiry': self.subscription_expiry.isoformat() if self.subscription_expiry else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'telegram_id': self.telegram_id,
            'has_profile_image': bool(self.profile_image)
        }
    
    def __repr__(self):
        return f'<User {self.username}>' 