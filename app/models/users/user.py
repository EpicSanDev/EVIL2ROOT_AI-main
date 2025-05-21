"""
Modèle d'utilisateur pour le système d'authentification et de paiement.
Ce module gère les utilisateurs, leurs abonnements et leurs paiements.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration de la base de données
DB_NAME = os.environ.get('DB_NAME', 'trading_db')
DB_USER = os.environ.get('DB_USER', 'trader')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'secure_password')
DB_HOST = os.environ.get('DB_HOST', 'db')
DB_PORT = os.environ.get('DB_PORT', '5432')

# Types d'abonnements
class SubscriptionType:
    FREE = 'free'
    BASIC = 'basic'
    PREMIUM = 'premium'
    ENTERPRISE = 'enterprise'

# Statuts des paiements
class PaymentStatus:
    PENDING = 'pending'
    CONFIRMED = 'confirmed'
    FAILED = 'failed'
    EXPIRED = 'expired'

class User:
    """
    Classe représentant un utilisateur du système.
    Gère l'authentification, les abonnements et les paiements.
    """
    
    def __init__(self, id: str = None, username: str = None, email: str = None, 
                 telegram_id: str = None, subscription_type: str = SubscriptionType.FREE,
                 subscription_expiry: datetime = None, is_active: bool = True):
        """
        Initialise un nouvel utilisateur.
        
        Args:
            id: Identifiant unique de l'utilisateur (UUID)
            username: Nom d'utilisateur
            email: Adresse email
            telegram_id: Identifiant Telegram
            subscription_type: Type d'abonnement
            subscription_expiry: Date d'expiration de l'abonnement
            is_active: Indique si le compte est actif
        """
        self.id = id or str(uuid.uuid4())
        self.username = username
        self.email = email
        self.telegram_id = telegram_id
        self.subscription_type = subscription_type
        self.subscription_expiry = subscription_expiry or (datetime.now() + timedelta(days=30))
        self.is_active = is_active
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Attributs pour Flask-Login
        self.is_authenticated = True
        self.is_anonymous = False
    
    def get_id(self):
        """Méthode requise par Flask-Login"""
        return str(self.id)
    
    def is_subscription_active(self) -> bool:
        """Vérifie si l'abonnement de l'utilisateur est actif"""
        if self.subscription_type == SubscriptionType.FREE:
            return True
        
        return self.is_active and self.subscription_expiry > datetime.now()
    
    def get_subscription_days_left(self) -> int:
        """Retourne le nombre de jours restants dans l'abonnement"""
        if self.subscription_type == SubscriptionType.FREE:
            return 0
        
        if not self.is_subscription_active():
            return 0
        
        delta = self.subscription_expiry - datetime.now()
        return max(0, delta.days)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'utilisateur en dictionnaire"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'telegram_id': self.telegram_id,
            'subscription_type': self.subscription_type,
            'subscription_expiry': self.subscription_expiry.isoformat() if self.subscription_expiry else None,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_subscription_active': self.is_subscription_active(),
            'subscription_days_left': self.get_subscription_days_left()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Crée un utilisateur à partir d'un dictionnaire"""
        user = cls(
            id=data.get('id'),
            username=data.get('username'),
            email=data.get('email'),
            telegram_id=data.get('telegram_id'),
            subscription_type=data.get('subscription_type', SubscriptionType.FREE),
            is_active=data.get('is_active', True)
        )
        
        # Convertir les dates si elles sont présentes
        if data.get('subscription_expiry'):
            if isinstance(data['subscription_expiry'], str):
                user.subscription_expiry = datetime.fromisoformat(data['subscription_expiry'])
            else:
                user.subscription_expiry = data['subscription_expiry']
        
        if data.get('created_at'):
            if isinstance(data['created_at'], str):
                user.created_at = datetime.fromisoformat(data['created_at'])
            else:
                user.created_at = data['created_at']
        
        if data.get('updated_at'):
            if isinstance(data['updated_at'], str):
                user.updated_at = datetime.fromisoformat(data['updated_at'])
            else:
                user.updated_at = data['updated_at']
        
        return user

class UserManager:
    """
    Gestionnaire des utilisateurs.
    Gère la création, la mise à jour et la suppression des utilisateurs.
    """
    
    @staticmethod
    def get_db_connection():
        """Établit une connexion à la base de données"""
        return psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
    
    @staticmethod
    def init_db():
        """Initialise la base de données avec les tables nécessaires"""
        conn = None
        try:
            conn = UserManager.get_db_connection()
            with conn.cursor() as cur:
                # Table des utilisateurs
                cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(36) PRIMARY KEY,
                    username VARCHAR(100) UNIQUE,
                    email VARCHAR(255) UNIQUE,
                    password_hash VARCHAR(255),
                    telegram_id VARCHAR(100) UNIQUE,
                    subscription_type VARCHAR(20) NOT NULL DEFAULT 'free',
                    subscription_expiry TIMESTAMP,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """)
                
                # Table des paiements
                cur.execute("""
                CREATE TABLE IF NOT EXISTS payments (
                    id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(36) NOT NULL REFERENCES users(id),
                    amount DECIMAL(18, 8) NOT NULL,
                    currency VARCHAR(10) NOT NULL,
                    payment_address VARCHAR(255) NOT NULL,
                    transaction_hash VARCHAR(255),
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    subscription_type VARCHAR(20) NOT NULL,
                    subscription_duration_days INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    confirmed_at TIMESTAMP
                );
                """)
                
                conn.commit()
                logging.info("Base de données initialisée avec succès")
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation de la base de données: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def create_user(username: str, email: Optional[str] = None, password: Optional[str] = None, 
                   telegram_id: Optional[str] = None) -> Optional[User]:
        """
        Crée un nouvel utilisateur dans la base de données.
        
        Args:
            username: Nom d'utilisateur
            email: Adresse email (optionnel)
            password: Mot de passe (optionnel)
            telegram_id: Identifiant Telegram (optionnel)
            
        Returns:
            L'utilisateur créé ou None en cas d'erreur
        """
        conn = None
        try:
            user_id = str(uuid.uuid4())
            password_hash = generate_password_hash(password) if password else None
            
            conn = UserManager.get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO users (id, username, email, password_hash, telegram_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, username, email, telegram_id, subscription_type, subscription_expiry, is_active, created_at, updated_at;
                """, (user_id, username, email, password_hash, telegram_id))
                
                user_data = cur.fetchone()
                conn.commit()
                
                if user_data:
                    return User(
                        id=user_data[0],
                        username=user_data[1],
                        email=user_data[2],
                        telegram_id=user_data[3],
                        subscription_type=user_data[4],
                        subscription_expiry=user_data[5],
                        is_active=user_data[6]
                    )
                return None
        except Exception as e:
            logging.error(f"Erreur lors de la création de l'utilisateur: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[User]:
        """Récupère un utilisateur par son ID"""
        conn = None
        try:
            conn = UserManager.get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                SELECT id, username, email, telegram_id, subscription_type, subscription_expiry, is_active, created_at, updated_at
                FROM users
                WHERE id = %s;
                """, (user_id,))
                
                user_data = cur.fetchone()
                
                if user_data:
                    return User(
                        id=user_data[0],
                        username=user_data[1],
                        email=user_data[2],
                        telegram_id=user_data[3],
                        subscription_type=user_data[4],
                        subscription_expiry=user_data[5],
                        is_active=user_data[6],
                        created_at=user_data[7],
                        updated_at=user_data[8]
                    )
                return None
        except Exception as e:
            logging.error(f"Erreur lors de la récupération de l'utilisateur: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def get_user_by_telegram_id(telegram_id: str) -> Optional[User]:
        """Récupère un utilisateur par son ID Telegram"""
        conn = None
        try:
            conn = UserManager.get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                SELECT id, username, email, telegram_id, subscription_type, subscription_expiry, is_active, created_at, updated_at
                FROM users
                WHERE telegram_id = %s;
                """, (telegram_id,))
                
                user_data = cur.fetchone()
                
                if user_data:
                    return User(
                        id=user_data[0],
                        username=user_data[1],
                        email=user_data[2],
                        telegram_id=user_data[3],
                        subscription_type=user_data[4],
                        subscription_expiry=user_data[5],
                        is_active=user_data[6],
                        created_at=user_data[7],
                        updated_at=user_data[8]
                    )
                return None
        except Exception as e:
            logging.error(f"Erreur lors de la récupération de l'utilisateur par Telegram ID: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def update_user_subscription(user_id: str, subscription_type: str, 
                               duration_days: int) -> bool:
        """
        Met à jour l'abonnement d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            subscription_type: Type d'abonnement
            duration_days: Durée de l'abonnement en jours
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        conn = None
        try:
            conn = UserManager.get_db_connection()
            
            # Récupérer l'utilisateur actuel
            user = UserManager.get_user_by_id(user_id)
            if not user:
                return False
            
            # Calculer la nouvelle date d'expiration
            if user.is_subscription_active() and user.subscription_type == subscription_type:
                # Prolonger l'abonnement existant
                new_expiry = user.subscription_expiry + timedelta(days=duration_days)
            else:
                # Nouvel abonnement
                new_expiry = datetime.now() + timedelta(days=duration_days)
            
            with conn.cursor() as cur:
                cur.execute("""
                UPDATE users
                SET subscription_type = %s, subscription_expiry = %s, updated_at = NOW()
                WHERE id = %s;
                """, (subscription_type, new_expiry, user_id))
                
                conn.commit()
                return True
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour de l'abonnement: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def create_payment(user_id: str, amount: float, currency: str, 
                     payment_address: str, subscription_type: str, 
                     duration_days: int) -> Optional[str]:
        """
        Crée un nouveau paiement pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            amount: Montant du paiement
            currency: Devise (BTC, ETH, etc.)
            payment_address: Adresse de paiement
            subscription_type: Type d'abonnement
            duration_days: Durée de l'abonnement en jours
            
        Returns:
            ID du paiement créé ou None en cas d'erreur
        """
        conn = None
        try:
            payment_id = str(uuid.uuid4())
            
            conn = UserManager.get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO payments (id, user_id, amount, currency, payment_address, 
                                     status, subscription_type, subscription_duration_days)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """, (payment_id, user_id, amount, currency, payment_address, 
                     PaymentStatus.PENDING, subscription_type, duration_days))
                
                result = cur.fetchone()
                conn.commit()
                
                if result:
                    return result[0]
                return None
        except Exception as e:
            logging.error(f"Erreur lors de la création du paiement: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def confirm_payment(payment_id: str, transaction_hash: str) -> bool:
        """
        Confirme un paiement et met à jour l'abonnement de l'utilisateur.
        
        Args:
            payment_id: ID du paiement
            transaction_hash: Hash de la transaction
            
        Returns:
            True si la confirmation a réussi, False sinon
        """
        conn = None
        try:
            conn = UserManager.get_db_connection()
            
            # Récupérer les informations du paiement
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                SELECT user_id, subscription_type, subscription_duration_days
                FROM payments
                WHERE id = %s AND status = %s;
                """, (payment_id, PaymentStatus.PENDING))
                
                payment_data = cur.fetchone()
                
                if not payment_data:
                    return False
                
                # Mettre à jour le statut du paiement
                cur.execute("""
                UPDATE payments
                SET status = %s, transaction_hash = %s, confirmed_at = NOW(), updated_at = NOW()
                WHERE id = %s;
                """, (PaymentStatus.CONFIRMED, transaction_hash, payment_id))
                
                # Mettre à jour l'abonnement de l'utilisateur
                success = UserManager.update_user_subscription(
                    payment_data['user_id'],
                    payment_data['subscription_type'],
                    payment_data['subscription_duration_days']
                )
                
                if success:
                    conn.commit()
                    return True
                else:
                    conn.rollback()
                    return False
        except Exception as e:
            logging.error(f"Erreur lors de la confirmation du paiement: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def get_pending_payments(user_id: str) -> List[Dict[str, Any]]:
        """
        Récupère les paiements en attente pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Liste des paiements en attente
        """
        conn = None
        try:
            conn = UserManager.get_db_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                SELECT id, amount, currency, payment_address, status, 
                       subscription_type, subscription_duration_days, created_at
                FROM payments
                WHERE user_id = %s AND status = %s
                ORDER BY created_at DESC;
                """, (user_id, PaymentStatus.PENDING))
                
                return cur.fetchall()
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des paiements en attente: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def get_payment_by_id(payment_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère un paiement par son ID.
        
        Args:
            payment_id: ID du paiement
            
        Returns:
            Informations sur le paiement ou None en cas d'erreur
        """
        conn = None
        try:
            conn = UserManager.get_db_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                SELECT id, user_id, amount, currency, payment_address, transaction_hash,
                       status, subscription_type, subscription_duration_days, 
                       created_at, updated_at, confirmed_at
                FROM payments
                WHERE id = %s;
                """, (payment_id,))
                
                return cur.fetchone()
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du paiement: {e}")
            return None
        finally:
            if conn:
                conn.close() 