"""
Service de gestion des abonnements.
Ce module gère la vérification et le renouvellement des abonnements.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta

from app.models.user import User, UserManager, SubscriptionType
from app.services.notification_service import notification_service

# Configuration du logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SubscriptionManager:
    """
    Service de gestion des abonnements.
    Gère la vérification et le renouvellement des abonnements.
    """
    
    @staticmethod
    async def check_expiring_subscriptions(days_threshold: int = 3) -> None:
        """
        Vérifie les abonnements qui expirent bientôt et envoie des notifications.
        
        Args:
            days_threshold: Nombre de jours avant l'expiration pour envoyer une notification
        """
        # Récupérer les abonnements qui expirent bientôt
        expiring_users = SubscriptionManager._get_expiring_users(days_threshold)
        
        for user in expiring_users:
            if user.telegram_id:
                # Calculer le nombre de jours restants
                days_left = user.get_subscription_days_left()
                
                # Préparer le message
                message = (
                    f"⚠️ *Votre abonnement expire bientôt*\n\n"
                    f"Bonjour, votre abonnement *{user.subscription_type.capitalize()}* "
                    f"expire dans *{days_left} jour{'s' if days_left > 1 else ''}*.\n\n"
                    f"Pour continuer à bénéficier de toutes les fonctionnalités premium, "
                    f"veuillez renouveler votre abonnement."
                )
                
                # Ajouter des boutons pour le renouvellement
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                keyboard = [
                    [InlineKeyboardButton("🔄 Renouveler mon abonnement", callback_data="subscribe")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Envoyer la notification
                await notification_service.send_notification(user.telegram_id, message, reply_markup)
                
                logger.info(f"Notification d'expiration envoyée à l'utilisateur {user.username} (ID: {user.id})")
    
    @staticmethod
    async def check_expired_subscriptions() -> None:
        """
        Vérifie les abonnements qui ont expiré et envoie des notifications.
        """
        # Récupérer les abonnements qui ont expiré
        expired_users = SubscriptionManager._get_expired_users()
        
        for user in expired_users:
            if user.telegram_id:
                # Préparer le message
                message = (
                    f"❌ *Votre abonnement a expiré*\n\n"
                    f"Bonjour, votre abonnement *{user.subscription_type.capitalize()}* a expiré.\n\n"
                    f"Vous n'avez plus accès aux fonctionnalités premium. "
                    f"Pour les réactiver, veuillez souscrire à un nouvel abonnement."
                )
                
                # Ajouter des boutons pour le renouvellement
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                keyboard = [
                    [InlineKeyboardButton("💰 S'abonner à nouveau", callback_data="subscribe")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Envoyer la notification
                await notification_service.send_notification(user.telegram_id, message, reply_markup)
                
                logger.info(f"Notification d'expiration envoyée à l'utilisateur {user.username} (ID: {user.id})")
    
    @staticmethod
    def _get_expiring_users(days_threshold: int) -> List[User]:
        """
        Récupère les utilisateurs dont l'abonnement expire bientôt.
        
        Args:
            days_threshold: Nombre de jours avant l'expiration
            
        Returns:
            Liste des utilisateurs dont l'abonnement expire bientôt
        """
        conn = None
        users = []
        
        try:
            conn = UserManager.get_db_connection()
            with conn.cursor() as cur:
                # Date limite d'expiration (aujourd'hui + days_threshold)
                expiry_date = datetime.now() + timedelta(days=days_threshold)
                
                # Requête pour récupérer les utilisateurs dont l'abonnement expire bientôt
                cur.execute("""
                SELECT id, username, email, telegram_id, subscription_type, subscription_expiry, is_active
                FROM users
                WHERE is_active = TRUE 
                AND telegram_id IS NOT NULL
                AND subscription_type != 'free'
                AND subscription_expiry <= %s
                AND subscription_expiry > NOW()
                """, (expiry_date,))
                
                # Traiter les résultats
                for row in cur.fetchall():
                    user = User(
                        id=row[0],
                        username=row[1],
                        email=row[2],
                        telegram_id=row[3],
                        subscription_type=row[4],
                        subscription_expiry=row[5],
                        is_active=row[6]
                    )
                    users.append(user)
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des utilisateurs dont l'abonnement expire bientôt: {e}")
        finally:
            if conn:
                conn.close()
        
        return users
    
    @staticmethod
    def _get_expired_users() -> List[User]:
        """
        Récupère les utilisateurs dont l'abonnement a expiré.
        
        Returns:
            Liste des utilisateurs dont l'abonnement a expiré
        """
        conn = None
        users = []
        
        try:
            conn = UserManager.get_db_connection()
            with conn.cursor() as cur:
                # Requête pour récupérer les utilisateurs dont l'abonnement a expiré
                cur.execute("""
                SELECT id, username, email, telegram_id, subscription_type, subscription_expiry, is_active
                FROM users
                WHERE is_active = TRUE 
                AND telegram_id IS NOT NULL
                AND subscription_type != 'free'
                AND subscription_expiry < NOW()
                AND subscription_expiry > NOW() - INTERVAL '1 day'  -- Expiré depuis moins d'un jour
                """)
                
                # Traiter les résultats
                for row in cur.fetchall():
                    user = User(
                        id=row[0],
                        username=row[1],
                        email=row[2],
                        telegram_id=row[3],
                        subscription_type=row[4],
                        subscription_expiry=row[5],
                        is_active=row[6]
                    )
                    users.append(user)
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des utilisateurs dont l'abonnement a expiré: {e}")
        finally:
            if conn:
                conn.close()
        
        return users

# Instance globale du gestionnaire d'abonnements
subscription_manager = SubscriptionManager() 