"""
Service de notification pour les utilisateurs premium.
Ce module gère l'envoi de notifications aux utilisateurs abonnés via Telegram.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from telegram import Bot, InlineKeyboardMarkup, InlineKeyboardButton
from dotenv import load_dotenv

from app.models.user import User, UserManager, SubscriptionType

# Charger les variables d'environnement
load_dotenv()

# Configuration du logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Récupérer le token Telegram depuis les variables d'environnement
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
if not TELEGRAM_TOKEN:
    logger.warning("Token Telegram non configuré. Les notifications ne fonctionneront pas.")

class NotificationService:
    """
    Service de notification pour les utilisateurs premium.
    Gère l'envoi de notifications aux utilisateurs abonnés via Telegram.
    """
    
    def __init__(self):
        """Initialise le service de notification"""
        self.token = TELEGRAM_TOKEN
        self.bot = Bot(token=self.token) if self.token else None
    
    async def send_notification(self, telegram_id: str, message: str, 
                              reply_markup: Optional[InlineKeyboardMarkup] = None) -> bool:
        """
        Envoie une notification à un utilisateur.
        
        Args:
            telegram_id: ID Telegram de l'utilisateur
            message: Message à envoyer
            reply_markup: Boutons à afficher (optionnel)
            
        Returns:
            True si l'envoi a réussi, False sinon
        """
        if not self.bot:
            logger.error("Bot Telegram non initialisé. Impossible d'envoyer la notification.")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=telegram_id,
                text=message,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de la notification à {telegram_id}: {e}")
            return False
    
    async def send_bulk_notification(self, message: str, subscription_type: Optional[str] = None, 
                                   reply_markup: Optional[InlineKeyboardMarkup] = None) -> Dict[str, bool]:
        """
        Envoie une notification à plusieurs utilisateurs.
        
        Args:
            message: Message à envoyer
            subscription_type: Type d'abonnement cible (optionnel, si None, tous les utilisateurs actifs)
            reply_markup: Boutons à afficher (optionnel)
            
        Returns:
            Dictionnaire avec les IDs des utilisateurs et le statut d'envoi
        """
        if not self.bot:
            logger.error("Bot Telegram non initialisé. Impossible d'envoyer les notifications.")
            return {}
        
        # Récupérer tous les utilisateurs actifs avec un abonnement valide
        users = self._get_active_users(subscription_type)
        
        results = {}
        for user in users:
            if user.telegram_id:
                success = await self.send_notification(user.telegram_id, message, reply_markup)
                results[user.telegram_id] = success
        
        return results
    
    async def send_signal_notification(self, symbol: str, direction: str, entry_price: float, 
                                     stop_loss: float, take_profit: float, 
                                     confidence: float, subscription_type: Optional[str] = None) -> Dict[str, bool]:
        """
        Envoie une notification de signal de trading.
        
        Args:
            symbol: Symbole du marché
            direction: Direction du trade (LONG/SHORT)
            entry_price: Prix d'entrée
            stop_loss: Prix du stop loss
            take_profit: Prix du take profit
            confidence: Niveau de confiance (0-1)
            subscription_type: Type d'abonnement cible (optionnel)
            
        Returns:
            Dictionnaire avec les IDs des utilisateurs et le statut d'envoi
        """
        # Formater le message
        emoji = "🟢" if direction.upper() == "LONG" else "🔴"
        confidence_pct = round(confidence * 100, 2)
        
        message = (
            f"{emoji} *Signal de trading: {symbol}*\n\n"
            f"Type: *{direction.upper()}*\n"
            f"Prix d'entrée: *{entry_price}*\n"
            f"Stop Loss: *{stop_loss}*\n"
            f"Take Profit: *{take_profit}*\n"
            f"Confiance: *{confidence_pct}%*\n\n"
            f"Signal généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}"
        )
        
        # Ajouter des boutons
        keyboard = [
            [
                InlineKeyboardButton("📊 Voir l'analyse", callback_data=f"analysis_{symbol}"),
                InlineKeyboardButton("📱 Ouvrir WebApp", url="https://evil2root-ai.com/webapp")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Envoyer la notification
        return await self.send_bulk_notification(message, subscription_type, reply_markup)
    
    async def send_market_update(self, symbol: str, price: float, change_24h: float, 
                               analysis: str, subscription_type: Optional[str] = None) -> Dict[str, bool]:
        """
        Envoie une mise à jour du marché.
        
        Args:
            symbol: Symbole du marché
            price: Prix actuel
            change_24h: Changement en 24h (%)
            analysis: Analyse courte du marché
            subscription_type: Type d'abonnement cible (optionnel)
            
        Returns:
            Dictionnaire avec les IDs des utilisateurs et le statut d'envoi
        """
        # Formater le message
        emoji = "🟢" if change_24h >= 0 else "🔴"
        
        message = (
            f"📊 *Mise à jour du marché: {symbol}*\n\n"
            f"Prix: *{price}*\n"
            f"Variation 24h: *{change_24h}%* {emoji}\n\n"
            f"*Analyse:*\n{analysis}\n\n"
            f"Mise à jour le {datetime.now().strftime('%d/%m/%Y à %H:%M')}"
        )
        
        # Ajouter des boutons
        keyboard = [
            [
                InlineKeyboardButton("📊 Voir le graphique", callback_data=f"chart_{symbol}"),
                InlineKeyboardButton("📱 Ouvrir WebApp", url="https://evil2root-ai.com/webapp")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Envoyer la notification
        return await self.send_bulk_notification(message, subscription_type, reply_markup)
    
    def _get_active_users(self, subscription_type: Optional[str] = None) -> List[User]:
        """
        Récupère les utilisateurs actifs avec un abonnement valide.
        
        Args:
            subscription_type: Type d'abonnement (optionnel)
            
        Returns:
            Liste des utilisateurs actifs
        """
        # Pour l'exemple, nous simulons cette fonction
        # Dans une implémentation réelle, vous devriez interroger la base de données
        
        # Création d'une connexion à la BD
        conn = None
        users = []
        
        try:
            conn = UserManager.get_db_connection()
            with conn.cursor() as cur:
                # Requête pour récupérer les utilisateurs actifs avec un abonnement valide
                query = """
                SELECT id, username, email, telegram_id, subscription_type, subscription_expiry, is_active
                FROM users
                WHERE is_active = TRUE AND telegram_id IS NOT NULL
                AND (subscription_type = 'free' OR (subscription_expiry > NOW()))
                """
                
                # Ajouter un filtre par type d'abonnement si nécessaire
                if subscription_type and subscription_type != SubscriptionType.FREE:
                    query += " AND subscription_type = %s"
                    cur.execute(query, (subscription_type,))
                else:
                    # Exclure les utilisateurs avec un abonnement gratuit si aucun type n'est spécifié
                    query += " AND subscription_type != 'free'"
                    cur.execute(query)
                
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
            logger.error(f"Erreur lors de la récupération des utilisateurs actifs: {e}")
        finally:
            if conn:
                conn.close()
        
        return users

# Instance globale du service de notification
notification_service = NotificationService() 