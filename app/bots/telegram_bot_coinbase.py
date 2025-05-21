"""
Intégration entre le bot Telegram et Coinbase Commerce pour les paiements.
Ce module ajoute des fonctionnalités de paiement via Coinbase Commerce au bot Telegram.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import qrcode
from io import BytesIO

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ContextTypes, ConversationHandler, CallbackQueryHandler,
    CommandHandler, MessageHandler, filters
)

from app.models.user import User, UserManager, SubscriptionType
from app.services.coinbase_payment import coinbase_payment_service

# Configuration du logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# États de conversation pour ConversationHandler
(
    CHOOSING_SUBSCRIPTION, CONFIRMING_PAYMENT, WAITING_FOR_PAYMENT,
    CHECKING_PAYMENT_STATUS
) = range(4)

# Callbacks des boutons
class CoinbaseCallbacks:
    SUBSCRIBE = "cb_subscribe"
    SUBSCRIPTION = "cb_subscription"
    CONFIRM_PAYMENT = "cb_confirm_payment"
    CHECK_PAYMENT = "cb_check_payment"
    CANCEL_PAYMENT = "cb_cancel_payment"
    BACK = "cb_back"

class TelegramCoinbaseIntegration:
    """
    Classe gérant l'intégration entre le bot Telegram et Coinbase Commerce.
    """
    
    def __init__(self):
        """Initialisation de l'intégration"""
        self.user_data = {}  # Stockage temporaire des données utilisateur
    
    def get_handlers(self):
        """
        Renvoie les gestionnaires pour l'intégration Coinbase.
        À utiliser pour ajouter ces gestionnaires à l'application Telegram.
        """
        # Gestionnaire de conversation pour l'abonnement via Coinbase
        subscription_conv_handler = ConversationHandler(
            entry_points=[
                CallbackQueryHandler(self.subscription_menu, pattern=f"^{CoinbaseCallbacks.SUBSCRIBE}$"),
            ],
            states={
                CHOOSING_SUBSCRIPTION: [
                    CallbackQueryHandler(
                        self.choose_subscription,
                        pattern=f"^{CoinbaseCallbacks.SUBSCRIPTION}_"
                    ),
                    CallbackQueryHandler(self.back_to_main, pattern=f"^{CoinbaseCallbacks.BACK}$")
                ],
                CONFIRMING_PAYMENT: [
                    CallbackQueryHandler(
                        self.create_coinbase_payment,
                        pattern=f"^{CoinbaseCallbacks.CONFIRM_PAYMENT}_"
                    ),
                    CallbackQueryHandler(self.subscription_menu, pattern=f"^{CoinbaseCallbacks.BACK}$")
                ],
                WAITING_FOR_PAYMENT: [
                    CallbackQueryHandler(
                        self.check_payment_status,
                        pattern=f"^{CoinbaseCallbacks.CHECK_PAYMENT}_"
                    ),
                    CallbackQueryHandler(
                        self.cancel_payment,
                        pattern=f"^{CoinbaseCallbacks.CANCEL_PAYMENT}$"
                    )
                ],
            },
            fallbacks=[
                CommandHandler("cancel", self.cancel_conversation),
                CommandHandler("start", self.back_to_main)
            ],
        )
        
        return [subscription_conv_handler]
    
    async def subscription_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Affiche le menu des abonnements"""
        query = update.callback_query
        await query.answer()
        
        # Récupérer l'utilisateur
        db_user = self._get_user_from_update(update)
        
        # Vérifier si l'utilisateur a déjà un abonnement actif
        if db_user and db_user.is_subscription_active() and db_user.subscription_type != SubscriptionType.FREE:
            # L'utilisateur a déjà un abonnement actif
            subscription_info = (
                f"💫 *Votre abonnement actuel (Coinbase Commerce)*\n\n"
                f"Type: *{db_user.subscription_type.capitalize()}*\n"
                f"Expire le: *{db_user.subscription_expiry.strftime('%d/%m/%Y')}*\n"
                f"Jours restants: *{db_user.get_subscription_days_left()}*\n\n"
                f"Souhaitez-vous prolonger votre abonnement ou changer de forfait?"
            )
            
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Prolonger", callback_data=f"{CoinbaseCallbacks.SUBSCRIPTION}_{db_user.subscription_type}"),
                    InlineKeyboardButton("🔝 Changer de forfait", callback_data=f"{CoinbaseCallbacks.SUBSCRIPTION}_change")
                ],
                [InlineKeyboardButton("◀️ Retour", callback_data=CoinbaseCallbacks.BACK)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(subscription_info, reply_markup=reply_markup, parse_mode='Markdown')
            
            if "subscription_change" in self.user_data.get(str(db_user.id), {}):
                return CHOOSING_SUBSCRIPTION
            else:
                return CHOOSING_SUBSCRIPTION
        
        else:
            # L'utilisateur n'a pas d'abonnement actif ou a un abonnement gratuit
            subscription_info = (
                "🚀 *Abonnez-vous pour accéder aux fonctionnalités premium (via Coinbase Commerce)*\n\n"
                "Choisissez l'un de nos forfaits adaptés à vos besoins:\n\n"
                "🔹 *Basic* - 10$/mois\n"
                "• Notifications de trading en temps réel\n"
                "• Accès à 5 paires de trading\n"
                "• Analyses quotidiennes\n\n"
                "🔹 *Premium* - 20$/mois\n"
                "• Tout ce qui est inclus dans Basic\n"
                "• Accès à 20 paires de trading\n"
                "• Alertes de marché personnalisées\n"
                "• Signaux de trading avancés\n\n"
                "🔹 *Enterprise* - 50$/mois\n"
                "• Tout ce qui est inclus dans Premium\n"
                "• Accès illimité à toutes les paires\n"
                "• Analyse approfondie par IA\n"
                "• Support prioritaire 24/7\n"
            )
            
            keyboard = [
                [InlineKeyboardButton("🔹 Basic", callback_data=f"{CoinbaseCallbacks.SUBSCRIPTION}_basic")],
                [InlineKeyboardButton("🔹 Premium", callback_data=f"{CoinbaseCallbacks.SUBSCRIPTION}_premium")],
                [InlineKeyboardButton("🔹 Enterprise", callback_data=f"{CoinbaseCallbacks.SUBSCRIPTION}_enterprise")],
                [InlineKeyboardButton("◀️ Retour", callback_data=CoinbaseCallbacks.BACK)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(subscription_info, reply_markup=reply_markup, parse_mode='Markdown')
            
            return CHOOSING_SUBSCRIPTION
    
    async def choose_subscription(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Gestionnaire pour le choix d'un abonnement"""
        query = update.callback_query
        await query.answer()
        
        user = self._get_user_from_update(update)
        if not user:
            await query.edit_message_text("Erreur: Utilisateur non trouvé. Veuillez redémarrer avec /start.")
            return ConversationHandler.END
        
        data = query.data.split('_')
        if len(data) < 2:
            return CHOOSING_SUBSCRIPTION
        
        subscription_type = data[1]
        
        if subscription_type == "change":
            # L'utilisateur veut changer de forfait
            user_id_str = str(user.id)
            self.user_data[user_id_str] = {"subscription_change": True}
            
            # Afficher les options d'abonnement
            subscription_info = (
                "🔄 *Changer de forfait (Coinbase Commerce)*\n\n"
                "Choisissez votre nouveau forfait:\n\n"
                "🔹 *Basic* - 10$/mois\n"
                "• Notifications de trading en temps réel\n"
                "• Accès à 5 paires de trading\n"
                "• Analyses quotidiennes\n\n"
                "🔹 *Premium* - 20$/mois\n"
                "• Tout ce qui est inclus dans Basic\n"
                "• Accès à 20 paires de trading\n"
                "• Alertes de marché personnalisées\n"
                "• Signaux de trading avancés\n\n"
                "🔹 *Enterprise* - 50$/mois\n"
                "• Tout ce qui est inclus dans Premium\n"
                "• Accès illimité à toutes les paires\n"
                "• Analyse approfondie par IA\n"
                "• Support prioritaire 24/7\n"
            )
            
            keyboard = [
                [InlineKeyboardButton("🔹 Basic", callback_data=f"{CoinbaseCallbacks.SUBSCRIPTION}_basic")],
                [InlineKeyboardButton("🔹 Premium", callback_data=f"{CoinbaseCallbacks.SUBSCRIPTION}_premium")],
                [InlineKeyboardButton("🔹 Enterprise", callback_data=f"{CoinbaseCallbacks.SUBSCRIPTION}_enterprise")],
                [InlineKeyboardButton("◀️ Retour", callback_data=CoinbaseCallbacks.BACK)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(subscription_info, reply_markup=reply_markup, parse_mode='Markdown')
            
            return CHOOSING_SUBSCRIPTION
        
        # Stocker le type d'abonnement choisi
        user_id_str = str(user.id)
        if not user_id_str in self.user_data:
            self.user_data[user_id_str] = {}
        
        self.user_data[user_id_str]['subscription_type'] = subscription_type
        
        # Récupérer le prix et la durée de l'abonnement
        price = coinbase_payment_service.get_subscription_price(subscription_type)
        duration = coinbase_payment_service.get_subscription_duration(subscription_type)
        
        self.user_data[user_id_str]['price'] = price
        self.user_data[user_id_str]['duration'] = duration
        
        # Afficher un récapitulatif avant de créer un paiement
        payment_summary = (
            f"💳 *Récapitulatif de votre abonnement (Coinbase Commerce)*\n\n"
            f"Type: *{subscription_type.capitalize()}*\n"
            f"Prix: *{price} USD*\n"
            f"Durée: *{duration} jours*\n\n"
            f"En confirmant, vous serez redirigé vers Coinbase Commerce pour effectuer votre paiement en cryptomonnaie.\n"
            f"Vous pourrez payer en Bitcoin, Ethereum, et d'autres crypto-monnaies populaires."
        )
        
        keyboard = [
            [InlineKeyboardButton("✅ Confirmer", callback_data=f"{CoinbaseCallbacks.CONFIRM_PAYMENT}_{subscription_type}")],
            [InlineKeyboardButton("◀️ Retour", callback_data=CoinbaseCallbacks.BACK)]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(payment_summary, reply_markup=reply_markup, parse_mode='Markdown')
        
        return CONFIRMING_PAYMENT
    
    async def create_coinbase_payment(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Crée un paiement via Coinbase Commerce"""
        query = update.callback_query
        await query.answer()
        
        user = self._get_user_from_update(update)
        if not user:
            await query.edit_message_text("Erreur: Utilisateur non trouvé. Veuillez redémarrer avec /start.")
            return ConversationHandler.END
        
        data = query.data.split('_')
        if len(data) < 2:
            return CONFIRMING_PAYMENT
        
        subscription_type = data[1]
        user_id_str = str(user.id)
        
        # Créer une charge via Coinbase Commerce
        metadata = {
            "telegram_id": str(update.effective_user.id),
            "username": update.effective_user.username or "unknown",
            "subscription_change": self.user_data.get(user_id_str, {}).get("subscription_change", False)
        }
        
        success, charge = coinbase_payment_service.create_charge(user.id, subscription_type, metadata)
        
        if not success:
            await query.edit_message_text(
                "❌ Une erreur s'est produite lors de la création du paiement. "
                "Veuillez réessayer plus tard ou contacter le support."
            )
            return ConversationHandler.END
        
        # Stocker les informations de la charge
        self.user_data[user_id_str]['charge_id'] = charge['id']
        self.user_data[user_id_str]['created_at'] = datetime.now().isoformat()
        
        # Récupérer l'URL de paiement hébergée par Coinbase
        hosted_url = charge.get('hosted_url', '')
        
        # Afficher les informations de paiement
        payment_info = (
            f"🧾 *Détails du paiement Coinbase Commerce*\n\n"
            f"Abonnement: *{subscription_type.capitalize()}*\n"
            f"Montant: *{self.user_data[user_id_str]['price']} USD*\n"
            f"Durée: *{self.user_data[user_id_str]['duration']} jours*\n\n"
            f"Veuillez cliquer sur le bouton ci-dessous pour procéder au paiement via Coinbase Commerce. "
            f"Vous pouvez payer avec plusieurs cryptomonnaies.\n\n"
            f"Une fois le paiement effectué, revenez ici et cliquez sur \"Vérifier le paiement\"."
        )
        
        keyboard = [
            [InlineKeyboardButton("💰 Payer maintenant", url=hosted_url)],
            [InlineKeyboardButton("🔄 Vérifier le paiement", callback_data=f"{CoinbaseCallbacks.CHECK_PAYMENT}_{charge['id']}")],
            [InlineKeyboardButton("❌ Annuler", callback_data=CoinbaseCallbacks.CANCEL_PAYMENT)]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(payment_info, reply_markup=reply_markup, parse_mode='Markdown')
        
        return WAITING_FOR_PAYMENT
    
    async def check_payment_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Vérifie le statut d'un paiement Coinbase Commerce"""
        query = update.callback_query
        await query.answer()
        
        user = self._get_user_from_update(update)
        if not user:
            await query.edit_message_text("Erreur: Utilisateur non trouvé. Veuillez redémarrer avec /start.")
            return ConversationHandler.END
        
        data = query.data.split('_')
        if len(data) < 2:
            return WAITING_FOR_PAYMENT
        
        charge_id = data[1]
        user_id_str = str(user.id)
        
        # Vérifier que c'est bien la charge de cet utilisateur
        if user_id_str not in self.user_data or self.user_data[user_id_str].get('charge_id') != charge_id:
            await query.edit_message_text("Erreur: Paiement non trouvé pour cet utilisateur.")
            return ConversationHandler.END
        
        # Vérifier le statut de la charge
        success, status, charge_data = coinbase_payment_service.verify_charge_status(charge_id)
        
        if not success:
            await query.edit_message_text(
                "❌ Une erreur s'est produite lors de la vérification du paiement. "
                "Veuillez réessayer plus tard ou contacter le support."
            )
            return ConversationHandler.END
        
        # Traiter en fonction du statut
        if status == "completed":
            # Paiement confirmé - mettre à jour l'abonnement
            subscription_type = self.user_data[user_id_str]['subscription_type']
            duration = self.user_data[user_id_str]['duration']
            
            # Mettre à jour l'abonnement dans la base de données
            from app.api.payment_webhooks import update_user_subscription
            success = update_user_subscription(user.id, subscription_type, duration)
            
            if success:
                # Récupérer l'utilisateur mis à jour
                updated_user = UserManager.get_user_by_id(user.id)
                
                # Envoyer la confirmation de paiement
                success_message = (
                    f"✅ *Paiement confirmé avec succès!*\n\n"
                    f"Merci pour votre abonnement *{updated_user.subscription_type.capitalize()}*.\n"
                    f"Votre abonnement est actif jusqu'au *{updated_user.subscription_expiry.strftime('%d/%m/%Y')}*.\n\n"
                    f"Vous avez maintenant accès à toutes les fonctionnalités premium de EVIL2ROOT Trading Bot."
                )
                
                keyboard = [
                    [InlineKeyboardButton("🏠 Menu principal", callback_data=CoinbaseCallbacks.BACK)]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(success_message, reply_markup=reply_markup, parse_mode='Markdown')
                
                # Nettoyer les données utilisateur
                if user_id_str in self.user_data:
                    del self.user_data[user_id_str]
                
                return ConversationHandler.END
            else:
                await query.edit_message_text(
                    "❌ Une erreur s'est produite lors de la mise à jour de l'abonnement. "
                    "Veuillez contacter le support."
                )
                return ConversationHandler.END
        
        elif status == "pending":
            # Paiement en attente
            payment_info = (
                f"⏳ *Paiement en cours de traitement*\n\n"
                f"Votre paiement a été détecté et est en cours de traitement. "
                f"Veuillez patienter quelques instants jusqu'à confirmation complète.\n\n"
                f"Vous pouvez vérifier à nouveau l'état du paiement en cliquant sur le bouton ci-dessous."
            )
            
            keyboard = [
                [InlineKeyboardButton("🔄 Vérifier à nouveau", callback_data=f"{CoinbaseCallbacks.CHECK_PAYMENT}_{charge_id}")],
                [InlineKeyboardButton("❌ Annuler", callback_data=CoinbaseCallbacks.CANCEL_PAYMENT)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(payment_info, reply_markup=reply_markup, parse_mode='Markdown')
            
            return WAITING_FOR_PAYMENT
        
        elif status in ["canceled", "expired"]:
            # Paiement annulé ou expiré
            payment_info = (
                f"❌ *Paiement {status}*\n\n"
                f"Votre demande de paiement a été {status}.\n"
                f"Vous pouvez réessayer ou choisir un autre type d'abonnement."
            )
            
            keyboard = [
                [InlineKeyboardButton("🔄 Réessayer", callback_data=CoinbaseCallbacks.SUBSCRIBE)],
                [InlineKeyboardButton("◀️ Retour", callback_data=CoinbaseCallbacks.BACK)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(payment_info, reply_markup=reply_markup, parse_mode='Markdown')
            
            # Nettoyer les données utilisateur
            if user_id_str in self.user_data:
                del self.user_data[user_id_str]
            
            return ConversationHandler.END
        
        else:
            # Statut inconnu ou nouveau - demander de vérifier à nouveau
            payment_info = (
                f"ℹ️ *Statut du paiement: {status}*\n\n"
                f"Votre paiement n'a pas encore été confirmé.\n"
                f"Si vous avez déjà effectué le paiement, veuillez patienter quelques instants "
                f"puis vérifier à nouveau.\n\n"
                f"Vous pouvez également cliquer sur le bouton \"Payer maintenant\" pour accéder "
                f"à la page de paiement Coinbase Commerce."
            )
            
            # Récupérer l'URL de paiement hébergée par Coinbase
            hosted_url = charge_data.get('hosted_url', '')
            
            keyboard = [
                [InlineKeyboardButton("💰 Payer maintenant", url=hosted_url)],
                [InlineKeyboardButton("🔄 Vérifier à nouveau", callback_data=f"{CoinbaseCallbacks.CHECK_PAYMENT}_{charge_id}")],
                [InlineKeyboardButton("❌ Annuler", callback_data=CoinbaseCallbacks.CANCEL_PAYMENT)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(payment_info, reply_markup=reply_markup, parse_mode='Markdown')
            
            return WAITING_FOR_PAYMENT
    
    async def cancel_payment(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Annule un paiement en cours"""
        query = update.callback_query
        await query.answer()
        
        user = self._get_user_from_update(update)
        if user and str(user.id) in self.user_data:
            del self.user_data[str(user.id)]
        
        await query.edit_message_text(
            "❌ Paiement annulé. Vous pouvez réessayer quand vous voulez."
        )
        
        return ConversationHandler.END
    
    async def back_to_main(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Retourne au menu principal"""
        query = update.callback_query
        await query.answer()
        
        # Envoyer un message pour revenir au menu principal
        await query.edit_message_text(
            "🏠 Retour au menu principal. Utilisez /start pour afficher le menu."
        )
        
        return ConversationHandler.END
    
    async def cancel_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Annule la conversation en cours"""
        if update.callback_query:
            await update.callback_query.answer()
            await update.callback_query.edit_message_text("❌ Opération annulée.")
        else:
            await update.message.reply_text("❌ Opération annulée.")
        
        # Nettoyer les données utilisateur
        user = self._get_user_from_update(update)
        if user and str(user.id) in self.user_data:
            del self.user_data[str(user.id)]
        
        return ConversationHandler.END
    
    def _get_user_from_update(self, update: Update) -> Optional[User]:
        """
        Récupère l'utilisateur depuis la base de données en fonction de la mise à jour Telegram.
        
        Args:
            update: Mise à jour Telegram
            
        Returns:
            Utilisateur trouvé ou None
        """
        telegram_id = str(update.effective_user.id)
        return UserManager.get_user_by_telegram_id(telegram_id)

# Instance globale de l'intégration Coinbase
telegram_coinbase_integration = TelegramCoinbaseIntegration() 