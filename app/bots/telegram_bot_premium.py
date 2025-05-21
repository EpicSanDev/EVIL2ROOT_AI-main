"""
Bot Telegram Premium pour EVIL2ROOT Trading Bot.
Permet aux utilisateurs de s'inscrire, de gérer leur abonnement et de recevoir des notifications.
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
import qrcode
from io import BytesIO
import base64

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, MessageHandler, 
    filters, ContextTypes, ConversationHandler
)
from dotenv import load_dotenv

# Import des modules personnalisés
from app.models.user import User, UserManager, SubscriptionType, PaymentStatus
from app.services.crypto_payment import CryptoPaymentService
# Nouvel import pour l'intégration Coinbase Commerce
from app.telegram_bot_coinbase import telegram_coinbase_integration, CoinbaseCallbacks

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
    raise ValueError("Token Telegram non configuré. Veuillez définir TELEGRAM_TOKEN dans le fichier .env")

# États de conversation pour ConversationHandler
(
    MAIN_MENU, CHOOSING_SUBSCRIPTION, CHOOSING_CURRENCY,
    WAITING_FOR_PAYMENT, WAITING_FOR_TX_HASH, WAITING_FOR_EMAIL
) = range(6)

# Callbacks des boutons
class CallbackData:
    SUBSCRIBE = "subscribe"
    ACCOUNT = "account"
    SUBSCRIPTION = "subscription"
    CURRENCY = "currency"
    CONFIRM_PAYMENT = "confirm_payment"
    CANCEL_PAYMENT = "cancel_payment"
    SUBMIT_TX = "submit_tx"
    BACK = "back"
    # Nouveau callback pour Coinbase Commerce
    COINBASE_PAYMENT = "coinbase_payment"

class TelegramBotPremium:
    """
    Bot Telegram Premium pour EVIL2ROOT Trading Bot.
    Permet aux utilisateurs de s'inscrire, de gérer leur abonnement et de recevoir des notifications.
    """
    
    def __init__(self):
        """Initialise le bot Telegram Premium"""
        self.token = TELEGRAM_TOKEN
        self.application = Application.builder().token(self.token).build()
        self.bot = Bot(token=self.token)
        
        # Données temporaires de session utilisateur
        self.user_data = {}
        
        # Initialiser la base de données au démarrage
        UserManager.init_db()
        
        # Configurer les gestionnaires de commandes
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Configure les gestionnaires de commandes et de callbacks"""
        
        # Gestionnaires de commandes
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        
        # Gestionnaire pour le menu principal
        self.application.add_handler(CallbackQueryHandler(self.account_info, pattern=f"^{CallbackData.ACCOUNT}$"))
        self.application.add_handler(CallbackQueryHandler(self.start_over, pattern=f"^{CallbackData.BACK}$"))
        
        # Gestionnaire de conversation pour l'abonnement
        subscription_conv_handler = ConversationHandler(
            entry_points=[
                CallbackQueryHandler(self.subscription_menu, pattern=f"^{CallbackData.SUBSCRIBE}$"),
            ],
            states={
                CHOOSING_SUBSCRIPTION: [
                    CallbackQueryHandler(
                        self.choose_subscription,
                        pattern=f"^{CallbackData.SUBSCRIPTION}_"
                    ),
                    CallbackQueryHandler(
                        self.coinbase_payment,
                        pattern=f"^{CallbackData.COINBASE_PAYMENT}$"
                    ),
                    CallbackQueryHandler(self.start_over, pattern=f"^{CallbackData.BACK}$")
                ],
                CHOOSING_CURRENCY: [
                    CallbackQueryHandler(
                        self.choose_currency,
                        pattern=f"^{CallbackData.CURRENCY}_"
                    ),
                    CallbackQueryHandler(self.subscription_menu, pattern=f"^{CallbackData.BACK}$")
                ],
                WAITING_FOR_PAYMENT: [
                    CallbackQueryHandler(
                        self.confirm_payment,
                        pattern=f"^{CallbackData.CONFIRM_PAYMENT}_"
                    ),
                    CallbackQueryHandler(
                        self.cancel_payment,
                        pattern=f"^{CallbackData.CANCEL_PAYMENT}$"
                    )
                ],
                WAITING_FOR_TX_HASH: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.process_transaction_hash),
                    CallbackQueryHandler(self.cancel_payment, pattern=f"^{CallbackData.CANCEL_PAYMENT}$")
                ],
            },
            fallbacks=[
                CommandHandler("cancel", self.cancel_conversation),
                CommandHandler("start", self.start_command)
            ],
        )
        self.application.add_handler(subscription_conv_handler)
        
        # Ajouter les gestionnaires pour Coinbase Commerce
        for handler in telegram_coinbase_integration.get_handlers():
            self.application.add_handler(handler)
            
        # Gestionnaires par défaut
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.text_message))
        self.application.add_handler(MessageHandler(filters.COMMAND, self.unknown_command))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Gestionnaire de la commande /start"""
        user = update.effective_user
        telegram_id = str(user.id)
        username = user.username or f"user_{telegram_id}"
        
        # Vérifier si l'utilisateur existe déjà
        db_user = UserManager.get_user_by_telegram_id(telegram_id)
        
        if not db_user:
            # Créer un nouvel utilisateur
            db_user = UserManager.create_user(
                username=username,
                telegram_id=telegram_id
            )
            welcome_message = (
                f"👋 Bienvenue sur EVIL2ROOT Trading Bot, {user.first_name}!\n\n"
                f"Je suis votre assistant pour accéder aux services premium de trading. "
                f"Pour accéder aux fonctionnalités avancées et aux notifications en temps réel, "
                f"vous devez vous abonner à l'un de nos plans."
            )
        else:
            welcome_message = (
                f"👋 Ravi de vous revoir, {user.first_name}!\n\n"
                f"Que souhaitez-vous faire aujourd'hui?"
            )
        
        # Créer les boutons du menu principal
        keyboard = [
            [
                InlineKeyboardButton("💰 S'abonner", callback_data=CallbackData.SUBSCRIBE),
                InlineKeyboardButton("👤 Mon compte", callback_data=CallbackData.ACCOUNT)
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Envoyer le message de bienvenue avec les boutons
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)
        
        return MAIN_MENU
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gestionnaire de la commande /help"""
        help_text = (
            "🤖 *EVIL2ROOT Trading Bot - Aide*\n\n"
            "Voici les commandes disponibles:\n\n"
            "/start - Démarrer le bot et afficher le menu principal\n"
            "/help - Afficher ce message d'aide\n"
            "/cancel - Annuler l'opération en cours\n\n"
            "Pour accéder aux fonctionnalités premium, vous devez vous abonner via le menu principal."
        )
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gestionnaire pour les messages non traités"""
        await update.message.reply_text(
            "Je ne comprends pas cette commande. Utilisez /start pour afficher le menu principal ou /help pour voir l'aide."
        )
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Gestionnaire d'erreurs"""
        logger.error(f"Erreur: {context.error} - Update: {update}")
        
        # Informer l'utilisateur qu'une erreur s'est produite
        if update and update.effective_chat:
            await self.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Une erreur s'est produite. Veuillez réessayer plus tard ou contacter le support."
            )
    
    async def start_over(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Retour au menu principal"""
        query = update.callback_query
        await query.answer()
        
        # Afficher le menu principal
        keyboard = [
            [
                InlineKeyboardButton("💰 S'abonner", callback_data=CallbackData.SUBSCRIBE),
                InlineKeyboardButton("👤 Mon compte", callback_data=CallbackData.ACCOUNT)
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text="Que souhaitez-vous faire?",
            reply_markup=reply_markup
        )
        
        return MAIN_MENU
    
    async def cancel_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Annuler la conversation en cours"""
        user = update.effective_user
        logger.info(f"L'utilisateur {user.id} a annulé la conversation.")
        
        await update.message.reply_text(
            "Opération annulée. Utilisez /start pour revenir au menu principal."
        )
        
        return ConversationHandler.END
    
    # Fonctions utilitaires
    def _generate_qr_code(self, data: str) -> str:
        """
        Génère un QR code à partir d'une chaîne de données.
        
        Args:
            data: Données à encoder dans le QR code
            
        Returns:
            QR code encodé en base64
        """
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffered = BytesIO()
        img.save(buffered)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    
    def _get_user_from_update(self, update: Update) -> Optional[User]:
        """
        Récupère l'utilisateur à partir de l'objet Update.
        
        Args:
            update: Objet Update de Telegram
            
        Returns:
            Objet User ou None si l'utilisateur n'existe pas
        """
        user = update.effective_user
        telegram_id = str(user.id)
        
        return UserManager.get_user_by_telegram_id(telegram_id)
    
    async def _send_notification(self, chat_id: str, message: str, reply_markup: InlineKeyboardMarkup = None) -> None:
        """
        Envoie une notification à un utilisateur.
        
        Args:
            chat_id: ID du chat
            message: Message à envoyer
            reply_markup: Boutons à afficher (optionnel)
        """
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de la notification: {e}")
    
    # Fonctions de gestion des abonnements
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
                f"💫 *Votre abonnement actuel*\n\n"
                f"Type: *{db_user.subscription_type.capitalize()}*\n"
                f"Expire le: *{db_user.subscription_expiry.strftime('%d/%m/%Y')}*\n"
                f"Jours restants: *{db_user.get_subscription_days_left()}*\n\n"
                f"Souhaitez-vous prolonger votre abonnement ou changer de forfait?"
            )
            
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Prolonger", callback_data=f"{CallbackData.SUBSCRIPTION}_{db_user.subscription_type}"),
                    InlineKeyboardButton("🔝 Changer de forfait", callback_data=f"{CallbackData.SUBSCRIPTION}_change")
                ],
                [InlineKeyboardButton("◀️ Retour", callback_data=CallbackData.BACK)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(subscription_info, reply_markup=reply_markup, parse_mode='Markdown')
            
            if "subscription_change" in self.user_data.get(db_user.id, {}):
                return CHOOSING_SUBSCRIPTION
            else:
                return CHOOSING_CURRENCY
        
        else:
            # L'utilisateur n'a pas d'abonnement actif ou a un abonnement gratuit
            subscription_info = (
                "🚀 *Abonnez-vous pour accéder aux fonctionnalités premium*\n\n"
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
                [InlineKeyboardButton("🔹 Basic", callback_data=f"{CallbackData.SUBSCRIPTION}_basic")],
                [InlineKeyboardButton("🔹 Premium", callback_data=f"{CallbackData.SUBSCRIPTION}_premium")],
                [InlineKeyboardButton("🔹 Enterprise", callback_data=f"{CallbackData.SUBSCRIPTION}_enterprise")],
                [InlineKeyboardButton("💳 Payer avec Coinbase", callback_data=f"{CallbackData.COINBASE_PAYMENT}")],
                [InlineKeyboardButton("◀️ Retour", callback_data=CallbackData.BACK)]
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
            self.user_data[user.id] = {"subscription_change": True}
            
            # Afficher les options d'abonnement
            subscription_info = (
                "🔄 *Changer de forfait*\n\n"
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
                [InlineKeyboardButton("🔹 Basic", callback_data=f"{CallbackData.SUBSCRIPTION}_basic")],
                [InlineKeyboardButton("🔹 Premium", callback_data=f"{CallbackData.SUBSCRIPTION}_premium")],
                [InlineKeyboardButton("🔹 Enterprise", callback_data=f"{CallbackData.SUBSCRIPTION}_enterprise")],
                [InlineKeyboardButton("◀️ Retour", callback_data=CallbackData.BACK)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(subscription_info, reply_markup=reply_markup, parse_mode='Markdown')
            
            return CHOOSING_SUBSCRIPTION
        
        # Stocker le type d'abonnement choisi
        if not user.id in self.user_data:
            self.user_data[user.id] = {}
        
        self.user_data[user.id]['subscription_type'] = subscription_type
        self.user_data[user.id]['subscription_duration'] = CryptoPaymentService.get_subscription_duration(subscription_type)
        
        # Proposer les méthodes de paiement
        payment_message = (
            f"💳 *Méthode de paiement*\n\n"
            f"Choisissez votre méthode de paiement pour l'abonnement *{subscription_type.capitalize()}*:"
        )
        
        keyboard = [
            [InlineKeyboardButton("₿ Bitcoin", callback_data=f"{CallbackData.CURRENCY}_BTC")],
            [InlineKeyboardButton("Ξ Ethereum", callback_data=f"{CallbackData.CURRENCY}_ETH")],
            [InlineKeyboardButton("₮ USDT", callback_data=f"{CallbackData.CURRENCY}_USDT")],
            [InlineKeyboardButton("◀️ Retour", callback_data=CallbackData.BACK)]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(payment_message, reply_markup=reply_markup, parse_mode='Markdown')
        
        return CHOOSING_CURRENCY
    
    async def choose_currency(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Gestionnaire pour le choix d'une méthode de paiement"""
        query = update.callback_query
        await query.answer()
        
        user = self._get_user_from_update(update)
        if not user:
            await query.edit_message_text("Erreur: Utilisateur non trouvé. Veuillez redémarrer avec /start.")
            return ConversationHandler.END
        
        data = query.data.split('_')
        if len(data) < 2:
            return CHOOSING_CURRENCY
        
        currency = data[1]
        subscription_type = self.user_data[user.id]['subscription_type']
        
        # Stocker la devise choisie
        self.user_data[user.id]['currency'] = currency
        
        # Calculer le montant à payer
        amount = CryptoPaymentService.get_subscription_price(subscription_type, currency)
        self.user_data[user.id]['amount'] = amount
        
        # Générer une adresse de paiement
        payment_address = CryptoPaymentService.get_payment_address(currency)
        self.user_data[user.id]['payment_address'] = payment_address
        
        # Créer un enregistrement de paiement dans la base de données
        payment_id = UserManager.create_payment(
            user_id=user.id,
            amount=amount,
            currency=currency,
            payment_address=payment_address,
            subscription_type=subscription_type,
            duration_days=self.user_data[user.id]['subscription_duration']
        )
        
        if not payment_id:
            await query.edit_message_text(
                "Une erreur s'est produite lors de la création du paiement. Veuillez réessayer plus tard."
            )
            return ConversationHandler.END
        
        self.user_data[user.id]['payment_id'] = payment_id
        
        # Générer un QR code pour le paiement
        qr_data = CryptoPaymentService.get_qr_code_url(payment_address, amount, currency)
        #qr_image = self._generate_qr_code(qr_data)
        
        # Afficher les informations de paiement
        payment_info = (
            f"🧾 *Détails du paiement*\n\n"
            f"Abonnement: *{subscription_type.capitalize()}*\n"
            f"Montant: *{amount} {currency}*\n"
            f"Durée: *{self.user_data[user.id]['subscription_duration']} jours*\n\n"
            f"Veuillez envoyer exactement *{amount} {currency}* à l'adresse suivante:\n\n"
            f"`{payment_address}`\n\n"
            f"⚠️ *Important*: Envoyez uniquement {currency} à cette adresse. "
            f"Tout autre crypto envoyé sera perdu.\n\n"
            f"Une fois le paiement effectué, veuillez soumettre le hash de transaction pour confirmation."
        )
        
        keyboard = [
            [InlineKeyboardButton("✅ J'ai payé", callback_data=f"{CallbackData.CONFIRM_PAYMENT}_{payment_id}")],
            [InlineKeyboardButton("❌ Annuler", callback_data=CallbackData.CANCEL_PAYMENT)]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(payment_info, reply_markup=reply_markup, parse_mode='Markdown')
        
        return WAITING_FOR_PAYMENT
    
    async def confirm_payment(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Gestionnaire pour la confirmation du paiement"""
        query = update.callback_query
        await query.answer()
        
        user = self._get_user_from_update(update)
        if not user:
            await query.edit_message_text("Erreur: Utilisateur non trouvé. Veuillez redémarrer avec /start.")
            return ConversationHandler.END
        
        data = query.data.split('_')
        if len(data) < 2:
            return WAITING_FOR_PAYMENT
        
        payment_id = data[1]
        
        # Demander le hash de la transaction
        await query.edit_message_text(
            "📝 Veuillez entrer le hash de transaction (TxID) pour confirmer votre paiement.\n\n"
            "Vous pouvez le trouver dans votre portefeuille crypto ou sur l'exploreur de blockchain.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Annuler", callback_data=CallbackData.CANCEL_PAYMENT)]
            ])
        )
        
        return WAITING_FOR_TX_HASH
    
    async def process_transaction_hash(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Gestionnaire pour le traitement du hash de transaction"""
        user = self._get_user_from_update(update)
        if not user:
            await update.message.reply_text("Erreur: Utilisateur non trouvé. Veuillez redémarrer avec /start.")
            return ConversationHandler.END
        
        tx_hash = update.message.text.strip()
        
        # Vérifier que le hash a un format valide
        if len(tx_hash) < 10 or len(tx_hash) > 100:
            await update.message.reply_text(
                "❌ Le hash de transaction fourni n'est pas valide. Veuillez réessayer."
            )
            return WAITING_FOR_TX_HASH
        
        # Récupérer les informations de paiement
        payment_id = self.user_data[user.id]['payment_id']
        currency = self.user_data[user.id]['currency']
        amount = self.user_data[user.id]['amount']
        payment_address = self.user_data[user.id]['payment_address']
        
        await update.message.reply_text(
            "⏳ Vérification de votre transaction en cours... Cela peut prendre quelques instants."
        )
        
        # Vérifier la transaction sur la blockchain (en asynchrone)
        loop = asyncio.get_event_loop()
        is_valid, tx_details = await loop.run_in_executor(
            None, 
            lambda: CryptoPaymentService.verify_transaction(tx_hash, currency, amount, payment_address)
        )
        
        if is_valid:
            # Confirmer le paiement dans la base de données
            payment_confirmed = UserManager.confirm_payment(payment_id, tx_hash)
            
            if payment_confirmed:
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
                    [InlineKeyboardButton("🏠 Menu principal", callback_data=CallbackData.BACK)]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(success_message, reply_markup=reply_markup, parse_mode='Markdown')
                
                # Nettoyer les données utilisateur
                if user.id in self.user_data:
                    del self.user_data[user.id]
                
                return MAIN_MENU
            else:
                await update.message.reply_text(
                    "❌ Une erreur s'est produite lors de la confirmation du paiement. "
                    "Veuillez contacter le support."
                )
                return ConversationHandler.END
        else:
            error_message = "La transaction n'a pas pu être vérifiée."
            if tx_details and "error" in tx_details:
                error_message = tx_details["error"]
            
            await update.message.reply_text(
                f"❌ *Erreur de vérification*: {error_message}\n\n"
                f"Veuillez vérifier votre hash de transaction et réessayer, ou contacter le support si le problème persiste.",
                parse_mode='Markdown'
            )
            return WAITING_FOR_TX_HASH
    
    async def cancel_payment(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Gestionnaire pour l'annulation du paiement"""
        query = update.callback_query
        await query.answer()
        
        user = self._get_user_from_update(update)
        if user and user.id in self.user_data:
            # Nettoyer les données utilisateur
            del self.user_data[user.id]
        
        await query.edit_message_text(
            "❌ Paiement annulé. Vous pouvez réessayer ultérieurement via le menu principal."
        )
        
        return ConversationHandler.END
    
    async def account_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Affiche les informations du compte utilisateur"""
        query = update.callback_query
        await query.answer()
        
        user = self._get_user_from_update(update)
        if not user:
            await query.edit_message_text("Erreur: Utilisateur non trouvé. Veuillez redémarrer avec /start.")
            return ConversationHandler.END
        
        # Afficher les informations du compte
        subscription_status = "Actif" if user.is_subscription_active() else "Inactif"
        subscription_type = user.subscription_type.capitalize()
        
        account_info = (
            f"👤 *Informations du compte*\n\n"
            f"Nom d'utilisateur: *{user.username}*\n"
            f"Type d'abonnement: *{subscription_type}*\n"
            f"Statut: *{subscription_status}*\n"
        )
        
        if user.subscription_type != SubscriptionType.FREE and user.is_subscription_active():
            account_info += (
                f"Date d'expiration: *{user.subscription_expiry.strftime('%d/%m/%Y')}*\n"
                f"Jours restants: *{user.get_subscription_days_left()}*\n"
            )
        
        keyboard = [
            [InlineKeyboardButton("💰 S'abonner", callback_data=CallbackData.SUBSCRIBE)],
            [InlineKeyboardButton("◀️ Retour", callback_data=CallbackData.BACK)]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(account_info, reply_markup=reply_markup, parse_mode='Markdown')
        
        return MAIN_MENU
    
    async def coinbase_payment(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Redirige vers le paiement Coinbase Commerce"""
        query = update.callback_query
        await query.answer()
        
        # Rediriger vers le menu d'abonnement Coinbase
        await query.edit_message_text(
            "🔄 Redirection vers le paiement via Coinbase Commerce...",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Continuer", callback_data=CoinbaseCallbacks.SUBSCRIBE)]
            ])
        )
        
        return ConversationHandler.END
    
    def run(self):
        """Démarre le bot"""
        self.application.run_polling() 