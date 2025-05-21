#!/usr/bin/env python3
"""
Script d'envoi de notification manuel.
Permet d'envoyer une notification à tous les utilisateurs premium ou à un utilisateur spécifique.
"""

import os
import sys
import logging
import asyncio
import argparse
from typing import Optional

# Ajustement du chemin pour l'import relatif
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.notification_service import notification_service
from app.models.user import UserManager, SubscriptionType

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/notifications.log')
    ]
)
logger = logging.getLogger(__name__)

async def send_bulk_notification(message: str, subscription_type: Optional[str] = None):
    """
    Envoie une notification à tous les utilisateurs ayant le type d'abonnement spécifié.
    
    Args:
        message: Message à envoyer
        subscription_type: Type d'abonnement (optionnel)
    """
    logger.info(f"Envoi de notification en masse à tous les utilisateurs {subscription_type or 'premium'}")
    
    results = await notification_service.send_bulk_notification(message, subscription_type)
    
    logger.info(f"Notification envoyée à {len(results)} utilisateurs")
    logger.info(f"Succès: {sum(1 for v in results.values() if v)}, Échecs: {sum(1 for v in results.values() if not v)}")

async def send_single_notification(telegram_id: str, message: str):
    """
    Envoie une notification à un utilisateur spécifique.
    
    Args:
        telegram_id: ID Telegram de l'utilisateur
        message: Message à envoyer
    """
    logger.info(f"Envoi de notification à l'utilisateur {telegram_id}")
    
    success = await notification_service.send_notification(telegram_id, message)
    
    if success:
        logger.info("Notification envoyée avec succès")
    else:
        logger.error("Échec de l'envoi de la notification")

async def send_trade_signal(symbol: str, direction: str, entry_price: float, 
                         stop_loss: float, take_profit: float, confidence: float, 
                         subscription_type: Optional[str] = None):
    """
    Envoie un signal de trading à tous les utilisateurs ayant le type d'abonnement spécifié.
    
    Args:
        symbol: Symbole du marché
        direction: Direction du trade (LONG/SHORT)
        entry_price: Prix d'entrée
        stop_loss: Prix du stop loss
        take_profit: Prix du take profit
        confidence: Niveau de confiance (0-1)
        subscription_type: Type d'abonnement (optionnel)
    """
    logger.info(f"Envoi de signal de trading pour {symbol} à tous les utilisateurs {subscription_type or 'premium'}")
    
    results = await notification_service.send_signal_notification(
        symbol, direction, entry_price, stop_loss, take_profit, confidence, subscription_type
    )
    
    logger.info(f"Signal de trading envoyé à {len(results)} utilisateurs")
    logger.info(f"Succès: {sum(1 for v in results.values() if v)}, Échecs: {sum(1 for v in results.values() if not v)}")

async def main():
    """
    Fonction principale du script.
    Parse les arguments et envoie la notification.
    """
    parser = argparse.ArgumentParser(description="Envoyer une notification via Telegram")
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Sous-commande pour envoyer une notification en masse
    bulk_parser = subparsers.add_parser("bulk", help="Envoyer une notification en masse")
    bulk_parser.add_argument("--message", "-m", required=True, help="Message à envoyer")
    bulk_parser.add_argument("--subscription", "-s", choices=["basic", "premium", "enterprise"], 
                           help="Type d'abonnement (optionnel)")
    
    # Sous-commande pour envoyer une notification à un utilisateur spécifique
    single_parser = subparsers.add_parser("single", help="Envoyer une notification à un utilisateur spécifique")
    single_parser.add_argument("--telegram-id", "-t", required=True, help="ID Telegram de l'utilisateur")
    single_parser.add_argument("--message", "-m", required=True, help="Message à envoyer")
    
    # Sous-commande pour envoyer un signal de trading
    signal_parser = subparsers.add_parser("signal", help="Envoyer un signal de trading")
    signal_parser.add_argument("--symbol", "-s", required=True, help="Symbole du marché")
    signal_parser.add_argument("--direction", "-d", required=True, choices=["LONG", "SHORT"], 
                             help="Direction du trade (LONG/SHORT)")
    signal_parser.add_argument("--entry-price", "-e", required=True, type=float, help="Prix d'entrée")
    signal_parser.add_argument("--stop-loss", "-sl", required=True, type=float, help="Prix du stop loss")
    signal_parser.add_argument("--take-profit", "-tp", required=True, type=float, help="Prix du take profit")
    signal_parser.add_argument("--confidence", "-c", required=True, type=float, 
                             help="Niveau de confiance (0-1)")
    signal_parser.add_argument("--subscription", "-sub", choices=["basic", "premium", "enterprise"], 
                             help="Type d'abonnement (optionnel)")
    
    args = parser.parse_args()
    
    # S'assurer que la base de données est initialisée
    UserManager.init_db()
    
    if args.command == "bulk":
        await send_bulk_notification(args.message, args.subscription)
    elif args.command == "single":
        await send_single_notification(args.telegram_id, args.message)
    elif args.command == "signal":
        await send_trade_signal(
            args.symbol, args.direction, args.entry_price, 
            args.stop_loss, args.take_profit, args.confidence, 
            args.subscription
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    # Exécuter la fonction principale de manière asynchrone
    asyncio.run(main()) 