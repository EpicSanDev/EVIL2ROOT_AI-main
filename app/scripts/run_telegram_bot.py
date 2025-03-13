#!/usr/bin/env python3
"""
Script d'exécution du bot Telegram premium.
Démarre le bot Telegram pour gérer les abonnements et les paiements.
"""

import os
import sys
import logging
import asyncio

# Ajustement du chemin pour l'import relatif
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.telegram_bot_premium import TelegramBotPremium
from app.models.user import UserManager

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/telegram_bot.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Fonction principale du script.
    Initialise et démarre le bot Telegram premium.
    """
    try:
        logger.info("Initialisation de la base de données...")
        UserManager.init_db()
        
        logger.info("Démarrage du bot Telegram premium...")
        bot = TelegramBotPremium()
        bot.run()
        
        logger.info("Bot Telegram démarré avec succès.")
    
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du bot Telegram: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 