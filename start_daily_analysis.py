#!/usr/bin/env python3
"""
Script de lancement du bot d'analyse quotidienne
Envoie des analyses complètes (technique, fondamentale, news) via Telegram à intervalles réguliers
"""

import os
import logging
from dotenv import load_dotenv
from app.daily_analysis_bot import run_daily_analysis_bot

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/daily_analysis.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('daily_analysis_launcher')
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # S'assurer que les variables d'environnement requises sont définies
    telegram_token = os.getenv('TELEGRAM_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not telegram_token or not telegram_chat_id:
        logger.error("TELEGRAM_TOKEN ou TELEGRAM_CHAT_ID manquant dans le fichier .env")
        logger.error("Veuillez configurer ces variables pour que le bot puisse envoyer des messages")
        exit(1)
    
    if not openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY manquant dans le fichier .env")
        logger.warning("L'analyse sera simplifiée sans l'utilisation de Claude 3.7")
    
    logger.info("Démarrage du bot d'analyse quotidienne...")
    
    # Démarrer le bot d'analyse
    try:
        run_daily_analysis_bot()
    except KeyboardInterrupt:
        logger.info("Bot d'analyse arrêté par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du bot d'analyse: {e}") 