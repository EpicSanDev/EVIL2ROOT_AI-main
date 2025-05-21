#!/usr/bin/env python3
"""
Script de vérification des abonnements.
Exécute les tâches planifiées pour vérifier les abonnements et envoyer des notifications.
"""

import os
import sys
import asyncio
import logging
import time
from datetime import datetime

# Ajustement du chemin pour l'import relatif
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.subscription_manager import subscription_manager
from app.models.user import UserManager

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/subscription_checker.log')
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """
    Fonction principale du script.
    Vérifie les abonnements et envoie des notifications.
    """
    logger.info("Démarrage de la vérification des abonnements...")
    
    try:
        # Initialiser la base de données si nécessaire
        UserManager.init_db()
        
        # Vérifier les abonnements qui expirent bientôt (dans 3 jours)
        logger.info("Vérification des abonnements qui expirent bientôt...")
        await subscription_manager.check_expiring_subscriptions(days_threshold=3)
        
        # Vérifier les abonnements qui ont expiré
        logger.info("Vérification des abonnements qui ont expiré...")
        await subscription_manager.check_expired_subscriptions()
        
        logger.info("Vérification des abonnements terminée avec succès.")
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification des abonnements: {e}")

if __name__ == "__main__":
    # Exécuter la fonction principale de manière asynchrone
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close() 