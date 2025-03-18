#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de lancement de l'API du bot de trading EVIL2ROOT.

Ce script initialise et démarre le serveur API.
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Ajuster le chemin d'importation
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.log_config import setup_logging
from src.api.app import start_api_server

# Charger les variables d'environnement
load_dotenv()

def parse_arguments():
    """Analyse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='API du bot de trading EVIL2ROOT')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Hôte sur lequel écouter (défaut: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port sur lequel écouter (défaut: 8000)')
    parser.add_argument('--debug', action='store_true',
                        help='Activer le mode debug')
    return parser.parse_args()

def main():
    """Point d'entrée principal."""
    args = parse_arguments()
    
    # Configurer la journalisation
    logger = setup_logging('api', 'api.log')
    
    # Configurer le niveau de journalisation
    if args.debug:
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Mode debug activé")
    
    # Afficher les informations de configuration
    logger.info(f"Démarrage de l'API sur {args.host}:{args.port}")
    
    # Démarrer le serveur API
    try:
        start_api_server(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except Exception as e:
        logger.exception(f"Erreur au démarrage de l'API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 