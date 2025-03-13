#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitaire pour la configuration des logs dans l'application.

Ce module fournit une fonction pour configurer les logs avec des chemins relatifs
au répertoire du projet et des valeurs par défaut sécurisées.
"""

import os
import logging
from pathlib import Path
from typing import Optional

def setup_logging(logger_name: str, log_filename: Optional[str] = None) -> logging.Logger:
    """
    Configure un logger avec les paramètres appropriés.
    
    Args:
        logger_name: Nom du logger à configurer
        log_filename: Nom du fichier de log (sans le chemin)
                      Si None, utilise <logger_name>.log
    
    Returns:
        Le logger configuré
    """
    # Utiliser Path pour créer des chemins indépendants du système d'exploitation
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "logs"
    
    # S'assurer que le répertoire de logs existe
    log_dir.mkdir(exist_ok=True)
    
    # Utiliser le nom du logger comme nom de fichier par défaut
    if log_filename is None:
        log_filename = f"{logger_name}.log"
    
    log_file = log_dir / log_filename
    
    # Configurer le niveau de log à partir de la variable d'environnement ou utiliser INFO par défaut
    log_level_name = os.environ.get("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    
    # Obtenir le logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Vérifier si des gestionnaires sont déjà attachés au logger pour éviter les doublons
    if not logger.handlers:
        # Format de log standard
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Handler pour la sortie console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Handler pour le fichier
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Configurer le logger racine pour qu'il n'interfère pas
        logging.getLogger().setLevel(logging.WARNING)
    
    return logger 