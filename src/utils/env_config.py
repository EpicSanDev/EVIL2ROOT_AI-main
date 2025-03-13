#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilitaire pour la gestion des variables d'environnement dans l'application.

Ce module fournit des fonctions pour récupérer des variables d'environnement avec des
valeurs par défaut sécurisées et appropriées au contexte (développement vs production).
"""

import os
import secrets
import logging
from typing import Any, Optional, Dict

# Vérifier si nous sommes en mode développement ou production
IS_PRODUCTION = os.environ.get('FLASK_ENV', 'development').lower() == 'production'

def generate_secure_password(length: int = 16) -> str:
    """
    Génère un mot de passe aléatoire sécurisé.
    
    Args:
        length: Longueur du mot de passe à générer
        
    Returns:
        Une chaîne aléatoire sécurisée
    """
    return secrets.token_urlsafe(length)

def get_env_var(var_name: str, default_value: Optional[Any] = None, 
                required_in_production: bool = False) -> Any:
    """
    Récupère une variable d'environnement avec une valeur par défaut.
    En production, si la variable est marquée comme requise, une exception est levée
    si la variable n'est pas définie.
    
    Args:
        var_name: Nom de la variable d'environnement
        default_value: Valeur par défaut si la variable n'est pas définie
        required_in_production: Si True, la variable est requise en production
        
    Returns:
        La valeur de la variable d'environnement ou la valeur par défaut
        
    Raises:
        ValueError: Si la variable est requise en production mais n'est pas définie
    """
    value = os.environ.get(var_name)
    
    if value is None:
        if IS_PRODUCTION and required_in_production:
            logging.error(f"Variable d'environnement requise manquante en production: {var_name}")
            raise ValueError(f"Variable d'environnement requise manquante en production: {var_name}")
        return default_value
    
    return value

def get_db_params() -> Dict[str, str]:
    """
    Récupère les paramètres de connexion à la base de données.
    En production, toutes les variables sont requises.
    En développement, des valeurs par défaut sont utilisées.
    
    Returns:
        Dictionnaire contenant les paramètres de connexion à la base de données
        
    Raises:
        ValueError: Si une variable requise est manquante en production
    """
    if IS_PRODUCTION:
        # En production, exiger que les identifiants soient définis explicitement
        db_params = {
            'dbname': get_env_var('DB_NAME', required_in_production=True),
            'user': get_env_var('DB_USER', required_in_production=True),
            'password': get_env_var('DB_PASSWORD', required_in_production=True),
            'host': get_env_var('DB_HOST', required_in_production=True),
            'port': get_env_var('DB_PORT', required_in_production=True)
        }
    else:
        # En développement, utiliser des valeurs par défaut mais générer un mot de passe aléatoire
        # pour éviter d'utiliser des mots de passe par défaut hardcodés
        db_password = get_env_var('DB_PASSWORD', generate_secure_password())
        
        db_params = {
            'dbname': get_env_var('DB_NAME', 'trading_db'),
            'user': get_env_var('DB_USER', 'trader'),
            'password': db_password,
            'host': get_env_var('DB_HOST', 'localhost'),
            'port': get_env_var('DB_PORT', '5432')
        }
        
        # Journaliser le mot de passe généré en développement
        if 'DB_PASSWORD' not in os.environ:
            logging.warning("Using generated secure password for database. This password will change each time the application restarts.")
            logging.info(f"Generated password: {db_params['password']}")
    
    return db_params

def get_redis_params() -> Dict[str, Any]:
    """
    Récupère les paramètres de connexion à Redis.
    
    Returns:
        Dictionnaire contenant les paramètres de connexion à Redis
    """
    return {
        'host': get_env_var('REDIS_HOST', 'localhost'),
        'port': int(get_env_var('REDIS_PORT', 6379))
    } 