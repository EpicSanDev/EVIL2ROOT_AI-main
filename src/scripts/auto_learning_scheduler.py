#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'automatisation du système d'auto-apprentissage des modèles de trading.
Ce script est conçu pour être exécuté périodiquement via cron, systemd timer ou Task Scheduler.
Il exécute un cycle complet d'apprentissage, incluant l'analyse des performances,
la détection des erreurs et l'ajustement des modèles.
"""

import os
import sys
import json
import logging
import argparse
import datetime
import traceback
from pathlib import Path

# Ajout du chemin racine au sys.path pour permettre les imports relatifs
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

try:
    from src.models.auto_learning.learning_orchestrator import LearningOrchestrator
except ImportError:
    print(f"Erreur: Impossible d'importer les modules nécessaires. Vérifiez que le chemin {ROOT_DIR} est correct.")
    sys.exit(1)

def setup_logging(config, log_dir=None, log_level=None):
    """
    Configuration du système de journalisation.
    
    Args:
        config (dict): Configuration contenant les paramètres de journalisation
        log_dir (str, optional): Répertoire de logs personnalisé
        log_level (str, optional): Niveau de log personnalisé
    """
    log_config = config.get('logging', {})
    
    # Priorité aux arguments de ligne de commande
    if log_dir is None:
        log_dir = log_config.get('log_dir', 'logs')
    
    log_level = log_level or log_config.get('level', 'INFO')
    
    # Création du répertoire de logs s'il n'existe pas
    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Date du jour pour le nom de fichier de log
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir_path / f"auto_learning_{today}.log"
    
    # Configuration du logging
    logging_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configuration du handler pour fichier
    handlers = []
    if log_config.get('file_enabled', True):
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(logging_format))
        handlers.append(file_handler)
    
    # Configuration du handler pour console
    if log_config.get('console_enabled', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(logging_format))
        handlers.append(console_handler)
    
    # Application de la configuration
    logging.basicConfig(
        level=level,
        format=logging_format,
        handlers=handlers
    )
    
    return logging.getLogger('auto_learning_scheduler')

def load_config(config_path):
    """
    Charge la configuration depuis un fichier JSON.
    
    Args:
        config_path (str): Chemin vers le fichier de configuration
        
    Returns:
        dict: Configuration chargée
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erreur lors du chargement de la configuration: {e}")
        print(f"Utilisation de la configuration par défaut.")
        return {}

def save_execution_record(results, output_dir):
    """
    Enregistre les résultats de l'exécution dans un fichier JSON.
    
    Args:
        results (dict): Résultats du cycle d'apprentissage
        output_dir (str): Répertoire de sortie
    """
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir_path / f"learning_cycle_{timestamp}.json"
    
    # Ajout des méta-données de l'exécution
    results['execution_metadata'] = {
        'timestamp': datetime.datetime.now().isoformat(),
        'executed_by': os.environ.get('USER', os.environ.get('USERNAME', 'unknown')),
        'host': os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'unknown'))
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return output_file

def main():
    # Parsing des arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Exécution programmée du système d\'auto-apprentissage')
    parser.add_argument('--config', type=str, default=os.path.join(ROOT_DIR, 'src/config/auto_learning_config.json'),
                        help='Chemin vers le fichier de configuration')
    parser.add_argument('--db-path', type=str, default=os.path.join(ROOT_DIR, 'data/trade_journal.db'),
                        help='Chemin vers la base de données du journal de trading')
    parser.add_argument('--models-dir', type=str, default=os.path.join(ROOT_DIR, 'models'),
                        help='Répertoire contenant les modèles entraînés')
    parser.add_argument('--reports-dir', type=str, default=os.path.join(ROOT_DIR, 'data/reports'),
                        help='Répertoire pour les rapports générés')
    parser.add_argument('--log-dir', type=str, help='Répertoire pour les fichiers de log')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Niveau de détail des logs')
    parser.add_argument('--force', action='store_true', 
                        help='Force l\'exécution même si la fréquence configurée n\'est pas atteinte')
    parser.add_argument('--output-dir', type=str, default=os.path.join(ROOT_DIR, 'data/learning_history'),
                        help='Répertoire pour les résultats de l\'exécution')
    
    args = parser.parse_args()
    
    # Chargement de la configuration
    config = load_config(args.config)
    
    # Configuration du logging
    logger = setup_logging(config, args.log_dir, args.log_level)
    
    logger.info("Démarrage du script d'auto-apprentissage")
    logger.info(f"Répertoire racine: {ROOT_DIR}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Base de données: {args.db_path}")
    
    try:
        # Initialisation de l'orchestrateur d'apprentissage
        orchestrator = LearningOrchestrator(
            config_path=args.config,
            db_path=args.db_path,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir
        )
        
        # Vérification de la fréquence d'exécution
        if not args.force:
            learning_frequency = config.get('learning_frequency_days', 7)
            last_execution_record = Path(args.output_dir) / "last_execution.txt"
            
            if last_execution_record.exists():
                with open(last_execution_record, 'r') as f:
                    last_run = datetime.datetime.fromisoformat(f.read().strip())
                    days_since_last_run = (datetime.datetime.now() - last_run).days
                    
                    if days_since_last_run < learning_frequency:
                        logger.info(f"Dernier cycle d'apprentissage exécuté il y a {days_since_last_run} jours. "
                                    f"La fréquence configurée est de {learning_frequency} jours. "
                                    f"Exécution ignorée. Utilisez --force pour forcer l'exécution.")
                        return 0
        
        # Exécution du cycle d'apprentissage
        logger.info("Exécution du cycle d'apprentissage...")
        results = orchestrator.run_learning_cycle()
        
        # Enregistrement des résultats
        output_file = save_execution_record(results, args.output_dir)
        logger.info(f"Résultats de l'exécution enregistrés dans {output_file}")
        
        # Mise à jour de la date de dernière exécution
        last_execution_record = Path(args.output_dir) / "last_execution.txt"
        with open(last_execution_record, 'w') as f:
            f.write(datetime.datetime.now().isoformat())
        
        # Statistiques sommaires
        logger.info("Résumé du cycle d'apprentissage:")
        logger.info(f"- Performances analysées sur {results.get('days_analyzed', 'N/A')} jours")
        logger.info(f"- Transactions analysées: {results.get('trades_analyzed', 0)}")
        logger.info(f"- Erreurs détectées: {len(results.get('detected_errors', []))}")
        logger.info(f"- Modèles ajustés: {len(results.get('adjustments_made', []))}")
        
        # Affichage des erreurs détectées
        if results.get('detected_errors'):
            logger.info("Principales erreurs détectées:")
            error_patterns = results.get('error_patterns', {})
            for error_type, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"- {error_type}: {count} occurrences")
        
        # Affichage des modèles ajustés
        if results.get('adjustments_made'):
            logger.info("Modèles ajustés:")
            for adjustment in results.get('adjustments_made', [])[:5]:
                logger.info(f"- {adjustment.get('model_name')} (v{adjustment.get('old_version')} -> "
                           f"v{adjustment.get('new_version')}): {adjustment.get('adjustment_reason')}")
        
        logger.info("Exécution terminée avec succès")
        return 0
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du cycle d'apprentissage: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 