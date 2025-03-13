#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script start_daily_analysis.py - Point d'entrée pour l'analyse quotidienne automatisée
Ce script analyse les données de marché, produit des prédictions et génère des rapports.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ajouter le répertoire src au path pour pouvoir importer les modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

try:
    from src.core.market_data import data_fetcher
    from src.core.trading import analysis
    from src.models import load_models
    from src.utils.logger import setup_logger
    from src.notification.report_generator import generate_report
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Vérifiez que vous exécutez ce script depuis le répertoire racine du projet")
    sys.exit(1)

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Analyse quotidienne des marchés financiers")
    parser.add_argument("--symbols", type=str, default="SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,TSLA",
                      help="Liste de symboles à analyser, séparés par des virgules")
    parser.add_argument("--detail", choices=["basic", "standard", "comprehensive"], default="standard",
                      help="Niveau de détail du rapport d'analyse")
    parser.add_argument("--output-dir", type=str, default="./data/reports",
                      help="Répertoire de sortie pour les rapports générés")
    parser.add_argument("--force-train", action="store_true",
                      help="Force l'entraînement des modèles avant l'analyse")
    parser.add_argument("--days", type=int, default=90,
                      help="Nombre de jours de données historiques à utiliser")
    parser.add_argument("--notify", action="store_true",
                      help="Envoyer une notification avec les résultats")
    return parser.parse_args()

def main():
    """Fonction principale d'exécution de l'analyse quotidienne."""
    args = parse_arguments()
    
    # Configuration du logger
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger(
        "daily_analysis", 
        log_file=log_dir / f"analysis_{datetime.now().strftime('%Y%m%d')}.log"
    )
    logger.info("Démarrage de l'analyse quotidienne des marchés")
    
    try:
        # Création du répertoire de sortie s'il n'existe pas
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        symbols = args.symbols.split(",")
        logger.info(f"Analyse des symboles: {symbols}")
        
        # Chargement ou entraînement des modèles
        if args.force_train:
            logger.info("Entraînement forcé des modèles...")
            models = load_models.train_all_models(symbols)
        else:
            logger.info("Chargement des modèles existants...")
            models = load_models.load_production_models()
        
        # Récupération des données de marché
        logger.info(f"Récupération des données pour les derniers {args.days} jours...")
        market_data = data_fetcher.fetch_historical_data(symbols, days=args.days)
        
        # Exécution de l'analyse
        logger.info("Exécution de l'analyse de marché...")
        analysis_results = analysis.run_market_analysis(
            symbols=symbols,
            market_data=market_data,
            models=models,
            detail_level=args.detail
        )
        
        # Génération des rapports
        logger.info("Génération des rapports d'analyse...")
        report_date = datetime.now().strftime("%Y-%m-%d")
        report_files = generate_report(
            analysis_results=analysis_results,
            output_dir=output_dir,
            report_date=report_date,
            detail_level=args.detail
        )
        
        # Envoyer des notifications si demandé
        if args.notify:
            logger.info("Envoi des notifications...")
            from src.services.notification_service import send_analysis_notification
            send_analysis_notification(
                symbols=symbols,
                analysis_results=analysis_results,
                report_files=report_files
            )
        
        logger.info(f"Analyse quotidienne terminée avec succès. Rapports disponibles dans: {output_dir}")
        return 0
        
    except Exception as e:
        logger.exception(f"Erreur lors de l'analyse quotidienne: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())