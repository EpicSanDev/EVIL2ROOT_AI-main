#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'exemple montrant comment utiliser le système d'auto-apprentissage
pour les modèles de trading.
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta
import argparse

# Ajouter le répertoire parent au chemin de recherche pour pouvoir importer les modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.auto_learning import (
    TradeJournal,
    ErrorDetector,
    PerformanceAnalyzer,
    ModelAdjuster
)
from src.models.auto_learning.learning_orchestrator import LearningOrchestrator

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('auto_learning_example.log')
    ]
)

logger = logging.getLogger(__name__)

def create_sample_config():
    """Crée un fichier de configuration d'exemple"""
    config = {
        "learning_frequency_days": 7,
        "analysis_window_days": 30,
        "error_analysis_window_days": 90,
        "min_trades_for_analysis": 10,
        "auto_adjust_enabled": True,
        "visualization_enabled": True,
        "email_reports_enabled": False,
        "email_recipients": ["user@example.com"]
    }
    
    with open('auto_learning_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info("Fichier de configuration d'exemple créé: auto_learning_config.json")
    return 'auto_learning_config.json'

def create_sample_trades(trade_journal, num_trades=50):
    """Crée des transactions d'exemple dans le journal"""
    logger.info(f"Création de {num_trades} transactions d'exemple...")
    
    # Modèles fictifs pour l'exemple
    models = [
        {"name": "trend_following_v1", "version": "1.0.0"},
        {"name": "rl_trading_model", "version": "2.1.3"},
        {"name": "price_action_trader", "version": "0.9.5"},
        {"name": "sentiment_analyzer", "version": "1.2.0"},
    ]
    
    # Symboles fictifs
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD"]
    
    # Périodes de temps
    start_date = datetime.now() - timedelta(days=60)
    
    # Créer des transactions avec différents résultats
    for i in range(num_trades):
        # Sélectionner un modèle aléatoire
        model_idx = i % len(models)
        model = models[model_idx]
        
        # Sélectionner un symbole aléatoire
        symbol_idx = (i + model_idx) % len(symbols)
        symbol = symbols[symbol_idx]
        
        # Définir les dates d'entrée et de sortie
        entry_time = start_date + timedelta(days=i)
        exit_time = entry_time + timedelta(hours=24 + (i % 48))
        
        # Définir le résultat de la transaction (60% de chances d'être profitable)
        is_profitable = (i % 10) < 6
        
        # Prix d'entrée et de sortie
        entry_price = 1000 + (i % 100)
        price_change = (50 + (i % 50)) * (1 if is_profitable else -1)
        exit_price = entry_price + price_change
        
        # Calculer le PnL
        position_size = 0.1 + (i % 10) / 100
        pnl = position_size * price_change
        pnl_percent = (price_change / entry_price) * 100
        
        # Créer des données de signaux et conditions de marché
        entry_signals = {
            "rsi": 30 + (i % 40),
            "macd": 0.5 if is_profitable else -0.5,
            "signal_time": (entry_time - timedelta(minutes=5 + (i % 25))).isoformat()
        }
        
        exit_signals = {
            "take_profit": 5.0 if is_profitable else 2.0,
            "stop_loss": entry_price * 0.95,
            "profit_target": 5.0,
            "signal_time": (exit_time - timedelta(minutes=3 + (i % 15))).isoformat()
        }
        
        market_conditions = {
            "volatility": 0.2 + (i % 10) / 100,
            "trend": "UPTREND" if is_profitable else "DOWNTREND" if i % 3 == 0 else "RANGING",
            "market_type": "trending" if is_profitable else "ranging",
            "volume": 1000000 + (i * 10000)
        }
        
        # Définir la direction de la transaction
        direction = "BUY" if i % 2 == 0 else "SELL"
        
        # Créer la transaction
        trade_data = {
            "symbol": symbol,
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": position_size,
            "direction": direction,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "fee": position_size * entry_price * 0.001,
            "strategy_name": model["name"],
            "model_version": model["version"],
            "entry_signals": entry_signals,
            "exit_signals": exit_signals,
            "market_conditions": market_conditions,
            "trade_metadata": {
                "is_overtrading": i % 20 == 0,
                "session_trades": i % 7,
                "market_hours": "regular"
            }
        }
        
        # Enregistrer la transaction
        trade_id = trade_journal.log_trade(trade_data)
        
        if i % 10 == 0:
            logger.info(f"Créé transaction {i+1}/{num_trades}, ID: {trade_id}, PnL: {pnl:.2f}")
    
    logger.info(f"{num_trades} transactions d'exemple créées avec succès")

def demo_auto_learning():
    """Démontre l'utilisation du système d'auto-apprentissage"""
    try:
        # 1. Créer une configuration d'exemple
        config_path = create_sample_config()
        
        # 2. Initialiser l'orchestrateur
        orchestrator = LearningOrchestrator(
            config_path=config_path,
            db_path="data/demo_trade_journal.db",
            models_dir="saved_models/demo",
            reports_dir="data/demo_reports"
        )
        
        # 3. Créer des transactions d'exemple
        create_sample_trades(orchestrator.trade_journal)
        
        # 4. Exécuter une analyse des performances
        logger.info("Exécution d'une analyse des performances...")
        performance_results = orchestrator.analyze_performance(days=30)
        
        if 'error' in performance_results:
            logger.warning(f"Erreur lors de l'analyse des performances: {performance_results['error']}")
        else:
            logger.info(f"Analyse des performances terminée: {len(performance_results.get('total_trades', 0))} transactions analysées")
            logger.info(f"Taux de réussite global: {performance_results.get('win_rate', 0):.2%}")
        
        # 5. Exécuter une analyse des erreurs
        logger.info("Exécution d'une analyse des erreurs...")
        error_results = orchestrator.detect_errors(days=60)
        
        if 'error' in error_results:
            logger.warning(f"Erreur lors de l'analyse des erreurs: {error_results['error']}")
        else:
            error_count = len(error_results.get('detected_errors', []))
            logger.info(f"Analyse des erreurs terminée: {error_count} erreurs détectées")
            
            # Afficher les types d'erreurs détectés
            error_types = error_results.get('error_patterns', {}).keys()
            if error_types:
                logger.info(f"Types d'erreurs détectés: {', '.join(error_types)}")
        
        # 6. Générer des visualisations
        logger.info("Génération des visualisations...")
        visualizations = orchestrator.performance_analyzer.generate_performance_visualizations()
        
        if 'error' in visualizations:
            logger.warning(f"Erreur lors de la génération des visualisations: {visualizations['error']}")
        else:
            logger.info(f"Visualisations générées: {list(visualizations.keys())}")
            logger.info(f"Rapport HTML disponible à: {visualizations.get('html_report', 'N/A')}")
        
        # 7. Exécuter un cycle complet d'apprentissage
        logger.info("Exécution d'un cycle complet d'auto-apprentissage...")
        learning_results = orchestrator.run_learning_cycle()
        
        logger.info(f"Cycle d'apprentissage terminé avec statut: {learning_results.get('status', 'unknown')}")
        
        if learning_results.get('status') == 'success':
            adjustments = learning_results.get('model_adjustments', {})
            logger.info(f"Modèles ajustés: {adjustments.get('models_adjusted', 0)}")
        
        logger.info("Démonstration du système d'auto-apprentissage terminée avec succès")
        
        return learning_results
        
    except Exception as e:
        logger.error(f"Erreur lors de la démonstration: {e}")
        return {"status": "error", "error_message": str(e)}

def demo_error_detection_with_claude():
    """Démontre l'utilisation du détecteur d'erreurs avec Claude 3.7 via OpenRouter"""
    try:
        # 1. Créer une instance de TradeJournal
        trade_journal = TradeJournal(db_path="data/demo_trade_journal.db")
        
        # 2. Créer des transactions d'exemple
        create_sample_trades(trade_journal)
        
        # 3. Configurer la clé API OpenRouter (remplacer par votre clé)
        api_key = input("Entrez votre clé API OpenRouter pour utiliser Claude 3.7: ").strip()
        if not api_key:
            logger.warning("Aucune clé API fournie. Analyse sans IA.")
            use_ai = False
        else:
            # Configurer le fichier .env avec la clé API
            ErrorDetector.configure_openrouter_api(api_key)
            use_ai = True
        
        # 4. Initialiser le détecteur d'erreurs
        error_detector = ErrorDetector(trade_journal=trade_journal, use_ai_analysis=use_ai)
        
        # 5. Analyser les transactions perdantes
        logger.info("Analyse des transactions perdantes avec détection d'erreurs...")
        error_results = error_detector.analyze_losing_trades(days=60)
        
        if 'error' in error_results:
            logger.warning(f"Erreur lors de l'analyse: {error_results['error']}")
            return error_results
        
        # 6. Afficher les résultats
        logger.info(f"Analyse terminée: {len(error_results.get('detected_errors', []))} transactions analysées")
        
        # Afficher les patterns d'erreur
        logger.info("\n=== PATTERNS D'ERREUR ===")
        for error_type, data in error_results.get('error_patterns', {}).items():
            logger.info(f"{error_type}: {data['count']} occurrences, perte moyenne: {data['avg_loss']:.2f}")
        
        # Afficher les recommandations
        logger.info("\n=== RECOMMANDATIONS ===")
        for rec in error_results.get('recommendations', []):
            logger.info(f"- {rec.get('error_type')}: {rec.get('description')}")
            
            # Afficher les actions recommandées
            actions = rec.get('actions', [])
            if isinstance(actions, list):
                for i, action in enumerate(actions, 1):
                    logger.info(f"  Action {i}: {action}")
            
        # 7. Afficher les informations sur l'analyse IA si disponible
        if 'analysis_method' in error_results and 'Claude' in error_results['analysis_method']:
            logger.info("\n=== ANALYSE IA AVEC CLAUDE 3.7 ===")
            logger.info(f"Méthode d'analyse: {error_results['analysis_method']}")
            
            # Afficher les insights de Claude
            ai_analysis = error_results.get('ai_analysis', {})
            if 'insights' in ai_analysis:
                logger.info("\nInsights de Claude 3.7:")
                for insight in ai_analysis['insights']:
                    logger.info(f"- {insight}")
        
        logger.info("\nDémonstration du détecteur d'erreurs terminée avec succès")
        return error_results
        
    except Exception as e:
        logger.error(f"Erreur lors de la démonstration: {e}")
        return {"status": "error", "error_message": str(e)}

if __name__ == "__main__":
    logger.info("=== Démarrage de la démonstration du système d'auto-apprentissage ===")
    
    # Ajouter un argument pour choisir la démo
    parser = argparse.ArgumentParser(description="Démonstration du système d'auto-apprentissage")
    parser.add_argument('--demo', choices=['complete', 'error-detection'], 
                      default='complete', help='Type de démonstration à exécuter')
    args = parser.parse_args()
    
    if args.demo == 'error-detection':
        # Exécuter uniquement la démo du détecteur d'erreurs avec Claude 3.7
        logger.info("Exécution de la démo du détecteur d'erreurs avec Claude 3.7")
        results = demo_error_detection_with_claude()
    else:
        # Exécuter la démo complète d'auto-apprentissage
        logger.info("Exécution de la démo complète d'auto-apprentissage")
        results = demo_auto_learning()
    
    logger.info(f"Résultats: {results.get('status', 'completed')}")
    logger.info("=== Démonstration terminée ===") 