#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation du module de monitoring amélioré.
Ce script démontre comment initialiser et utiliser les différentes fonctionnalités
du service de monitoring pour le trading bot EVIL2ROOT.
"""

import time
import random
import logging
import numpy as np
from datetime import datetime, timedelta
import threading
import sys
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import du module de monitoring amélioré
from monitoring_enhanced import init_enhanced_monitoring, get_monitoring_service

def simulate_trading_activity(monitoring, symbols, duration=60):
    """
    Simule l'activité de trading pour démontrer les métriques.
    
    Args:
        monitoring: Service de monitoring
        symbols: Liste de symboles à simuler
        duration: Durée de la simulation en secondes
    """
    logger.info(f"Démarrage de la simulation d'activité de trading pour {duration} secondes")
    
    start_time = time.time()
    portfolio_value = 10000.0
    balance = 5000.0
    open_positions = {symbol: {'buy': 0, 'sell': 0} for symbol in symbols}
    
    while time.time() - start_time < duration:
        # Simuler des signaux de trading
        for symbol in symbols:
            if random.random() < 0.3:  # 30% de chance de générer un signal
                direction = random.choice(['buy', 'sell'])
                confidence = random.uniform(0.5, 0.95)
                model_type = random.choice(['price_prediction', 'momentum', 'ensemble'])
                
                # Enregistrer le signal
                monitoring.record_trading_signal(symbol, direction, confidence, model_type)
                
                # Simuler l'exécution du trade avec 80% de chance de succès
                success = random.random() < 0.8
                if success:
                    monitoring.record_executed_trade(symbol, direction, True)
                    open_positions[symbol][direction] += 1
                else:
                    monitoring.record_executed_trade(symbol, direction, False)
        
        # Simuler les variations de portefeuille
        portfolio_value += random.uniform(-100, 150)
        balance += random.uniform(-50, 75)
        
        # Mettre à jour les métriques du portefeuille
        monitoring.update_portfolio_metrics(portfolio_value, balance, open_positions)
        
        # Simuler des métriques de performance pour chaque symbole
        for symbol in symbols:
            timeframe = random.choice(['1h', '4h', '1d'])
            win_rate = random.uniform(0.4, 0.7)
            profit_factor = random.uniform(1.1, 2.5)
            avg_win = random.uniform(50, 200)
            avg_loss = random.uniform(30, 100)
            
            monitoring.update_performance_metrics(
                symbol, timeframe, win_rate, profit_factor, avg_win, avg_loss
            )
        
        # Simuler des métriques de risque
        for timeframe in ['1h', '4h', '1d']:
            max_drawdown = random.uniform(0.05, 0.2)
            sharpe_ratio = random.uniform(0.8, 2.5)
            sortino_ratio = random.uniform(1.0, 3.0)
            calmar_ratio = random.uniform(0.5, 3.5)
            
            monitoring.update_risk_metrics(
                timeframe, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio
            )
        
        # Pause entre les mises à jour
        time.sleep(2)
    
    logger.info("Fin de la simulation d'activité de trading")

def simulate_model_metrics(monitoring, symbols, duration=60):
    """
    Simule les métriques de performance des modèles.
    
    Args:
        monitoring: Service de monitoring
        symbols: Liste de symboles à simuler
        duration: Durée de la simulation en secondes
    """
    logger.info(f"Démarrage de la simulation des métriques de modèle pour {duration} secondes")
    
    start_time = time.time()
    model_names = ['price_prediction', 'risk_management', 'sl_tp_predictor', 'sentiment']
    timeframes = ['1h', '4h', '1d']
    
    while time.time() - start_time < duration:
        # Simuler des métriques de performance pour différents modèles
        for model_name in model_names:
            for symbol in symbols:
                for timeframe in timeframes:
                    # Simuler des métriques de classification ou régression selon le modèle
                    if model_name in ['price_prediction', 'sl_tp_predictor']:
                        # Métriques de régression
                        metrics = {
                            'rmse': random.uniform(0.01, 0.1),
                            'mae': random.uniform(0.005, 0.05)
                        }
                    else:
                        # Métriques de classification
                        metrics = {
                            'accuracy': random.uniform(0.6, 0.9),
                            'precision': random.uniform(0.55, 0.85),
                            'recall': random.uniform(0.6, 0.9),
                            'f1': random.uniform(0.58, 0.88)
                        }
                    
                    # Mettre à jour les métriques du modèle
                    monitoring.update_model_performance(model_name, timeframe, symbol, metrics)
                    
                    # Simuler des métriques de calibration
                    ece = random.uniform(0.01, 0.15)
                    reliability_bins = {
                        '0.1': random.uniform(0.05, 0.15),
                        '0.3': random.uniform(0.25, 0.35),
                        '0.5': random.uniform(0.45, 0.55),
                        '0.7': random.uniform(0.65, 0.75),
                        '0.9': random.uniform(0.85, 0.95)
                    }
                    
                    monitoring.update_calibration_metrics(
                        model_name, timeframe, symbol, ece, reliability_bins
                    )
                    
                    # Simuler des métriques d'apprentissage en ligne
                    drift_score = random.uniform(0, 0.3)
                    memory_size = random.randint(500, 5000)
                    loss = random.uniform(0.01, 0.5)
                    
                    monitoring.update_online_learning_metrics(
                        model_name, timeframe, symbol, drift_score, memory_size, loss
                    )
        
        # Pause entre les mises à jour
        time.sleep(5)
    
    logger.info("Fin de la simulation des métriques de modèle")

def simulate_api_requests(monitoring, duration=60):
    """
    Simule des requêtes API pour démontrer les métriques correspondantes.
    
    Args:
        monitoring: Service de monitoring
        duration: Durée de la simulation en secondes
    """
    logger.info(f"Démarrage de la simulation des requêtes API pour {duration} secondes")
    
    start_time = time.time()
    endpoints = ['/api/market/data', '/api/trading/signals', '/api/portfolio/status', '/api/models/performance']
    methods = ['GET', 'POST', 'PUT']
    
    while time.time() - start_time < duration:
        # Simuler des requêtes API
        for _ in range(random.randint(1, 5)):
            endpoint = random.choice(endpoints)
            method = random.choice(methods)
            
            # Utiliser le contexte API pour mesurer la latence
            with monitoring.api_request_context(endpoint, method) as status_code:
                # Simuler le traitement de la requête
                time.sleep(random.uniform(0.05, 0.5))
                
                # Définir le code de statut (majoritairement des succès)
                status_code.value = 200 if random.random() < 0.9 else random.choice([400, 404, 500])
        
        # Simuler des quotas d'API
        providers = ['binance', 'coinbase', 'openai', 'alphavantage']
        for provider in providers:
            remaining = random.randint(50, 1000)
            monitoring.update_api_quota(provider, remaining)
        
        # Simuler une erreur API occasionnelle
        if random.random() < 0.2:
            endpoint = random.choice(endpoints)
            error_types = ['timeout', 'rate_limit', 'connection_error', 'authentication_error']
            error_type = random.choice(error_types)
            monitoring.record_api_error(endpoint, error_type)
        
        # Pause entre les simulations
        time.sleep(1)
    
    logger.info("Fin de la simulation des requêtes API")

def simulate_db_operations(monitoring, duration=60):
    """
    Simule des opérations de base de données pour démontrer les métriques correspondantes.
    
    Args:
        monitoring: Service de monitoring
        duration: Durée de la simulation en secondes
    """
    logger.info(f"Démarrage de la simulation des opérations DB pour {duration} secondes")
    
    start_time = time.time()
    operation_types = ['select', 'insert', 'update', 'delete']
    
    while time.time() - start_time < duration:
        # Simuler des opérations DB
        for _ in range(random.randint(1, 10)):
            operation_type = random.choice(operation_types)
            
            # Utiliser le contexte DB pour mesurer la durée
            with monitoring.db_query_context(operation_type):
                # Simuler l'exécution de la requête
                time.sleep(random.uniform(0.01, 0.2))
        
        # Simuler des statistiques de connexion
        pool_size = random.randint(5, 20)
        active_connections = random.randint(1, pool_size)
        monitoring.update_db_connection_stats(pool_size, active_connections)
        
        # Pause entre les simulations
        time.sleep(0.5)
    
    logger.info("Fin de la simulation des opérations DB")

def main():
    """Fonction principale pour exécuter la démonstration."""
    logger.info("Démarrage de la démonstration du monitoring amélioré")
    
    # Initialiser le service de monitoring
    monitoring = init_enhanced_monitoring(
        port=8000,
        export_directory="logs/metrics_demo",
        model_metrics_interval=30
    )
    
    # Liste de symboles pour la simulation
    symbols = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'ADA/USD']
    
    # Démarrer les simulations dans des threads séparés
    threads = []
    
    thread_trading = threading.Thread(
        target=simulate_trading_activity,
        args=(monitoring, symbols, 300)  # 5 minutes
    )
    threads.append(thread_trading)
    
    thread_models = threading.Thread(
        target=simulate_model_metrics,
        args=(monitoring, symbols, 300)  # 5 minutes
    )
    threads.append(thread_models)
    
    thread_api = threading.Thread(
        target=simulate_api_requests,
        args=(monitoring, 300)  # 5 minutes
    )
    threads.append(thread_api)
    
    thread_db = threading.Thread(
        target=simulate_db_operations,
        args=(monitoring, 300)  # 5 minutes
    )
    threads.append(thread_db)
    
    # Démarrer tous les threads
    for thread in threads:
        thread.start()
    
    # Attendre que tous les threads se terminent
    for thread in threads:
        thread.join()
    
    logger.info("Démonstration terminée.")
    logger.info("Les métriques sont disponibles sur http://localhost:8000/")
    logger.info("Des rapports JSON ont été exportés dans logs/metrics_demo/")
    
    # Garder le serveur actif pour consultation
    logger.info("Appuyez sur Ctrl+C pour quitter...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    logger.info("Arrêt du service de monitoring...")
    monitoring.stop_server()
    logger.info("Service arrêté.")

if __name__ == "__main__":
    main() 