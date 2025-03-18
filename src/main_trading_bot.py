#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import json
import time
import signal
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join('logs', f'trading_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger("TradingBot")

# Import des modules du bot
from services.market_data import BinanceConnector
from models.sentiment import MarketSentimentAnalyzer
from models.rl.advanced_rl_agent import RLAgentManager, MarketRegimeRLTrainer
from core import (
    BacktestEnvironment,
    SentimentBasedStrategy,
    TechnicalStrategy,
    RLBasedStrategy,
    HybridStrategy
)
from core.trading import TradingEngine
from core.risk_management import RiskManager

# Classe principale du bot de trading
class TradingBot:
    """Bot de trading intégrant apprentissage par renforcement, analyse de sentiment et data en temps réel"""
    
    def __init__(self, config_path='config/bot_config.json'):
        """
        Initialisation du bot de trading
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Chargement de la configuration
        self.config = self._load_config(config_path)
        
        # Chargement des clés API depuis .env
        load_dotenv()
        self.api_keys = self._load_api_keys()
        
        # Initialisation des composants
        self._init_components()
    
    def _load_config(self, config_path):
        """Charge la configuration depuis un fichier JSON"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            # Configuration par défaut
            return {
                "trading": {
                    "symbols": ["BTCUSDT", "ETHUSDT"],
                    "timeframes": ["1h", "4h", "1d"],
                    "initial_capital": 10000,
                    "leverage": 1.0,
                    "transaction_fee": 0.001,
                    "dry_run": True
                },
                "risk": {
                    "max_position_size": 0.2,
                    "max_drawdown": 0.1,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.1
                },
                "strategy": {
                    "default": "hybrid",
                    "weights": {
                        "technical": 0.4,
                        "sentiment": 0.3,
                        "rl": 0.3
                    }
                },
                "sentiment": {
                    "update_interval_minutes": 60,
                    "sources": ["twitter", "newsapi", "finnhub"]
                },
                "rl": {
                    "model_dir": "saved_models/rl_agents",
                    "use_market_regime": True,
                    "default_agent": "default_agent",
                    "model_type": "PPO"
                }
            }
    
    def _load_api_keys(self):
        """Charge les clés API depuis les variables d'environnement"""
        api_keys = {}
        
        # Clés pour les sources de données de sentiment
        api_keys['twitter_consumer_key'] = os.getenv('TWITTER_CONSUMER_KEY')
        api_keys['twitter_consumer_secret'] = os.getenv('TWITTER_CONSUMER_SECRET')
        api_keys['twitter_access_token'] = os.getenv('TWITTER_ACCESS_TOKEN')
        api_keys['twitter_access_token_secret'] = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        api_keys['newsapi'] = os.getenv('NEWSAPI_KEY')
        api_keys['finnhub'] = os.getenv('FINNHUB_KEY')
        
        # Clés pour les connexions de marché
        api_keys['binance_api_key'] = os.getenv('BINANCE_API_KEY')
        api_keys['binance_api_secret'] = os.getenv('BINANCE_API_SECRET')
        
        # Supprimer les clés None
        api_keys = {k: v for k, v in api_keys.items() if v is not None}
        
        return api_keys
    
    def _init_components(self):
        """Initialise tous les composants du bot"""
        trading_config = self.config.get("trading", {})
        risk_config = self.config.get("risk", {})
        strategy_config = self.config.get("strategy", {})
        sentiment_config = self.config.get("sentiment", {})
        rl_config = self.config.get("rl", {})
        
        # Initialiser les connecteurs de marché
        self.market_connectors = {}
        for symbol in trading_config.get("symbols", []):
            self.market_connectors[symbol] = BinanceConnector(
                api_key=self.api_keys.get('binance_api_key'),
                api_secret=self.api_keys.get('binance_api_secret'),
                testnet=trading_config.get("dry_run", True)
            )
        
        # Initialiser l'analyseur de sentiment
        self.sentiment_analyzer = MarketSentimentAnalyzer(api_keys=self.api_keys)
        
        # Initialiser le gestionnaire d'agents RL
        self.rl_manager = RLAgentManager(model_dir=rl_config.get("model_dir", "saved_models/rl_agents"))
        
        # Initialiser les stratégies
        self.strategies = {
            "technical": TechnicalStrategy(),
            "sentiment": SentimentBasedStrategy(
                api_keys=self.api_keys,
                max_position=risk_config.get("max_position_size", 0.2)
            ),
            "rl": RLBasedStrategy(
                agent_id=rl_config.get("default_agent", "default_agent"),
                model_dir=rl_config.get("model_dir", "saved_models/rl_agents"),
                model_type=rl_config.get("model_type", "PPO"),
                use_market_regime=rl_config.get("use_market_regime", True),
                api_keys=self.api_keys
            ),
            "hybrid": HybridStrategy(
                tech_weight=strategy_config.get("weights", {}).get("technical", 0.4),
                sentiment_weight=strategy_config.get("weights", {}).get("sentiment", 0.3),
                rl_weight=strategy_config.get("weights", {}).get("rl", 0.3),
                api_keys=self.api_keys,
                agent_id=rl_config.get("default_agent", "default_agent"),
                model_dir=rl_config.get("model_dir", "saved_models/rl_agents")
            )
        }
        
        # Sélectionner la stratégie par défaut
        self.current_strategy = self.strategies.get(
            strategy_config.get("default", "hybrid"),
            self.strategies["hybrid"]
        )
        
        # Initialiser le gestionnaire de risque
        self.risk_manager = RiskManager(
            max_position_size=risk_config.get("max_position_size", 0.2),
            max_drawdown=risk_config.get("max_drawdown", 0.1),
            stop_loss_pct=risk_config.get("stop_loss_pct", 0.05),
            take_profit_pct=risk_config.get("take_profit_pct", 0.1)
        )
        
        # Initialiser le moteur de trading
        self.trading_engine = TradingEngine(
            market_connectors=self.market_connectors,
            strategy=self.current_strategy,
            risk_manager=self.risk_manager,
            initial_capital=trading_config.get("initial_capital", 10000),
            leverage=trading_config.get("leverage", 1.0),
            transaction_fee=trading_config.get("transaction_fee", 0.001)
        )
        
        # Initialisation des threads
        self.market_data_thread = None
        self.sentiment_thread = None
    
    def start(self):
        """Démarre le bot de trading"""
        if self.running:
            self.logger.warning("Le bot est déjà en cours d'exécution.")
            return
        
        self.logger.info("Démarrage du bot de trading...")
        self.running = True
        
        # Connecter tous les connecteurs de marché
        for symbol, connector in self.market_connectors.items():
            self.logger.info(f"Connexion au marché pour {symbol}...")
            connector.connect()
        
        # Démarrer le thread de mise à jour des données de marché
        self.market_data_thread = threading.Thread(target=self._market_data_update_loop)
        self.market_data_thread.daemon = True
        self.market_data_thread.start()
        
        # Démarrer le thread de mise à jour du sentiment
        self.sentiment_thread = threading.Thread(target=self._sentiment_update_loop)
        self.sentiment_thread.daemon = True
        self.sentiment_thread.start()
        
        # Configurer la gestion du signal d'arrêt
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Démarrer la boucle principale
        try:
            self._main_loop()
        except Exception as e:
            self.logger.error(f"Erreur dans la boucle principale: {str(e)}")
            self.stop()
    
    def stop(self):
        """Arrête le bot de trading"""
        if not self.running:
            return
        
        self.logger.info("Arrêt du bot de trading...")
        self.running = False
        
        # Déconnecter tous les connecteurs de marché
        for symbol, connector in self.market_connectors.items():
            self.logger.info(f"Déconnexion du marché pour {symbol}...")
            connector.disconnect()
        
        # Attendre la fin des threads
        if self.market_data_thread and self.market_data_thread.is_alive():
            self.market_data_thread.join(timeout=5)
        
        if self.sentiment_thread and self.sentiment_thread.is_alive():
            self.sentiment_thread.join(timeout=5)
        
        self.logger.info("Bot de trading arrêté.")
    
    def _signal_handler(self, sig, frame):
        """Gestionnaire de signaux pour arrêter proprement le bot"""
        self.logger.info(f"Signal reçu: {sig}")
        self.stop()
        sys.exit(0)
    
    def _market_data_update_loop(self):
        """Boucle de mise à jour des données de marché en temps réel"""
        self.logger.info("Démarrage de la boucle de mise à jour des données de marché...")
        
        # Récupérer les timeframes depuis la configuration
        timeframes = self.config.get("trading", {}).get("timeframes", ["1h", "4h", "1d"])
        
        while self.running:
            try:
                for symbol, connector in self.market_connectors.items():
                    # Vérifier que la connexion est active
                    if not connector.ping():
                        self.logger.warning(f"Connexion perdue pour {symbol}, tentative de reconnexion...")
                        connector.connect()
                        
                    # Récupérer les données historiques pour chaque timeframe
                    for tf in timeframes:
                        # Calculer la date de début en fonction du timeframe
                        if tf == '1h':
                            days_back = 7  # 7 jours
                        elif tf == '4h':
                            days_back = 30  # 30 jours
                        elif tf == '1d':
                            days_back = 90  # 90 jours
                        else:
                            days_back = 30  # Par défaut
                        
                        start_time = datetime.now() - timedelta(days=days_back)
                        
                        # Récupérer les données
                        self.logger.debug(f"Récupération des données pour {symbol} timeframe {tf}...")
                        df = connector.get_historical_klines(
                            symbol=symbol,
                            interval=tf,
                            start_time=start_time
                        )
                        
                        # Mettre à jour les données dans le moteur de trading
                        self.trading_engine.update_market_data(symbol, tf, df)
                
                # Attendre avant la prochaine mise à jour (30 secondes)
                time.sleep(30)
            
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle de mise à jour des données: {str(e)}")
                time.sleep(60)  # Attendre un peu plus longtemps en cas d'erreur
    
    def _sentiment_update_loop(self):
        """Boucle de mise à jour de l'analyse de sentiment"""
        self.logger.info("Démarrage de la boucle de mise à jour du sentiment...")
        
        # Récupérer l'intervalle de mise à jour depuis la configuration
        update_interval = self.config.get("sentiment", {}).get("update_interval_minutes", 60)
        
        while self.running:
            try:
                for symbol in self.market_connectors.keys():
                    # Adapter la requête au symbole
                    if symbol.endswith('USDT'):
                        base_asset = symbol[:-4]
                    else:
                        base_asset = symbol.split('/')[0] if '/' in symbol else symbol[:3]
                    
                    # Requête personnalisée pour l'actif
                    query = f"{base_asset} crypto"
                    
                    # Analyser le sentiment
                    self.logger.debug(f"Analyse du sentiment pour {symbol} avec la requête '{query}'...")
                    sentiment_result = self.sentiment_analyzer.analyze_news_sentiment(query, symbol)
                    
                    # Mettre à jour le sentiment dans le moteur de trading
                    self.trading_engine.update_sentiment(symbol, sentiment_result)
                    
                    # Détecter le régime de marché
                    # Récupérer les données de prix récentes
                    df = self.trading_engine.get_market_data(symbol, '1d')
                    if df is not None and not df.empty:
                        regime_info = self.sentiment_analyzer.detect_market_regime(df, query, symbol)
                        self.trading_engine.update_market_regime(symbol, regime_info)
                
                # Attendre avant la prochaine mise à jour
                time.sleep(update_interval * 60)
            
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle de mise à jour du sentiment: {str(e)}")
                time.sleep(300)  # Attendre 5 minutes en cas d'erreur
    
    def _main_loop(self):
        """Boucle principale du bot de trading"""
        self.logger.info("Démarrage de la boucle principale du bot...")
        
        # Fréquence de trading (en secondes)
        trading_frequency = self.config.get("trading", {}).get("frequency_seconds", 60)
        
        while self.running:
            try:
                # Exécuter un cycle de trading pour chaque symbole
                for symbol in self.market_connectors.keys():
                    self.logger.debug(f"Exécution d'un cycle de trading pour {symbol}...")
                    
                    # Obtenir l'observation actuelle
                    observation = self.trading_engine.get_current_observation(symbol)
                    
                    if observation:
                        # Déterminer l'action avec la stratégie actuelle
                        action = self.current_strategy.get_action(observation)
                        
                        # Appliquer la gestion des risques
                        action = self.risk_manager.validate_action(
                            action, 
                            symbol, 
                            self.trading_engine.get_portfolio_value(),
                            self.trading_engine.get_current_position(symbol)
                        )
                        
                        # Exécuter l'action
                        self.trading_engine.execute_action(symbol, action)
                    
                    else:
                        self.logger.warning(f"Observation non disponible pour {symbol}")
                
                # Journalisation périodique des performances
                self._log_performance()
                
                # Attendre avant le prochain cycle de trading
                time.sleep(trading_frequency)
            
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle principale: {str(e)}")
                time.sleep(60)
    
    def _log_performance(self):
        """Journalise les performances actuelles du bot"""
        try:
            # Valeur du portefeuille
            portfolio_value = self.trading_engine.get_portfolio_value()
            
            # Capital initial
            initial_capital = self.config.get("trading", {}).get("initial_capital", 10000)
            
            # Rendement en pourcentage
            performance_pct = (portfolio_value / initial_capital - 1) * 100
            
            # Positions ouvertes
            positions = self.trading_engine.get_open_positions()
            
            # Journaliser les performances
            self.logger.info(f"Valeur du portefeuille: {portfolio_value:.2f} (Performance: {performance_pct:.2f}%)")
            self.logger.info(f"Positions ouvertes: {positions}")
            
            # Sauvegarder les performances dans un fichier
            performance_dir = os.path.join(os.path.dirname(__file__), '../data/performance')
            os.makedirs(performance_dir, exist_ok=True)
            
            perf_file = os.path.join(performance_dir, 'trading_performance.csv')
            
            # Créer ou mettre à jour le fichier de performances
            perf_data = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'performance_pct': performance_pct,
                'positions': json.dumps(positions)
            }
            
            # Convertir en DataFrame
            perf_df = pd.DataFrame([perf_data])
            
            # Ajouter au fichier existant ou créer un nouveau
            if os.path.exists(perf_file):
                perf_df.to_csv(perf_file, mode='a', header=False, index=False)
            else:
                perf_df.to_csv(perf_file, index=False)
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la journalisation des performances: {str(e)}")
    
    def run_backtest(self, symbol, start_date, end_date, timeframes=None):
        """
        Exécute un backtest pour un symbole donné.
        
        Args:
            symbol: Symbole à tester
            start_date: Date de début du backtest
            end_date: Date de fin du backtest
            timeframes: Liste des timeframes à utiliser
        
        Returns:
            Dict: Résultats du backtest
        """
        self.logger.info(f"Exécution d'un backtest pour {symbol} du {start_date} au {end_date}")
        
        if timeframes is None:
            timeframes = self.config.get("trading", {}).get("timeframes", ["1h", "4h", "1d"])
        
        # Récupérer les données historiques
        data = {}
        connector = self.market_connectors.get(symbol)
        
        if not connector:
            self.logger.error(f"Pas de connecteur disponible pour {symbol}")
            return None
        
        # Essayer de se connecter si ce n'est pas déjà fait
        if not connector.ping():
            connector.connect()
        
        # Récupérer les données pour chaque timeframe
        for tf in timeframes:
            try:
                df = connector.get_historical_klines(
                    symbol=symbol,
                    interval=tf,
                    start_time=start_date,
                    end_time=end_date
                )
                data[tf] = df
            except Exception as e:
                self.logger.error(f"Erreur lors de la récupération des données pour {symbol} {tf}: {str(e)}")
        
        if not data:
            self.logger.error("Aucune donnée disponible pour le backtest")
            return None
        
        # Créer l'environnement de backtesting
        backtest_env = BacktestEnvironment(
            data=data,
            initial_capital=self.config.get("trading", {}).get("initial_capital", 10000),
            transaction_fee=self.config.get("trading", {}).get("transaction_fee", 0.001),
            slippage=0.0005,
            timeframes=timeframes
        )
        
        # Exécuter le backtest avec la stratégie actuelle
        report = backtest_env.run_backtest(self.current_strategy)
        
        # Générer les visualisations
        output_dir = os.path.join(os.path.dirname(__file__), '../results/backtest')
        os.makedirs(output_dir, exist_ok=True)
        
        # Nom du fichier basé sur la stratégie et la date
        strategy_name = next(
            (name for name, strat in self.strategies.items() if strat == self.current_strategy),
            "unknown"
        )
        filename_base = f"{symbol}_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Graphique des résultats
        from core import plot_backtest_results, plot_monthly_heatmap, plot_trade_analysis, generate_performance_report
        
        plot_path = os.path.join(output_dir, f"{filename_base}_results.png")
        backtest_env.plot_results(save_path=plot_path, plot_trades=True)
        
        # Heatmap des rendements mensuels
        main_tf = timeframes[0]
        dates = data[main_tf].index[:len(backtest_env.equity_curve)]
        heatmap_path = os.path.join(output_dir, f"{filename_base}_monthly.png")
        plot_monthly_heatmap(backtest_env.equity_curve, dates, save_path=heatmap_path)
        
        # Rapport HTML
        report_path = os.path.join(output_dir, f"{filename_base}_report.html")
        generate_performance_report(
            metrics=report['metrics'],
            equity_curve=backtest_env.equity_curve,
            trades=backtest_env.trades_history,
            output_path=report_path
        )
        
        self.logger.info(f"Backtest terminé, résultats sauvegardés dans {output_dir}")
        
        return report

# Fonction principale pour exécuter le bot
def main():
    """Point d'entrée principal pour exécuter le bot de trading"""
    # Analyse des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Bot de trading avancé avec RL et analyse de sentiment')
    
    parser.add_argument('--config', type=str, default='config/bot_config.json',
                        help='Chemin vers le fichier de configuration')
    
    parser.add_argument('--backtest', action='store_true',
                        help='Exécuter en mode backtest')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Symbole à trader ou à tester en backtest')
    
    parser.add_argument('--start-date', type=str,
                        help='Date de début pour le backtest (format: YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str,
                        help='Date de fin pour le backtest (format: YYYY-MM-DD)')
    
    parser.add_argument('--strategy', type=str, choices=['technical', 'sentiment', 'rl', 'hybrid'],
                        help='Stratégie à utiliser')
    
    args = parser.parse_args()
    
    # Créer les répertoires nécessaires
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialiser le bot
    bot = TradingBot(config_path=args.config)
    
    # Changer la stratégie si spécifiée
    if args.strategy and args.strategy in bot.strategies:
        bot.current_strategy = bot.strategies[args.strategy]
        logger.info(f"Utilisation de la stratégie: {args.strategy}")
    
    # Mode backtest ou mode trading en direct
    if args.backtest:
        # Convertir les dates pour le backtest
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else datetime.now() - timedelta(days=365)
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
        
        # Exécuter le backtest
        bot.run_backtest(args.symbol, start_date, end_date)
    else:
        # Mode trading en direct
        try:
            bot.start()
            
            # Garder le processus en vie pour Docker/Kubernetes
            while True:
                time.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("Interruption clavier détectée, arrêt du bot...")
        finally:
            bot.stop()

if __name__ == "__main__":
    main() 