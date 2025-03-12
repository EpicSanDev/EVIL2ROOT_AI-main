import os
import logging
import asyncio
import schedule
import time
from datetime import datetime, timedelta
import threading
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pathlib

from app.telegram_bot import TelegramBot
from app.models.news_retrieval import NewsRetriever
from app.daily_analysis_bot import DailyAnalysisBot
import yfinance as yf

# Importation des modèles existants
from app.models.price_prediction import PricePredictionModel
from app.models.sentiment_analysis import SentimentAnalyzer, MarketRegimeDetector
from app.models.transformer_model import FinancialTransformer
from app.models.risk_management import RiskManagementModel
from app.models.indicator_management import IndicatorManagementModel

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_analysis_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('market_analysis_scheduler')

class MarketAnalysisScheduler:
    """
    Bot qui analyse les marchés toutes les 30 minutes et envoie des signaux via Telegram.
    Récupère également les actualités toutes les 4 heures pour mettre à jour les modèles.
    """
    
    def __init__(self, symbols: List[str]):
        """
        Initialise le scheduler d'analyse de marché
        
        Args:
            symbols: Liste des symboles à analyser
        """
        # Charger les variables d'environnement
        load_dotenv()
        
        # Configuration de base
        self.symbols = symbols
        self.telegram_bot = TelegramBot()
        self.news_retriever = NewsRetriever()
        
        # Initialisation du bot d'analyse (pour utiliser ses fonctionnalités)
        self.analysis_bot = DailyAnalysisBot(symbols)
        
        # Répertoire des modèles
        self.models_dir = os.environ.get('MODELS_DIR', 'saved_models')
        pathlib.Path(self.models_dir).mkdir(exist_ok=True)
        
        # Initialisation des modèles de prédiction
        logger.info("Initialisation des modèles de prédiction...")
        self.price_prediction_models = {}
        self.sentiment_analyzers = {}
        self.market_regime_detectors = {}
        self.risk_managers = {}
        self.indicator_managers = {}
        
        # Initialiser les modèles pour chaque symbole
        for symbol in symbols:
            self.price_prediction_models[symbol] = PricePredictionModel()
            self.sentiment_analyzers[symbol] = SentimentAnalyzer()
            self.market_regime_detectors[symbol] = MarketRegimeDetector()
            self.risk_managers[symbol] = RiskManagementModel()
            self.indicator_managers[symbol] = IndicatorManagementModel()
        
        # Flag pour suivre l'état d'entraînement des modèles
        self.models_trained = False
        
        # Marquer la dernière mise à jour des news
        self.last_news_update = None
        
        logger.info(f"Scheduler d'analyse de marché initialisé avec {len(symbols)} symboles")
    
    def start_scheduled_tasks(self):
        """Démarre les tâches planifiées"""
        # Vérifier et entraîner les modèles avant de commencer
        if not self.models_trained:
            asyncio.run(self.telegram_bot.send_message("🤖 *PRÉPARATION DES MODÈLES* 🤖\n\nEntraînement des modèles en cours... Veuillez patienter."))
            self.train_all_models()
            asyncio.run(self.telegram_bot.send_message("✅ Entraînement des modèles terminé. L'analyse de marché va démarrer."))
        
        # Configurer le planning des analyses (toutes les 30 minutes)
        schedule.every(30).minutes.do(self.analyze_markets)
        
        # Configurer la mise à jour des news (toutes les 4 heures)
        schedule.every(4).hours.do(self.update_models_with_news)
        
        # Exécuter la première analyse et mise à jour immédiatement
        self.analyze_markets()
        self.update_models_with_news()
        
        logger.info("Tâches planifiées configurées: analyse toutes les 30 minutes, mise à jour des news toutes les 4 heures")
        
        # Boucle principale pour exécuter les tâches planifiées
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Vérifier toutes les minutes
        except KeyboardInterrupt:
            logger.info("Scheduler arrêté par l'utilisateur")
    
    def train_all_models(self):
        """Entraîne tous les modèles nécessaires pour les analyses"""
        logger.info("Démarrage de l'entraînement des modèles pour tous les symboles...")
        
        try:
            # Utiliser la méthode d'entraînement du bot d'analyse
            self.analysis_bot.train_all_models()
            
            # Marquer les modèles comme entraînés
            self.models_trained = True
            logger.info("Tous les modèles ont été entraînés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement des modèles: {e}")
            raise
    
    def analyze_markets(self):
        """Analyse les marchés et envoie des signaux via Telegram"""
        logger.info("Démarrage de l'analyse de marché...")
        
        try:
            signals = []
            
            for symbol in self.symbols:
                logger.info(f"Analyse du marché pour {symbol}")
                
                try:
                    # Récupérer les données de marché
                    market_data = self.analysis_bot.fetch_market_data(symbol, period="3mo", interval="1d")
                    
                    # Récupérer les données fondamentales
                    fundamental_data = self.analysis_bot.fetch_fundamental_data(symbol)
                    
                    # Récupérer les news récentes
                    news = self.news_retriever.get_combined_news(symbol, max_results=5)
                    
                    # Faire des prédictions
                    price_predictions = self.analysis_bot.predict_prices(symbol, market_data)
                    
                    # Analyse de sentiment
                    sentiment_analysis = self.analysis_bot.analyze_sentiment(symbol, market_data, news)
                    
                    # Évaluation des risques
                    risk_assessment = self.analysis_bot.assess_risk(symbol, market_data, fundamental_data)
                    
                    # Déterminer si un signal doit être envoyé
                    signal = self._generate_trading_signal(
                        symbol, 
                        market_data, 
                        price_predictions, 
                        sentiment_analysis, 
                        risk_assessment
                    )
                    
                    if signal:
                        signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Erreur lors de l'analyse de {symbol}: {e}")
            
            # Envoyer les signaux via Telegram
            if signals:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                message = f"🔍 *SIGNAUX DE TRADING ({timestamp})* 🔍\n\n"
                message += "\n\n".join(signals)
                
                asyncio.run(self.telegram_bot.send_message(message))
                logger.info(f"Signaux envoyés: {len(signals)} signaux")
            else:
                logger.info("Aucun signal à envoyer")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des marchés: {e}")
            
            # Notification d'erreur
            error_message = f"⚠️ *ERREUR D'ANALYSE* ⚠️\n\nUne erreur est survenue lors de l'analyse des marchés: {str(e)}"
            asyncio.run(self.telegram_bot.send_message(error_message))
    
    def update_models_with_news(self):
        """Met à jour les modèles avec les dernières actualités"""
        logger.info("Mise à jour des modèles avec les dernières actualités...")
        
        try:
            # Mettre à jour la date de dernière mise à jour
            now = datetime.now()
            self.last_news_update = now
            
            # Récupérer les actualités pour chaque symbole
            for symbol in self.symbols:
                logger.info(f"Récupération des actualités pour {symbol}")
                
                try:
                    # Récupérer les données de marché
                    market_data = self.analysis_bot.fetch_market_data(symbol, period="3mo", interval="1d")
                    
                    # Récupérer les news récentes (plus que d'habitude pour l'entraînement)
                    news = self.news_retriever.get_combined_news(symbol, max_results=20)
                    
                    if news:
                        # Mettre à jour le modèle de sentiment
                        logger.info(f"Mise à jour du modèle de sentiment pour {symbol} avec {len(news)} nouvelles actualités")
                        self.sentiment_analyzers[symbol].update(market_data, news)
                        
                        # Enregistrer le modèle mis à jour
                        model_path = os.path.join(self.models_dir, f"{symbol}_sentiment_model")
                        self.sentiment_analyzers[symbol].save(model_path)
                        
                    else:
                        logger.info(f"Aucune actualité trouvée pour {symbol}")
                        
                except Exception as e:
                    logger.error(f"Erreur lors de la mise à jour du modèle pour {symbol}: {e}")
            
            # Envoyer un message de confirmation
            update_message = f"📰 *MISE À JOUR DES MODÈLES* 📰\n\nLes modèles ont été mis à jour avec les dernières actualités.\nProchaine mise à jour dans 4 heures."
            asyncio.run(self.telegram_bot.send_message(update_message))
            
            logger.info("Mise à jour des modèles terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des modèles: {e}")
            
            # Notification d'erreur
            error_message = f"⚠️ *ERREUR DE MISE À JOUR* ⚠️\n\nUne erreur est survenue lors de la mise à jour des modèles: {str(e)}"
            asyncio.run(self.telegram_bot.send_message(error_message))
    
    def _generate_trading_signal(self, symbol, market_data, price_predictions, sentiment_analysis, risk_assessment):
        """
        Génère un signal de trading basé sur les analyses
        
        Returns:
            Signal formaté ou None si pas de signal
        """
        # Extraire les données pertinentes
        current_price = market_data.iloc[-1]['Close'] if not market_data.empty else 0
        
        # Prix prédits
        next_day_price = price_predictions.get('next_day_prediction', current_price)
        three_day_price = price_predictions.get('three_day_prediction', current_price)
        week_prediction = price_predictions.get('week_prediction', current_price)
        
        # Sentiment et régime de marché
        market_sentiment = sentiment_analysis.get('market_sentiment', 'neutral')
        news_sentiment = sentiment_analysis.get('news_sentiment', 'neutral')
        market_regime = sentiment_analysis.get('market_regime', 'neutral')
        
        # Risque
        risk_level = risk_assessment.get('risk_level', 'medium')
        stop_loss = risk_assessment.get('recommended_stop_loss', current_price * 0.95)
        take_profit = risk_assessment.get('recommended_take_profit', current_price * 1.05)
        
        # Calculer la probabilité de mouvement et la direction
        price_change_pct = ((next_day_price / current_price) - 1) * 100
        three_day_change_pct = ((three_day_price / current_price) - 1) * 100
        week_change_pct = ((week_prediction / current_price) - 1) * 100
        
        # Déterminer la direction
        direction = None
        signal_strength = 0
        
        # Règles de génération de signaux
        # 1. Direction basée sur la prédiction du prix
        if price_change_pct > 1.5 or three_day_change_pct > 3 or week_change_pct > 5:
            direction = "ACHAT"
            signal_strength += 1
        elif price_change_pct < -1.5 or three_day_change_pct < -3 or week_change_pct < -5:
            direction = "VENTE"
            signal_strength += 1
        
        # 2. Sentiment influence la force du signal
        if market_sentiment == "bullish" and news_sentiment == "positive":
            if direction == "ACHAT":
                signal_strength += 1
            elif direction == "VENTE":
                signal_strength -= 0.5
        elif market_sentiment == "bearish" and news_sentiment == "negative":
            if direction == "VENTE":
                signal_strength += 1
            elif direction == "ACHAT":
                signal_strength -= 0.5
        
        # 3. Régime de marché influence la force du signal
        if market_regime == "trending_up" and direction == "ACHAT":
            signal_strength += 1
        elif market_regime == "trending_down" and direction == "VENTE":
            signal_strength += 1
        elif market_regime == "volatile":
            signal_strength -= 0.5
        
        # 4. Risque influence la force du signal
        if risk_level == "low":
            signal_strength += 0.5
        elif risk_level == "high":
            signal_strength -= 0.5
        
        # Ne générer un signal que si la force est suffisante
        if direction and signal_strength >= 1.5:
            # Emoji en fonction de la direction
            emoji = "🟢" if direction == "ACHAT" else "🔴"
            
            # Formatage du signal
            signal = f"{emoji} *SIGNAL: {direction} {symbol}* {emoji}\n\n"
            signal += f"📊 *Prix actuel:* {current_price:.2f}\n"
            signal += f"🔮 *Prédiction à 1 jour:* {next_day_price:.2f} ({price_change_pct:.2f}%)\n"
            signal += f"🔮 *Prédiction à 3 jours:* {three_day_price:.2f} ({three_day_change_pct:.2f}%)\n"
            signal += f"🔮 *Prédiction à 1 semaine:* {week_prediction:.2f} ({week_change_pct:.2f}%)\n\n"
            signal += f"🧠 *Sentiment:* {market_sentiment.capitalize()} (marché), {news_sentiment.capitalize()} (news)\n"
            signal += f"📈 *Régime de marché:* {market_regime.replace('_', ' ').capitalize()}\n"
            signal += f"⚠️ *Niveau de risque:* {risk_level.capitalize()}\n\n"
            signal += f"🛑 *Stop Loss recommandé:* {stop_loss:.2f}\n"
            signal += f"🎯 *Take Profit recommandé:* {take_profit:.2f}\n"
            signal += f"💪 *Force du signal:* {signal_strength:.1f}/3.0\n"
            
            return signal
        
        return None

def run_market_analysis_scheduler():
    """Fonction principale pour exécuter le scheduler d'analyse de marché"""
    # Charger la liste des symboles depuis les variables d'environnement
    load_dotenv()
    
    # Lire les symboles à analyser
    symbols_str = os.environ.get('TRADING_SYMBOLS', '')
    if symbols_str:
        symbols = [s.strip() for s in symbols_str.split(',')]
    else:
        # Symboles par défaut si non spécifiés
        symbols = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", 
            "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "ADA-USD",
            "EURUSD=X", "GBPUSD=X", "USDJPY=X"
        ]
    
    logger.info(f"Démarrage du scheduler d'analyse de marché avec {len(symbols)} symboles")
    
    # Créer et démarrer le scheduler
    scheduler = MarketAnalysisScheduler(symbols)
    
    try:
        scheduler.start_scheduled_tasks()
    except KeyboardInterrupt:
        logger.info("Scheduler arrêté par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur critique dans le scheduler: {e}")
        
        # Notification d'erreur critique
        try:
            error_message = f"🚨 *ERREUR CRITIQUE* 🚨\n\nLe scheduler d'analyse de marché a rencontré une erreur critique et s'est arrêté: {str(e)}"
            asyncio.run(scheduler.telegram_bot.send_message(error_message))
        except:
            pass

if __name__ == "__main__":
    run_market_analysis_scheduler() 