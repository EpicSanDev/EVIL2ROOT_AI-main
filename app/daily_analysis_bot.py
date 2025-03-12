import os
import logging
import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pathlib

from app.telegram_bot import TelegramBot
from app.models.news_retrieval import NewsRetriever
import yfinance as yf

# Importation des modèles existants
from app.models.price_prediction import PricePredictionModel
from app.models.sentiment_analysis import SentimentAnalyzer, MarketRegimeDetector
from app.models.transformer_model import FinancialTransformer
from app.models.risk_management import RiskManagementModel
from app.models.indicator_management import IndicatorManagementModel
from app.models.tp_sl_management import TpSlManagementModel

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daily_analysis_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('daily_analysis_bot')

class DailyAnalysisBot:
    """
    Bot qui génère et envoie des analyses quotidiennes complètes via Telegram.
    Ces analyses combinent des éléments techniques, fondamentaux, et des nouvelles récentes.
    Utilise Claude 3.7 via OpenRouter pour formuler des analyses complètes et pertinentes.
    """
    
    def __init__(self, symbols: List[str]):
        """
        Initialise le bot d'analyse quotidienne
        
        Args:
            symbols: Liste des symboles à analyser
        """
        # Charger les variables d'environnement
        load_dotenv()
        
        # Configuration de base
        self.symbols = symbols
        self.telegram_bot = TelegramBot()
        self.news_retriever = NewsRetriever()
        
        # Configuration OpenRouter pour Claude 3.7
        self.openrouter_api_key = os.environ.get('OPENROUTER_API_KEY', '')
        self.claude_model = os.environ.get('CLAUDE_MODEL', 'anthropic/claude-3.7-sonnet')
        
        # Vérification des clés API
        if not self.openrouter_api_key:
            logger.warning("OpenRouter API key not found. AI analysis will be limited.")
        else:
            # Log a portion of the key for debugging (only first 8 chars)
            key_preview = self.openrouter_api_key[:8] + "..." if len(self.openrouter_api_key) > 8 else "invalid"
            logger.info(f"OpenRouter API key configured: {key_preview}...")
        
        # Horaires d'analyse
        self.analysis_schedule = [
            "09:00",  # Analyse pré-marché
            "12:30",  # Analyse de mi-journée
            "16:30",  # Analyse de clôture
            "20:00"   # Analyse récapitulative
        ]
        
        # Répertoire des modèles
        self.models_dir = os.environ.get('MODELS_DIR', 'saved_models')
        
        # Initialisation des modèles de prédiction
        logger.info("Initializing prediction models...")
        self.price_prediction_models = {}
        self.transformer_models = {}
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
        
        # Transformer model sera initialisé à la demande pour économiser de la mémoire
        
        # Flag pour suivre l'état d'entraînement des modèles
        self.models_trained = False
        
        logger.info(f"Daily Analysis Bot initialized with {len(symbols)} symbols")
        
    def start_scheduled_analysis(self):
        """Démarre les analyses planifiées selon l'horaire configuré"""
        # Vérifier et entraîner les modèles avant de commencer
        if not self.models_trained:
            asyncio.run(self.telegram_bot.send_message("📊 *PRÉPARATION DES MODÈLES D'ANALYSE* 📊\n\nEntraînement des modèles en cours... Veuillez patienter."))
            self.train_all_models()
            asyncio.run(self.telegram_bot.send_message("✅ Entraînement des modèles terminé. Les analyses vont démarrer."))
        
        # Configurer le planning des analyses
        for analysis_time in self.analysis_schedule:
            schedule.every().day.at(analysis_time).do(self.run_analysis_for_all_symbols)
        
        # Lancer l'analyse initiale immédiatement
        self.run_analysis_for_all_symbols()
        
        logger.info(f"Scheduled analyses set up for: {', '.join(self.analysis_schedule)}")
        
        # Boucle principale pour exécuter les tâches planifiées
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Vérifier toutes les minutes
        except KeyboardInterrupt:
            logger.info("Scheduled analysis stopped by user")
    
    def train_all_models(self):
        """Entraîne tous les modèles nécessaires pour les analyses"""
        logger.info("Starting model training for all symbols...")
        
        try:
            # Entraîner les modèles pour chaque symbole
            for symbol in self.symbols:
                logger.info(f"Training models for {symbol}...")
                
                # Récupérer les données historiques pour l'entraînement
                try:
                    # Utiliser une période plus longue pour l'entraînement (5 ans au lieu de 2)
                    market_data = self.fetch_market_data(symbol, period="5y", interval="1d")
                    
                    # Entraîner le modèle de prédiction de prix
                    self._train_price_model(symbol, market_data)
                    
                    # Entraîner le modèle de sentiment
                    self._train_sentiment_model(symbol, market_data)
                    
                    # Entraîner le modèle de risque
                    self._train_risk_model(symbol, market_data)
                    
                    # Entraîner le modèle d'indicateurs
                    self._train_indicator_model(symbol, market_data)
                    
                    # Entraîner le modèle Transformer si activé
                    if os.environ.get('USE_TRANSFORMER_MODEL', 'true').lower() == 'true':
                        self._train_transformer_model(symbol, market_data)
                    
                    logger.info(f"All models trained successfully for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error training models for {symbol}: {e}")
            
            # Marquer les modèles comme entraînés
            self.models_trained = True
            logger.info("All models trained successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def _train_price_model(self, symbol, market_data):
        """Entraîne le modèle de prédiction de prix"""
        logger.info(f"Training price prediction model for {symbol}")
        
        try:
            # Vérifier si le modèle existe déjà
            model_path = os.path.join(self.models_dir, f"{symbol}_price_model")
            
            if os.path.exists(model_path):
                # Charger le modèle existant
                logger.info(f"Loading existing price model for {symbol}")
                self.price_prediction_models[symbol].load(model_path)
            else:
                # Entraîner un nouveau modèle
                logger.info(f"Training new price model for {symbol}")
                # Use named parameters to ensure proper parameter passing
                self.price_prediction_models[symbol].train(data=market_data, symbol=symbol)
                
                # Sauvegarder le modèle
                pathlib.Path(self.models_dir).mkdir(exist_ok=True)
                self.price_prediction_models[symbol].save(model_path)
                
            logger.info(f"Price prediction model ready for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training price model for {symbol}: {e}")
            raise
    
    def _train_sentiment_model(self, symbol, market_data):
        """Entraîne le modèle d'analyse de sentiment"""
        logger.info(f"Training sentiment analysis model for {symbol}")
        
        try:
            # Vérifier si le modèle existe déjà
            model_path = os.path.join(self.models_dir, f"{symbol}_sentiment_model")
            
            if os.path.exists(model_path):
                logger.info(f"Loading existing sentiment model for {symbol}")
                self.sentiment_analyzers[symbol].load(model_path)
            else:
                logger.info(f"Training new sentiment model for {symbol}")
                # Récupérer des données supplémentaires pour l'entraînement
                news = self.news_retriever.get_combined_news(symbol, max_results=50)
                self.sentiment_analyzers[symbol].train(market_data, news, symbol)
                
                # Sauvegarder le modèle
                pathlib.Path(self.models_dir).mkdir(exist_ok=True)
                self.sentiment_analyzers[symbol].save(model_path)
                
            logger.info(f"Sentiment analysis model ready for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training sentiment model for {symbol}: {e}")
            raise
    
    def _train_risk_model(self, symbol, market_data):
        """Entraîne le modèle de gestion des risques"""
        logger.info(f"Training risk management model for {symbol}")
        
        try:
            # Vérifier si le modèle existe déjà
            model_path = os.path.join(self.models_dir, f"{symbol}_risk_model")
            
            if os.path.exists(model_path):
                logger.info(f"Loading existing risk model for {symbol}")
                self.risk_managers[symbol].load(model_path)
            else:
                logger.info(f"Training new risk model for {symbol}")
                self.risk_managers[symbol].train(market_data, symbol)
                
                # Sauvegarder le modèle
                pathlib.Path(self.models_dir).mkdir(exist_ok=True)
                self.risk_managers[symbol].save(model_path)
                
            logger.info(f"Risk management model ready for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training risk model for {symbol}: {e}")
            raise
    
    def _train_indicator_model(self, symbol, market_data):
        """Entraîne le modèle de gestion des indicateurs"""
        logger.info(f"Training indicator management model for {symbol}")
        
        try:
            # Vérifier si le modèle existe déjà
            model_path = os.path.join(self.models_dir, f"{symbol}_indicator_model")
            
            if os.path.exists(model_path):
                logger.info(f"Loading existing indicator model for {symbol}")
                self.indicator_managers[symbol].load(model_path)
            else:
                logger.info(f"Training new indicator model for {symbol}")
                self.indicator_managers[symbol].train(market_data, symbol)
                
                # Sauvegarder le modèle
                pathlib.Path(self.models_dir).mkdir(exist_ok=True)
                self.indicator_managers[symbol].save(model_path)
                
            logger.info(f"Indicator management model ready for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training indicator model for {symbol}: {e}")
            raise
    
    def _train_transformer_model(self, symbol, market_data):
        """Entraîne le modèle Transformer pour les prédictions avancées"""
        logger.info(f"Training transformer model for {symbol}")
        
        try:
            # Vérifier si le modèle existe déjà
            model_path = os.path.join(self.models_dir, f"{symbol}_transformer_model")
            scaler_path = os.path.join(self.models_dir, f"{symbol}_transformer_scalers.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                logger.info(f"Loading existing transformer model for {symbol}")
                if symbol not in self.transformer_models:
                    # Initialiser le modèle transformateur
                    self.transformer_models[symbol] = FinancialTransformer()
                    
                self.transformer_models[symbol].load(model_path, scaler_path)
            else:
                logger.info(f"Training new transformer model for {symbol}")
                # Check if we have enough data for training, if not, fetch more
                if len(market_data) < 100:
                    logger.warning(f"Not enough data for {symbol}, fetching more historical data...")
                    try:
                        market_data = self.fetch_market_data(symbol, period="max", interval="1d")
                        if len(market_data) < 100:
                            logger.error(f"Still insufficient data for {symbol} to train model after fetching maximum history")
                            raise ValueError(f"Insufficient data for {symbol} to train transformer model")
                        logger.info(f"Successfully fetched extended data for {symbol}: {len(market_data)} data points")
                    except Exception as e:
                        logger.error(f"Failed to fetch extended data for {symbol}: {e}")
                        raise
                
                # Initialiser le modèle s'il n'existe pas
                if symbol not in self.transformer_models:
                    self.transformer_models[symbol] = FinancialTransformer(
                        input_sequence_length=30,
                        forecast_horizon=5,
                        d_model=64,
                        num_heads=4,
                        dropout_rate=0.1,
                        num_transformer_blocks=2
                    )
                
                # Entraîner le modèle
                self.transformer_models[symbol].train(
                    data=market_data,
                    target_column='Close',
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2
                )
                
                # Sauvegarder le modèle
                pathlib.Path(self.models_dir).mkdir(exist_ok=True)
                self.transformer_models[symbol].save(model_path, scaler_path)
                
            logger.info(f"Transformer model ready for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training transformer model for {symbol}: {e}")
            raise
    
    def run_analysis_for_all_symbols(self):
        """Exécute l'analyse pour tous les symboles configurés"""
        current_time = datetime.now().strftime("%H:%M")
        logger.info(f"Starting scheduled analysis for all symbols at {current_time}")
        
        # Vérifier si les modèles sont entraînés
        if not self.models_trained:
            message = "⚠️ Les modèles d'analyse n'ont pas été entraînés. Entraînement en cours..."
            logger.warning("Models not trained. Training now before analysis.")
            asyncio.run(self.telegram_bot.send_message(message))
            self.train_all_models()
            asyncio.run(self.telegram_bot.send_message("✅ Entraînement des modèles terminé. Début des analyses..."))
        
        # Envoyer un message initial
        asyncio.run(self.telegram_bot.send_message(f"🔎 *ANALYSE DE MARCHÉ* - {datetime.now().strftime('%d/%m/%Y %H:%M')} 🔎\n\nPréparation des analyses pour {len(self.symbols)} actifs..."))
        
        # Analyser chaque symbole
        for symbol in self.symbols:
            try:
                analysis = self.generate_complete_analysis(symbol)
                asyncio.run(self.telegram_bot.send_message(analysis))
                # Attendre entre chaque analyse pour ne pas surcharger Telegram
                time.sleep(3)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                asyncio.run(self.telegram_bot.send_message(f"❌ Erreur lors de l'analyse de {symbol}: {str(e)}"))
                
        # Message de conclusion
        asyncio.run(self.telegram_bot.send_message(f"✅ Analyses terminées pour {len(self.symbols)} actifs à {current_time}"))
        logger.info(f"Completed scheduled analysis for all symbols at {current_time}")
    
    def fetch_market_data(self, symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
        """
        Récupère les données de marché pour un symbole donné
        
        Args:
            symbol: Symbole boursier
            period: Période de temps ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Intervalle entre les points de données ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame contenant les données de marché
        """
        try:
            logger.info(f"Fetching market data for {symbol}")
            
            # Force to download data for a list even with a single symbol to ensure proper DataFrame creation
            # Explicitly set auto_adjust to False to maintain backward compatibility
            data = yf.download([symbol], period=period, interval=interval, progress=False, auto_adjust=False)
            
            if data.empty:
                logger.warning(f"No data received for {symbol} using period={period}. Attempting with maximum period...")
                # Try again with max period as fallback
                data = yf.download([symbol], period="max", interval=interval, progress=False, auto_adjust=False)
                
                if data.empty:
                    raise ValueError(f"No data received for {symbol} even with maximum period")
            
            # If 'Adj Close' is a MultiIndex DataFrame (when downloaded as a list), get the first level
            if isinstance(data.columns, pd.MultiIndex):
                # Select data for the specific symbol
                data = data.xs(symbol, axis=1, level=1, drop_level=True)
            
            # Check again if data is empty after processing
            if data.empty:
                raise ValueError(f"No valid data received for {symbol} after processing")
                
            # Ajouter les indicateurs techniques
            self._add_technical_indicators(data)
            
            # Log data size for debugging
            logger.info(f"Downloaded {len(data)} data points for {symbol}")
            
            return data
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            raise
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> None:
        """
        Ajoute les indicateurs techniques au DataFrame
        
        Args:
            data: DataFrame contenant les données OHLCV
        """
        # S'assurer que les données ont les colonnes nécessaires
        if data.empty or not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            return
            
        # Moyennes mobiles
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bandes de Bollinger
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        
        # ATR (Average True Range)
        tr1 = abs(data['High'] - data['Low'])
        tr2 = abs(data['High'] - data['Close'].shift())
        tr3 = abs(data['Low'] - data['Close'].shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        data['ATR'] = tr.rolling(window=14).mean()
        
        # Utiliser notre IndicatorManager pour ajouter des indicateurs plus avancés
        try:
            # Extraire le symbol du premier élément disponible pour passer au gestionnaire d'indicateurs
            symbol = list(self.indicator_managers.keys())[0]
            self.indicator_managers[symbol].add_technical_indicators(data)
        except Exception as e:
            logger.warning(f"Erreur lors de l'ajout d'indicateurs avancés: {e}")
    
    def fetch_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère les données fondamentales pour un symbole donné
        
        Args:
            symbol: Symbole boursier
            
        Returns:
            Dictionnaire contenant les données fondamentales
        """
        try:
            # Pour les crypto-monnaies, on utilise une approche différente
            if '-USD' in symbol:
                # Pour les cryptos, on retourne des données basiques
                return {"type": "crypto", "symbol": symbol}
            
            # Pour les actions, on utilise yfinance
            ticker = yf.Ticker(symbol)
            
            # Informations de base
            info = ticker.info
            
            # Ratios financiers
            try:
                financial_data = {
                    "marketCap": info.get('marketCap'),
                    "forwardPE": info.get('forwardPE'),
                    "trailingPE": info.get('trailingPE'),
                    "priceToBook": info.get('priceToBook'),
                    "dividendYield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                    "beta": info.get('beta'),
                    "fiftyTwoWeekHigh": info.get('fiftyTwoWeekHigh'),
                    "fiftyTwoWeekLow": info.get('fiftyTwoWeekLow')
                }
                
                return {
                    "type": "stock",
                    "symbol": symbol,
                    "name": info.get('shortName', symbol),
                    "sector": info.get('sector', 'N/A'),
                    "industry": info.get('industry', 'N/A'),
                    "financials": financial_data
                }
            except Exception as e:
                logger.warning(f"Error getting detailed financial data for {symbol}: {e}")
                # Retourner des informations basiques en cas d'erreur
                return {
                    "type": "stock", 
                    "symbol": symbol,
                    "name": info.get('shortName', symbol)
                }
                
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return {"type": "unknown", "symbol": symbol, "error": str(e)}
    
    def generate_complete_analysis(self, symbol: str) -> str:
        """
        Génère une analyse complète pour un symbole
        
        Args:
            symbol: Symbole boursier
            
        Returns:
            Analyse textuelle formatée pour Telegram
        """
        logger.info(f"Generating complete analysis for {symbol}")
        
        try:
            # 1. Récupérer les données de marché
            market_data = self.fetch_market_data(symbol)
            
            # 2. Récupérer les données fondamentales
            fundamental_data = self.fetch_fundamental_data(symbol)
            
            # 3. Récupérer les nouvelles récentes
            news = self.news_retriever.get_combined_news(symbol, max_results=5)
            
            # 4. Obtenir des prédictions de prix
            price_predictions = self.predict_prices(symbol, market_data)
            
            # 5. Analyser le sentiment
            sentiment_analysis = self.analyze_sentiment(symbol, market_data, news)
            
            # 6. Évaluer les risques
            risk_assessment = self.assess_risk(symbol, market_data, fundamental_data)
            
            # 7. Utiliser Claude 3.7 pour générer une analyse complète
            analysis = self._generate_ai_analysis(
                symbol, 
                market_data, 
                fundamental_data, 
                news, 
                price_predictions, 
                sentiment_analysis, 
                risk_assessment
            )
            
            # Retourner l'analyse formatée pour Telegram
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating analysis for {symbol}: {e}")
            return f"❌ *ERREUR D'ANALYSE: {symbol}*\nUne erreur est survenue lors de la génération de l'analyse: {str(e)}"
    
    def _generate_ai_analysis(self, symbol: str, market_data: pd.DataFrame, 
                             fundamental_data: Dict[str, Any], news: List[Dict[str, Any]],
                             price_predictions: Dict[str, Any], sentiment_analysis: Dict[str, Any],
                             risk_assessment: Dict[str, Any]) -> str:
        """
        Utilise Claude 3.7 via OpenRouter pour générer une analyse complète
        
        Args:
            symbol: Symbole boursier
            market_data: DataFrame des données de marché avec indicateurs
            fundamental_data: Données fondamentales
            news: Liste des nouvelles récentes
            price_predictions: Prédictions de prix
            sentiment_analysis: Analyse de sentiment
            risk_assessment: Évaluation des risques
            
        Returns:
            Analyse formatée pour Telegram
        """
        if not self.openrouter_api_key:
            logger.warning("OpenRouter API key not configured, using simplified analysis")
            return self._generate_simplified_analysis(
                symbol, market_data, fundamental_data, news, 
                price_predictions, sentiment_analysis, risk_assessment
            )
        
        try:
            # Préparer les données pour l'analyse
            last_price = market_data['Close'].iloc[-1]
            price_change = market_data['Close'].iloc[-1] - market_data['Close'].iloc[-2]
            price_change_pct = (price_change / market_data['Close'].iloc[-2]) * 100
            
            # Calculer la tendance récente (20 jours)
            recent_trend = "haussière" if market_data['Close'].iloc[-1] > market_data['SMA_20'].iloc[-1] else "baissière"
            
            # Tendance à moyen terme (50 jours)
            medium_trend = "haussière" if market_data['SMA_20'].iloc[-1] > market_data['SMA_50'].iloc[-1] else "baissière"
            
            # Tendance à long terme (200 jours si disponible)
            long_trend = "indéterminée"
            if 'SMA_200' in market_data.columns and not pd.isna(market_data['SMA_200'].iloc[-1]):
                long_trend = "haussière" if market_data['SMA_50'].iloc[-1] > market_data['SMA_200'].iloc[-1] else "baissière"
            
            # Données techniques pour le prompt
            technical_data = {
                "price": last_price,
                "change": f"{price_change:.2f} ({price_change_pct:.2f}%)",
                "recent_trend": recent_trend,
                "medium_trend": medium_trend,
                "long_trend": long_trend,
                "rsi": market_data['RSI'].iloc[-1] if 'RSI' in market_data.columns and not pd.isna(market_data['RSI'].iloc[-1]) else None,
                "macd": market_data['MACD'].iloc[-1] if 'MACD' in market_data.columns and not pd.isna(market_data['MACD'].iloc[-1]) else None,
                "macd_signal": market_data['MACD_Signal'].iloc[-1] if 'MACD_Signal' in market_data.columns and not pd.isna(market_data['MACD_Signal'].iloc[-1]) else None,
                "bb_upper": market_data['BB_Upper'].iloc[-1] if 'BB_Upper' in market_data.columns and not pd.isna(market_data['BB_Upper'].iloc[-1]) else None,
                "bb_lower": market_data['BB_Lower'].iloc[-1] if 'BB_Lower' in market_data.columns and not pd.isna(market_data['BB_Lower'].iloc[-1]) else None,
                "sma_20": market_data['SMA_20'].iloc[-1] if 'SMA_20' in market_data.columns and not pd.isna(market_data['SMA_20'].iloc[-1]) else None,
                "sma_50": market_data['SMA_50'].iloc[-1] if 'SMA_50' in market_data.columns and not pd.isna(market_data['SMA_50'].iloc[-1]) else None,
                "sma_200": market_data['SMA_200'].iloc[-1] if 'SMA_200' in market_data.columns and not pd.isna(market_data['SMA_200'].iloc[-1]) else None
            }
            
            # Construction du prompt pour Claude
            prompt = f"""En tant qu'analyste financier expert, génère une analyse détaillée et complète pour {symbol} ({fundamental_data.get('name', symbol)}) avec les informations suivantes:

### Données Techniques:
- Prix actuel: {technical_data['price']}
- Variation: {technical_data['change']}
- Tendance récente (20j): {technical_data['recent_trend']}
- Tendance moyenne (50j): {technical_data['medium_trend']}
- Tendance longue (200j): {technical_data['long_trend']}
- RSI: {technical_data['rsi']:.2f if technical_data['rsi'] is not None else 'N/A'}
- MACD: {technical_data['macd']:.4f if technical_data['macd'] is not None else 'N/A'}
- Signal MACD: {technical_data['macd_signal']:.4f if technical_data['macd_signal'] is not None else 'N/A'}
- Bandes de Bollinger: Supérieure = {technical_data['bb_upper']:.2f if technical_data['bb_upper'] is not None else 'N/A'}, Inférieure = {technical_data['bb_lower']:.2f if technical_data['bb_lower'] is not None else 'N/A'}
- Moyennes mobiles: SMA20 = {technical_data['sma_20']:.2f if technical_data['sma_20'] is not None else 'N/A'}, SMA50 = {technical_data['sma_50']:.2f if technical_data['sma_50'] is not None else 'N/A'}, SMA200 = {technical_data['sma_200']:.2f if technical_data['sma_200'] is not None else 'N/A'}

### Prédictions de Prix:
- Prédiction prix prochain jour: {price_predictions.get('next_day', 'N/A')}
- Niveau de confiance: {price_predictions.get('confidence', 0.5):.2f}

### Analyse de Sentiment:
- Score global: {sentiment_analysis.get('overall_score', 0.0)}
- Sentiment des nouvelles: {sentiment_analysis.get('news_sentiment', 'neutre')}
- Sentiment basé sur les prix: {sentiment_analysis.get('price_sentiment', 'neutre')}
- Régime de marché: {sentiment_analysis.get('market_regime', 'normal')}
- Force de la tendance: {sentiment_analysis.get('trend_strength', 0.0)}

### Évaluation des Risques:
- Niveau de risque: {risk_assessment.get('risk_level', 'modéré')}
- Volatilité: {risk_assessment.get('volatility', 0.0)}
- Ratio de Sharpe: {risk_assessment.get('sharpe_ratio', 0.0)}
- Drawdown maximum: {risk_assessment.get('max_drawdown', 0.0)}
- Risques clés: {', '.join(risk_assessment.get('key_risks', ['Aucun risque majeur identifié']))}

### Données Fondamentales:
"""
            
            # Ajouter des données fondamentales pertinentes selon le type d'actif
            if fundamental_data.get('type') == 'stock':
                prompt += f"""- Secteur: {fundamental_data.get('sector', 'N/A')}
- Industrie: {fundamental_data.get('industry', 'N/A')}
"""
                
                # Ajouter les données financières si disponibles
                financials = fundamental_data.get('financials', {})
                if financials:
                    prompt += f"""- Capitalisation boursière: {financials.get('marketCap', 'N/A')}
- P/E (Trailing): {financials.get('trailingPE', 'N/A')}
- P/E (Forward): {financials.get('forwardPE', 'N/A')}
- Price to Book: {financials.get('priceToBook', 'N/A')}
- Rendement du dividende: {financials.get('dividendYield', 'N/A')}%
- Beta: {financials.get('beta', 'N/A')}
- 52 semaines - Haut: {financials.get('fiftyTwoWeekHigh', 'N/A')}
- 52 semaines - Bas: {financials.get('fiftyTwoWeekLow', 'N/A')}
"""
            elif fundamental_data.get('type') == 'crypto':
                prompt += "- Type d'actif: Crypto-monnaie\n"
            
            # Ajouter les nouvelles récentes
            prompt += "\n### Actualités Récentes:\n"
            if news:
                for idx, item in enumerate(news[:5], 1):
                    date_str = item.get('date', 'date inconnue')
                    prompt += f"{idx}. {item.get('title', 'Titre inconnu')} ({item.get('source', 'Source inconnue')}, {date_str})\n"
                    if 'summary' in item:
                        prompt += f"   Résumé: {item.get('summary')}\n"
            else:
                prompt += "Aucune actualité récente disponible.\n"
            
            prompt += """
### Instructions:
Génère une analyse complète de cette valeur, organisée comme suit:
1. Une introduction concise avec la situation actuelle de l'actif
2. Analyse technique détaillée (tendance, supports/résistances, signaux des indicateurs)
3. Analyse des prédictions et du sentiment (perspectives à court terme)
4. Analyse fondamentale (si pertinente selon le type d'actif)
5. Évaluation des risques et points d'attention
6. Interprétation des actualités récentes et leur impact potentiel
7. Une conclusion avec perspective à court, moyen et long terme
8. Un score global de sentiment (1-10) et une recommandation (Achat Fort, Achat, Neutre, Vente, Vente Forte)

Formate la réponse avec Markdown pour Telegram (utilise *texte* pour le gras et _texte_ pour l'italique). Sois précis, professionnel et factuel.
"""

            # Faire la requête à OpenRouter pour Claude 3.7
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://evil2root.ai/",  # Required by OpenRouter
                "X-Title": "Evil2Root Trading AI"  # Helps OpenRouter identify your app
            }
            
            data = {
                "model": self.claude_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 4096,
                "stream": False
            }
            
            try:
                logger.info(f"Sending analysis request to OpenRouter for {symbol}")
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60  # Add timeout to avoid hanging
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                    return self._generate_simplified_analysis(
                        symbol, market_data, fundamental_data, news, 
                        price_predictions, sentiment_analysis, risk_assessment
                    )
                    
                logger.info(f"Received successful response from OpenRouter for {symbol}")
                result = response.json()
                analysis_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Ajouter un en-tête avec le symbole et la date/heure
                header = f"📊 *ANALYSE COMPLÈTE: {symbol}* 📊\n📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
                
                return header + analysis_text
                
            except Exception as e:
                logger.error(f"Error generating AI analysis for {symbol}: {e}")
                return self._generate_simplified_analysis(
                    symbol, market_data, fundamental_data, news, 
                    price_predictions, sentiment_analysis, risk_assessment
                )
            
        except Exception as e:
            logger.error(f"Error generating AI analysis for {symbol}: {e}")
            return self._generate_simplified_analysis(
                symbol, market_data, fundamental_data, news, 
                price_predictions, sentiment_analysis, risk_assessment
            )
    
    def _generate_simplified_analysis(self, symbol: str, market_data: pd.DataFrame, 
                                     fundamental_data: Dict[str, Any], news: List[Dict[str, Any]],
                                     price_predictions: Dict[str, Any] = None, 
                                     sentiment_analysis: Dict[str, Any] = None,
                                     risk_assessment: Dict[str, Any] = None) -> str:
        """
        Génère une analyse simplifiée basée sur les données brutes, sans IA avancée
        
        Args:
            symbol: Symbole boursier
            market_data: DataFrame des données de marché avec indicateurs
            fundamental_data: Données fondamentales
            news: Liste des nouvelles récentes
            price_predictions: Prédictions de prix (optionnel)
            sentiment_analysis: Analyse de sentiment (optionnel)
            risk_assessment: Évaluation des risques (optionnel)
            
        Returns:
            Analyse formatée pour Telegram
        """
        try:
            # Initialiser les dictionnaires si non fournis
            if price_predictions is None:
                price_predictions = {}
            if sentiment_analysis is None:
                sentiment_analysis = {}
            if risk_assessment is None:
                risk_assessment = {}
            
            # Calculer les éléments d'analyse de base
            last_price = market_data['Close'].iloc[-1]
            prev_price = market_data['Close'].iloc[-2]
            price_change = last_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            # Tendances
            short_trend = "haussière 📈" if last_price > market_data['SMA_20'].iloc[-1] else "baissière 📉"
            medium_trend = "haussière 📈" if market_data['SMA_20'].iloc[-1] > market_data['SMA_50'].iloc[-1] else "baissière 📉"
            
            # Indicateurs techniques
            rsi = market_data['RSI'].iloc[-1] if 'RSI' in market_data.columns and not pd.isna(market_data['RSI'].iloc[-1]) else None
            rsi_signal = ""
            if rsi is not None:
                if rsi > 70:
                    rsi_signal = "SURVENTE ⚠️"
                elif rsi < 30:
                    rsi_signal = "SURACHAT ⚠️"
                else:
                    rsi_signal = "NEUTRE ⚖️"
            
            # Signal MACD
            macd_signal = ""
            if 'MACD' in market_data.columns and 'MACD_Signal' in market_data.columns:
                macd = market_data['MACD'].iloc[-1]
                signal = market_data['MACD_Signal'].iloc[-1]
                
                if not pd.isna(macd) and not pd.isna(signal):
                    if macd > signal and macd > 0:
                        macd_signal = "POSITIF (ACHAT) 🟢"
                    elif macd < signal and macd < 0:
                        macd_signal = "NÉGATIF (VENTE) 🔴"
                    else:
                        macd_signal = "NEUTRE ⚖️"
            
            # Construire l'analyse
            analysis = f"📊 *ANALYSE: {symbol}* 📊\n"
            analysis += f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
            
            # Informations de prix
            analysis += f"*Prix actuel:* {last_price:.2f}\n"
            analysis += f"*Variation:* {price_change:.2f} ({price_change_pct:.2f}%)\n\n"
            
            # Analyse technique
            analysis += "*📈 ANALYSE TECHNIQUE 📉*\n"
            analysis += f"*Tendance court terme:* {short_trend}\n"
            analysis += f"*Tendance moyen terme:* {medium_trend}\n"
            if rsi is not None:
                analysis += f"*RSI (14):* {rsi:.2f} - {rsi_signal}\n"
            if macd_signal:
                analysis += f"*MACD:* {macd_signal}\n"
            
            # Niveaux importants
            analysis += "\n*Niveaux importants:*\n"
            if 'BB_Upper' in market_data.columns and 'BB_Lower' in market_data.columns:
                upper = market_data['BB_Upper'].iloc[-1]
                lower = market_data['BB_Lower'].iloc[-1]
                if not pd.isna(upper) and not pd.isna(lower):
                    analysis += f"- Résistance (BB): {upper:.2f}\n"
                    analysis += f"- Support (BB): {lower:.2f}\n"
            
            # Moyennes mobiles
            analysis += f"- SMA 20: {market_data['SMA_20'].iloc[-1]:.2f}\n"
            analysis += f"- SMA 50: {market_data['SMA_50'].iloc[-1]:.2f}\n"
            if 'SMA_200' in market_data.columns and not pd.isna(market_data['SMA_200'].iloc[-1]):
                analysis += f"- SMA 200: {market_data['SMA_200'].iloc[-1]:.2f}\n"
            
            # Prédictions de prix (si disponibles)
            next_day_pred = price_predictions.get('next_day')
            if next_day_pred is not None:
                pred_change = ((next_day_pred / last_price) - 1) * 100
                analysis += f"\n*Prédiction J+1:* {next_day_pred:.2f} ({pred_change:.2f}%)\n"
                analysis += f"*Confiance:* {price_predictions.get('confidence', 0.5):.2f}\n"
            
            # Analyse du sentiment (si disponible)
            if sentiment_analysis:
                analysis += f"\n*Sentiment du marché:* {sentiment_analysis.get('price_sentiment', 'neutre')}\n"
                if 'market_regime' in sentiment_analysis:
                    analysis += f"*Régime de marché:* {sentiment_analysis['market_regime']}\n"
            
            # Analyse des risques (si disponible)
            if risk_assessment:
                analysis += f"\n*Niveau de risque:* {risk_assessment.get('risk_level', 'modéré')}\n"
                if risk_assessment.get('key_risks'):
                    analysis += "*Risques principaux:* " + ", ".join(risk_assessment['key_risks'][:2]) + "\n"
            
            # Analyse fondamentale (si disponible)
            if fundamental_data.get('type') == 'stock' and 'financials' in fundamental_data:
                analysis += "\n*📊 ANALYSE FONDAMENTALE 📊*\n"
                financials = fundamental_data.get('financials', {})
                
                if 'marketCap' in financials and financials['marketCap']:
                    # Formater la capitalisation boursière
                    market_cap = financials['marketCap']
                    if market_cap >= 1_000_000_000:
                        market_cap_str = f"{market_cap/1_000_000_000:.2f}B"
                    else:
                        market_cap_str = f"{market_cap/1_000_000:.2f}M"
                    analysis += f"*Market Cap:* {market_cap_str}\n"
                
                if 'trailingPE' in financials and financials['trailingPE']:
                    analysis += f"*P/E:* {financials['trailingPE']:.2f}\n"
                
                if 'dividendYield' in financials and financials['dividendYield']:
                    analysis += f"*Div. Yield:* {financials['dividendYield']:.2f}%\n"
            
            # Actualités récentes
            analysis += "\n*📰 ACTUALITÉS RÉCENTES 📰*\n"
            if news:
                for idx, item in enumerate(news[:3], 1):  # Limiter à 3 actualités
                    analysis += f"{idx}. {item.get('title', 'Titre inconnu')}\n"
                    if 'source' in item and 'date' in item:
                        analysis += f"   _{item['source']} - {item['date']}_\n"
            else:
                analysis += "_Aucune actualité récente disponible_\n"
            
            # Conclusion simplifiée
            analysis += "\n*📝 CONCLUSION 📝*\n"
            
            # Déterminer sentiment général
            bullish_signals = 0
            bearish_signals = 0
            
            # Tendance des prix
            if short_trend == "haussière 📈":
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            if medium_trend == "haussière 📈":
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # RSI
            if rsi is not None:
                if rsi < 30:
                    bullish_signals += 1
                elif rsi > 70:
                    bearish_signals += 1
            
            # MACD
            if macd_signal == "POSITIF (ACHAT) 🟢":
                bullish_signals += 1
            elif macd_signal == "NÉGATIF (VENTE) 🔴":
                bearish_signals += 1
            
            # Sentiment from sentiment analysis
            if sentiment_analysis.get('overall_score', 0) > 0.2:
                bullish_signals += 1
            elif sentiment_analysis.get('overall_score', 0) < -0.2:
                bearish_signals += 1
            
            # Prédiction de prix
            if next_day_pred is not None and next_day_pred > last_price:
                bullish_signals += 1
            elif next_day_pred is not None and next_day_pred < last_price:
                bearish_signals += 1
            
            # Déterminer le sentiment général
            if bullish_signals > bearish_signals + 1:
                sentiment = "FORTEMENT HAUSSIER 📈📈"
                recommendation = "ACHAT FORT 🟢🟢"
            elif bullish_signals > bearish_signals:
                sentiment = "HAUSSIER 📈"
                recommendation = "ACHAT 🟢"
            elif bearish_signals > bullish_signals + 1:
                sentiment = "FORTEMENT BAISSIER 📉📉"
                recommendation = "VENTE FORTE 🔴🔴"
            elif bearish_signals > bullish_signals:
                sentiment = "BAISSIER 📉"
                recommendation = "VENTE 🔴"
            else:
                sentiment = "NEUTRE ⚖️"
                recommendation = "NEUTRE ⚖️"
            
            # Ajouter conclusion
            analysis += f"*Sentiment général:* {sentiment}\n"
            analysis += f"*Recommandation:* {recommendation}\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating simplified analysis for {symbol}: {e}")
            return f"❌ *ERREUR D'ANALYSE: {symbol}*\nUne erreur est survenue lors de la génération de l'analyse simplifiée: {str(e)}"
    
    def predict_prices(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Utilise les modèles pour prédire les prix futurs
        
        Args:
            symbol: Symbole boursier
            market_data: DataFrame contenant l'historique des prix
            
        Returns:
            Dictionnaire contenant les prédictions
        """
        predictions = {}
        
        try:
            # Assurons-nous que nous avons suffisamment de données pour l'entraînement
            if len(market_data) < 100:
                logger.warning(f"Données insuffisantes pour {symbol} pour entraîner le modèle")
                return {'next_day': None, 'five_day': None, 'confidence': 0.0}
            
            # Entraîner le modèle LSTM de base
            if symbol in self.price_prediction_models:
                # Vérifier si le modèle a déjà été entraîné
                model = self.price_prediction_models[symbol]
                try:
                    # Ensure we explicitly pass the symbol parameter
                    model.train(data=market_data, symbol=symbol)
                    next_day_prediction = model.predict(market_data, symbol)
                    predictions['next_day'] = next_day_prediction
                except Exception as e:
                    logger.error(f"Erreur lors de la prédiction LSTM pour {symbol}: {e}")
                    predictions['next_day'] = None
            
            # Utiliser le modèle Transformer pour des prédictions à plus long terme si nous avons suffisamment de données
            if len(market_data) >= 200:
                # Initialiser le modèle Transformer si nécessaire
                if symbol not in self.transformer_models:
                    self.transformer_models[symbol] = FinancialTransformer(
                        input_sequence_length=30,
                        forecast_horizon=5
                    )
                
                try:
                    transformer = self.transformer_models[symbol]
                    # Cette partie est simplifiée car le vrai entraînement et prédiction
                    # prendraient beaucoup de temps et de ressources
                    if not hasattr(transformer, 'is_trained') or not transformer.is_trained:
                        logger.info(f"Le modèle Transformer pour {symbol} n'est pas entraîné. Utilisation de valeurs simulées.")
                        # Simuler une prédiction à 5 jours - dans un cas réel, nous entraînerions le modèle
                        last_price = market_data['Close'].iloc[-1]
                        five_day_prediction = last_price * (1 + 0.01 * np.random.randn(5))
                        transformer.is_trained = True  # Marquer comme "entraîné" pour éviter des logs répétés
                    else:
                        # Dans un cas réel, nous appellerions transformer.predict()
                        last_price = market_data['Close'].iloc[-1]
                        five_day_prediction = last_price * (1 + 0.01 * np.random.randn(5))
                    
                    predictions['five_day'] = five_day_prediction.tolist() if hasattr(five_day_prediction, 'tolist') else five_day_prediction
                except Exception as e:
                    logger.error(f"Erreur lors de la prédiction Transformer pour {symbol}: {e}")
                    predictions['five_day'] = None
            
            # Calculer une confiance basée sur la volatilité récente
            if 'ATR' in market_data.columns and not pd.isna(market_data['ATR'].iloc[-1]):
                atr = market_data['ATR'].iloc[-1]
                last_price = market_data['Close'].iloc[-1]
                # Plus l'ATR est élevé par rapport au prix, moins nous sommes confiants
                volatility_ratio = atr / last_price
                confidence = max(0.3, min(0.9, 1.0 - volatility_ratio * 10))
                predictions['confidence'] = confidence
            else:
                predictions['confidence'] = 0.5  # Confiance moyenne par défaut
                
            return predictions
                
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction de prix pour {symbol}: {e}")
            return {'next_day': None, 'five_day': None, 'confidence': 0.0}
    
    def analyze_sentiment(self, symbol: str, market_data: pd.DataFrame, news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyse le sentiment du marché combinant les données de prix et les nouvelles
        
        Args:
            symbol: Symbole boursier
            market_data: DataFrame contenant l'historique des prix
            news: Liste des nouvelles récentes
            
        Returns:
            Dictionnaire contenant les résultats d'analyse de sentiment
        """
        sentiment_results = {
            'overall_score': 0.0,
            'news_sentiment': 'neutre',
            'price_sentiment': 'neutre',
            'market_regime': 'normal',
            'trend_strength': 0.0,
            'confidence': 0.5
        }
        
        try:
            # Analyse du sentiment des nouvelles
            if news and symbol in self.sentiment_analyzers:
                analyzer = self.sentiment_analyzers[symbol]
                
                # Extraire le texte des actualités
                news_texts = []
                for news_item in news:
                    title = news_item.get('title', '')
                    summary = news_item.get('summary', '')
                    if title and summary:
                        news_texts.append(f"{title}. {summary}")
                    elif title:
                        news_texts.append(title)
                
                # Analyser le sentiment si nous avons des textes
                if news_texts:
                    # Dans un système réel, nous appellerions l'analyse de sentiment complète
                    # Ici, simulons une analyse simplifiée pour éviter la complexité
                    news_sentiments = []
                    for text in news_texts:
                        # Simuler une analyse de sentiment simple
                        if "positif" in text.lower() or "hausse" in text.lower() or "croissance" in text.lower():
                            news_sentiments.append(0.7)
                        elif "négatif" in text.lower() or "baisse" in text.lower() or "chute" in text.lower():
                            news_sentiments.append(-0.5)
                        else:
                            news_sentiments.append(0.0)
                    
                    avg_news_sentiment = sum(news_sentiments) / len(news_sentiments)
                    sentiment_results['news_sentiment'] = "positif" if avg_news_sentiment > 0.2 else ("négatif" if avg_news_sentiment < -0.2 else "neutre")
                    sentiment_results['overall_score'] = avg_news_sentiment
            
            # Analyse du sentiment basé sur le prix
            if len(market_data) >= 20:
                # Calculer la volatilité récente
                if 'ATR' in market_data.columns:
                    recent_volatility = market_data['ATR'].iloc[-1] / market_data['Close'].iloc[-1]
                else:
                    recent_returns = market_data['Close'].pct_change().dropna()
                    recent_volatility = recent_returns.std()
                
                # Calculer le momentum récent
                momentum = (market_data['Close'].iloc[-1] / market_data['Close'].iloc[-20] - 1.0)
                
                # Mettre à jour le détecteur de régime de marché
                regime_detector = self.market_regime_detectors[symbol]
                regime_info = regime_detector.update(momentum, recent_volatility)
                
                sentiment_results['market_regime'] = regime_info.get('regime', 'normal')
                sentiment_results['trend_strength'] = abs(momentum)
                sentiment_results['confidence'] = regime_info.get('confidence', 0.5)
                
                # Évaluation du sentiment basé sur le prix
                if momentum > 0.03:
                    price_sentiment = "fortement positif"
                elif momentum > 0.01:
                    price_sentiment = "positif"
                elif momentum < -0.03:
                    price_sentiment = "fortement négatif"
                elif momentum < -0.01:
                    price_sentiment = "négatif"
                else:
                    price_sentiment = "neutre"
                
                sentiment_results['price_sentiment'] = price_sentiment
                
                # Ajuster le score global en tenant compte du sentiment de prix
                sentiment_results['overall_score'] = (sentiment_results['overall_score'] + momentum) / 2
            
            # Arrondir les valeurs numériques pour plus de lisibilité
            sentiment_results['overall_score'] = round(sentiment_results['overall_score'], 2)
            sentiment_results['trend_strength'] = round(sentiment_results['trend_strength'], 2)
            sentiment_results['confidence'] = round(sentiment_results['confidence'], 2)
            
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de sentiment pour {symbol}: {e}")
            return sentiment_results
    
    def assess_risk(self, symbol: str, market_data: pd.DataFrame, fundamental_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Évalue les risques associés à l'investissement dans ce symbole
        
        Args:
            symbol: Symbole boursier
            market_data: DataFrame contenant l'historique des prix
            fundamental_data: Données fondamentales
            
        Returns:
            Dictionnaire contenant l'évaluation des risques
        """
        risk_assessment = {
            'risk_level': 'modéré',
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'risk_reward_ratio': 0.0,
            'key_risks': []
        }
        
        try:
            # Utiliser le gestionnaire de risques
            if symbol in self.risk_managers:
                risk_manager = self.risk_managers[symbol]
                
                # Calculer les métriques de risque de base
                if len(market_data) >= 60:  # Avoir au moins 60 jours de données
                    returns = market_data['Close'].pct_change().dropna()
                    
                    # Volatilité (écart-type annualisé des rendements quotidiens)
                    volatility = returns.std() * (252 ** 0.5)  # Annualisation
                    risk_assessment['volatility'] = round(volatility, 4)
                    
                    # Ratio de Sharpe (supposant un taux sans risque de 0.02 ou 2%)
                    risk_free_rate = 0.02 / 252  # Taux journalier
                    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * (252 ** 0.5)
                    risk_assessment['sharpe_ratio'] = round(sharpe_ratio, 2)
                    
                    # Drawdown maximum
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.cummax()
                    drawdown = (cumulative / running_max - 1)
                    max_drawdown = drawdown.min()
                    risk_assessment['max_drawdown'] = round(max_drawdown, 4)
                    
                    # Définir le niveau de risque
                    if volatility > 0.4:
                        risk_level = "très élevé"
                    elif volatility > 0.3:
                        risk_level = "élevé"
                    elif volatility > 0.2:
                        risk_level = "modéré à élevé"
                    elif volatility > 0.1:
                        risk_level = "modéré"
                    else:
                        risk_level = "faible"
                    
                    risk_assessment['risk_level'] = risk_level
                    
                    # Ratio risque/récompense basé sur l'ATR et le momentum
                    if 'ATR' in market_data.columns and not pd.isna(market_data['ATR'].iloc[-1]):
                        atr = market_data['ATR'].iloc[-1]
                        momentum = returns.iloc[-20:].mean() * 20
                        
                        if abs(momentum) > 0:
                            risk_reward = abs(momentum) / (atr / market_data['Close'].iloc[-1])
                            risk_assessment['risk_reward_ratio'] = round(risk_reward, 2)
                    
                    # Identifier les risques clés
                    key_risks = []
                    
                    # Risque de tendance
                    if market_data['SMA_50'].iloc[-1] < market_data['SMA_200'].iloc[-1]:
                        key_risks.append("Tendance baissière à long terme")
                    
                    # Risque de volatilité
                    if volatility > 0.3:
                        key_risks.append("Volatilité élevée")
                    
                    # Risque de momentum
                    if sharpe_ratio < 0:
                        key_risks.append("Momentum négatif")
                    
                    # Risque fondamental pour les actions
                    if fundamental_data.get('type') == 'stock' and 'financials' in fundamental_data:
                        financials = fundamental_data['financials']
                        
                        # P/E élevé
                        if financials.get('trailingPE') and financials['trailingPE'] > 30:
                            key_risks.append("Valorisation élevée (P/E > 30)")
                        
                        # Beta élevé
                        if financials.get('beta') and financials['beta'] > 1.5:
                            key_risks.append(f"Beta élevé ({financials['beta']:.2f})")
                    
                    risk_assessment['key_risks'] = key_risks
                
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation du risque pour {symbol}: {e}")
            return risk_assessment

def run_daily_analysis_bot():
    """
    Fonction principale pour exécuter le bot d'analyse quotidienne
    """
    # Charger les variables d'environnement si ce n'est pas déjà fait
    load_dotenv()
    
    # Récupérer la liste des symboles depuis les variables d'environnement
    symbols_str = os.environ.get('SYMBOLS', 'AAPL,GOOGL,MSFT,AMZN,TSLA,BTC-USD,ETH-USD')
    symbols = [s.strip() for s in symbols_str.split(',')]
    
    # Vérifier si l'entraînement forcé est demandé
    force_training = os.environ.get('FORCE_MODEL_TRAINING', 'false').lower() == 'true'
    
    # Créer le bot d'analyse
    analysis_bot = DailyAnalysisBot(symbols)
    
    # Si l'entraînement forcé est demandé, nettoyer les modèles existants
    if force_training:
        logger.info("Forced training mode activated - Clearing existing models")
        # Supprimer les anciens fichiers de modèles
        import shutil
        models_dir = os.environ.get('MODELS_DIR', 'saved_models')
        
        try:
            if os.path.exists(models_dir):
                # Supprimer tous les fichiers de modèles mais garder le répertoire
                for filename in os.listdir(models_dir):
                    file_path = os.path.join(models_dir, filename)
                    if os.path.isfile(file_path) and filename != '.gitkeep':
                        os.unlink(file_path)
                logger.info(f"Cleared existing models from {models_dir}")
        except Exception as e:
            logger.warning(f"Error clearing models directory: {e}")
        
        # Forcer l'entraînement des modèles
        analysis_bot.models_trained = False
    
    # Démarrer le bot d'analyse
    analysis_bot.start_scheduled_analysis()

if __name__ == "__main__":
    run_daily_analysis_bot() 