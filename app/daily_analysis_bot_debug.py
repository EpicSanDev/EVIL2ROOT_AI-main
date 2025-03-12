"""
Version de daily_analysis_bot.py avec plus de journalisation pour le débogage
"""
import os
import sys
import logging
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd

# Ajouter le répertoire courant au path Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration du logging avec niveau DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('daily_analysis_bot_debug')

# Classe simplifiée du modèle de prédiction
class SimplePricePredictionModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_scalers = {}
        self.logger = logger
        
    def train(self, data, symbol=None, optimize=True, epochs=100, validation_split=0.2):
        """
        Version simplifiée de la méthode train avec journalisation détaillée
        """
        import traceback
        import inspect
        
        if symbol is None:
            # SI le symbole est None, c'est ici que l'erreur se produit
            error_msg = "train() missing 1 required positional argument: 'symbol'"
            logger.error(f"ERREUR: {error_msg}")
            raise TypeError(error_msg)
        
        # Déterminer qui nous a appelé
        caller = inspect.getouterframes(inspect.currentframe())[1]
        caller_info = f"{caller.filename}:{caller.lineno} in {caller.function}"
        self.logger.info(f"DÉBUT DE TRAIN pour {symbol} - appelé depuis: {caller_info}")
        
        # Simuler l'entraînement
        self.logger.info(f"Training model for symbol: {symbol}")
        self.models[symbol] = "trained_model"
        self.logger.info(f"Model training completed for symbol: {symbol}")
        return True
        
    def predict(self, data, symbol):
        """Version simplifiée de la méthode predict"""
        if symbol not in self.models:
            self.logger.warning(f"Model not trained for {symbol}")
            return None
        return 100.0  # Valeur prédite factice

# Classe simplifiée du bot d'analyse quotidienne
class DebugDailyAnalysisBot:
    def __init__(self, symbols):
        """Initialisation avec journalisation détaillée"""
        logger.info(f"Initializing bot with symbols: {symbols}")
        
        # Stockage des informations de base
        self.symbols = symbols
        self.models_dir = os.environ.get('MODEL_DIR', 'saved_models')
        
        # Initialisation des modèles
        logger.info("Initializing prediction models...")
        self.price_prediction_models = {}
        
        for symbol in symbols:
            logger.info(f"Creating model for {symbol}")
            self.price_prediction_models[symbol] = SimplePricePredictionModel()
        
        # Flag pour suivre l'état d'entraînement des modèles
        self.models_trained = False
        
        logger.info(f"Bot initialized with {len(symbols)} symbols")
    
    def fetch_market_data(self, symbol, period="5y", interval="1d"):
        """Récupérer les données de marché"""
        logger.info(f"Fetching market data for {symbol}")
        # Use auto_adjust=False to maintain backward compatibility
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
        logger.info(f"Downloaded {len(data)} data points for {symbol}")
        return data
    
    def _train_price_model(self, symbol, market_data):
        """Entraîne le modèle de prédiction de prix avec journalisation détaillée"""
        logger.info(f"ENTREE dans _train_price_model pour {symbol}")
        
        try:
            # Vérifier le type de modèle
            model_type = type(self.price_prediction_models[symbol]).__name__
            logger.info(f"Type de modèle pour {symbol}: {model_type}")
            
            # Entraîner un nouveau modèle
            logger.info(f"Training new price model for {symbol}")
            
            # Point critique: APPEL À train()
            # Utiliser getattr pour obtenir la méthode puis l'appeler pour voir comment elle est définie
            train_method = getattr(self.price_prediction_models[symbol], 'train')
            logger.info(f"Train method: {train_method}")
            logger.info(f"Train method arguments: {train_method.__code__.co_varnames}")
            logger.info(f"Calling train with market_data and symbol={symbol}")
            
            # Appel à train avec tous les arguments nommés pour être sûr
            self.price_prediction_models[symbol].train(data=market_data, symbol=symbol)
            
            logger.info(f"Price prediction model ready for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training price model for {symbol}: {e}", exc_info=True)
            raise
        
        logger.info(f"SORTIE de _train_price_model pour {symbol}")
    
    def train_all_models(self):
        """Entraîne tous les modèles avec journalisation détaillée"""
        logger.info("ENTREE dans train_all_models")
        logger.info("Starting model training for all symbols...")
        
        try:
            # Entraîner les modèles pour chaque symbole
            for symbol in self.symbols:
                logger.info(f"Training models for {symbol}...")
                
                try:
                    # Utiliser une période plus courte pour le test
                    market_data = self.fetch_market_data(symbol, period="1mo", interval="1d")
                    
                    # Entraîner le modèle de prédiction de prix
                    self._train_price_model(symbol, market_data)
                    
                    logger.info(f"All models trained successfully for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error training models for {symbol}: {e}", exc_info=True)
            
            # Marquer les modèles comme entraînés
            self.models_trained = True
            logger.info("All models trained successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
        
        logger.info("SORTIE de train_all_models")

def run_debug_bot():
    """Exécuter le bot de débogage"""
    # Charger les variables d'environnement
    load_dotenv()
    
    # Récupérer la liste des symboles
    symbols_str = os.environ.get('SYMBOLS', 'AAPL,GOOGL,MSFT,AMZN')
    symbols = [s.strip() for s in symbols_str.split(',')][:2]  # Limiter à 2 symboles pour le débogage
    
    # Créer et exécuter le bot
    logger.info("Lancement du bot de débogage...")
    bot = DebugDailyAnalysisBot(symbols)
    bot.train_all_models()
    logger.info("Exécution du bot terminée")

if __name__ == "__main__":
    run_debug_bot() 