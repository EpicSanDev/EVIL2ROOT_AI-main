import os
import sys
import pandas as pd
import numpy as np
import logging
import yfinance as yf

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Version simplifiée du modèle qui ne dépend pas des bibliothèques externes
class SimplePricePredictionModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_scalers = {}
        self.logger = logger
        
    def train(self, data, symbol, optimize=True, epochs=100, validation_split=0.2):
        """
        Version simplifiée de la méthode train pour tester l'appel
        """
        import traceback
        import inspect
        
        # Ajout d'un log détaillé
        caller = inspect.getouterframes(inspect.currentframe())[1]
        caller_info = f"{caller.filename}:{caller.lineno} in {caller.function}"
        self.logger.info(f"DÉBUT DE TRAIN pour {symbol} - appelé depuis: {caller_info}")
        
        try:
            self.logger.info(f"Training model for symbol: {symbol}")
            # Simuler l'entraînement sans dépendances complexes
            self.models[symbol] = "trained_model"
            self.logger.info(f"Model training completed for symbol: {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            traceback.print_exc()
            return None

# Notre classe DailyAnalysisBot simplifiée
class SimpleDailyAnalysisBot:
    def __init__(self, symbols):
        self.symbols = symbols
        self.price_prediction_models = {}
        
        # Initialiser les modèles pour chaque symbole
        for symbol in symbols:
            self.price_prediction_models[symbol] = SimplePricePredictionModel()
            
        logger.info(f"Bot initialized with {len(symbols)} symbols")
    
    def fetch_market_data(self, symbol, period="1y", interval="1d"):
        """Récupérer les données de marché"""
        logger.info(f"Fetching market data for {symbol}")
        data = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
        logger.info(f"Downloaded {len(data)} data points for {symbol}")
        return data
    
    def _train_price_model(self, symbol, market_data):
        """Entraîne le modèle de prédiction de prix"""
        logger.info(f"Training price prediction model for {symbol}")
        
        try:
            # Entraîner un nouveau modèle
            logger.info(f"Training new price model for {symbol}")
            self.price_prediction_models[symbol].train(data=market_data, symbol=symbol)
            logger.info(f"Price prediction model ready for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training price model for {symbol}: {e}")
            raise
    
    def train_all_models(self):
        """Entraîne tous les modèles nécessaires pour les analyses"""
        logger.info("Starting model training for all symbols...")
        
        try:
            # Entraîner les modèles pour chaque symbole
            for symbol in self.symbols:
                logger.info(f"Training models for {symbol}...")
                
                # Récupérer les données historiques pour l'entraînement
                try:
                    # Utiliser une période plus courte pour le test
                    market_data = self.fetch_market_data(symbol, period="1mo", interval="1d")
                    
                    # Entraîner le modèle de prédiction de prix
                    self._train_price_model(symbol, market_data)
                    
                    logger.info(f"All models trained successfully for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error training models for {symbol}: {e}")
            
            logger.info("All models trained successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

def main():
    # Créer le bot avec un seul symbole pour simplifier le test
    symbols = ['AAPL']
    
    try:
        # Initialiser le bot
        logger.info("Initializing bot...")
        bot = SimpleDailyAnalysisBot(symbols)
        logger.info("Bot initialized successfully")
        
        # Entraîner tous les modèles
        logger.info("Training all models...")
        bot.train_all_models()
        logger.info("All tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")

if __name__ == "__main__":
    main() 