"""
Script qui teste notre solution avec le modèle simplifié
"""
import os
import sys
import logging
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_with_mock')

# Import du modèle PricePredictionModel simplifié
from app.models.price_prediction_mock import PricePredictionModel

# Modèles simplifiés pour compléter les tests
class IndicatorManagementModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def train(self, data, symbol):
        logger.info(f"Training indicator model for {symbol}")
        self.models[symbol] = "mock_model"
        return True
    
    def save(self, model_path):
        logger.info(f"Saving indicator model to {model_path}")
        return True
    
    def load(self, model_path):
        logger.info(f"Loading indicator model from {model_path}")
        return True

class TpSlManagementModel:
    def __init__(self):
        self.models = {}
        
    def train(self, data, symbol):
        logger.info(f"Training TP/SL model for {symbol}")
        self.models[symbol] = "mock_model"
        return True
    
    def save(self, model_path):
        logger.info(f"Saving TP/SL model to {model_path}")
        return True
    
    def load(self, model_path):
        logger.info(f"Loading TP/SL model from {model_path}")
        return True

# Simulons le code problématique dans model_trainer.py
def simulate_model_trainer_bug():
    logger.info("Simulation du bug dans model_trainer.py")
    
    # Cas 1: Sans le paramètre 'symbol'
    try:
        train_data = pd.DataFrame({'Close': [100, 101, 102]})
        tpsl_model = TpSlManagementModel()
        
        # Version buggée: manque le paramètre 'symbol'
        logger.info("Tentative d'appel à train sans le paramètre 'symbol'")
        tpsl_model.train(train_data)
        
        logger.info("✅ Appel réussi - mais cela ne devrait pas être le cas!")
    except TypeError as e:
        logger.info(f"❌ Erreur attendue: {e}")
    
    # Cas 2: Avec le paramètre 'symbol'
    try:
        train_data = pd.DataFrame({'Close': [100, 101, 102]})
        tpsl_model = TpSlManagementModel()
        
        # Version corrigée: avec le paramètre 'symbol'
        logger.info("Tentative d'appel à train avec le paramètre 'symbol'")
        tpsl_model.train(train_data, "TEST")
        
        logger.info("✅ Appel réussi - c'est normal!")
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")

# Classe DailyAnalysisBot simplifiée
class SimpleDailyAnalysisBot:
    def __init__(self, symbols):
        """Initialisation avec journalisation détaillée"""
        logger.info(f"Initializing bot with symbols: {symbols}")
        
        # Stockage des informations de base
        self.symbols = symbols
        self.models_dir = 'saved_models'
        
        # Initialisation des modèles
        logger.info("Initializing prediction models...")
        self.price_prediction_models = {}
        self.indicator_managers = {}
        self.tpsl_models = {}
        
        for symbol in symbols:
            logger.info(f"Creating model for {symbol}")
            self.price_prediction_models[symbol] = PricePredictionModel()
            self.indicator_managers[symbol] = IndicatorManagementModel()
            self.tpsl_models[symbol] = TpSlManagementModel()
        
        # Flag pour suivre l'état d'entraînement des modèles
        self.models_trained = False
        
        logger.info(f"Bot initialized with {len(symbols)} symbols")
    
    def fetch_market_data(self, symbol, period="1y", interval="1d"):
        """Récupérer les données de marché"""
        logger.info(f"Fetching market data for {symbol}")
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
            self.price_prediction_models[symbol].train(market_data, symbol)
            
            logger.info(f"Price prediction model ready for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training price model for {symbol}: {e}", exc_info=True)
            raise
        
        logger.info(f"SORTIE de _train_price_model pour {symbol}")
    
    def _train_indicator_model(self, symbol, market_data):
        """Entraîne le modèle d'indicateurs"""
        logger.info(f"Training indicator model for {symbol}")
        
        try:
            # Entraîner un nouveau modèle
            logger.info(f"Training new indicator model for {symbol}")
            self.indicator_managers[symbol].train(market_data, symbol)
            
            logger.info(f"Indicator model ready for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training indicator model for {symbol}: {e}", exc_info=True)
            raise
    
    def _train_tpsl_model(self, symbol, market_data):
        """Entraîne le modèle de take profit / stop loss"""
        logger.info(f"Training TP/SL model for {symbol}")
        
        try:
            # Entraîner un nouveau modèle
            logger.info(f"Training new TP/SL model for {symbol}")
            self.tpsl_models[symbol].train(market_data, symbol)
            
            logger.info(f"TP/SL model ready for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training TP/SL model for {symbol}: {e}", exc_info=True)
            raise
    
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
                    
                    # Entraîner les modèles
                    self._train_price_model(symbol, market_data)
                    self._train_indicator_model(symbol, market_data)
                    self._train_tpsl_model(symbol, market_data)
                    
                    logger.info(f"All models trained successfully for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error training models for {symbol}: {e}", exc_info=True)
            
            # Marquer les modèles comme entraînés
            self.models_trained = True
            logger.info("All models trained successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
        
        logger.info("SORTIE de train_all_models")

def main():
    """Fonction principale"""
    logger.info("=== SIMULATION AVEC MODÈLES SIMPLIFIÉS ===")
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Tester la classe TpSlManagementModel
    logger.info("\n--- Test de TpSlManagementModel ---")
    simulate_model_trainer_bug()
    
    # Tester le bot complet avec les modèles simplifiés
    logger.info("\n--- Test du bot complet ---")
    symbols = ['AAPL', 'GOOGL']
    bot = SimpleDailyAnalysisBot(symbols)
    bot.train_all_models()
    
    logger.info("=== SIMULATION TERMINÉE ===")

if __name__ == "__main__":
    main() 