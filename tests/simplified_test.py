import os
import sys
import pandas as pd
import numpy as np
import logging

# Ajouter le répertoire parent au chemin Python pour permettre les importations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Classes de modèle simplifiées pour le test
class SimplePriceModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_scalers = {}
        
    def train(self, data, symbol):
        logger.info(f"Training simplified price model for {symbol}")
        # Simuler l'entraînement du modèle
        self.models[symbol] = "trained_model"
        return True

class SimpleIndicatorModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def train(self, data, symbol):
        logger.info(f"Training simplified indicator model for {symbol}")
        # Simuler l'entraînement du modèle
        self.models[symbol] = "trained_model"
        return True

# Créer un dataset de test simple
def create_test_data():
    # Créer un dataset simple pour le test
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Créer des données de prix simulées
    close = 100 + np.cumsum(np.random.normal(0, 1, 100))
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'Close': close
    }, index=dates)
    
    return df

def test_models_directly():
    try:
        logger.info("Starting simplified test")
        
        # Créer des données de test
        data = create_test_data()
        logger.info(f"Test data created: {len(data)} points")
        
        # Tester SimplePriceModel
        try:
            logger.info("Testing SimplePriceModel...")
            price_model = SimplePriceModel()
            price_model.train(data, "TEST")
            logger.info("Price model training successful")
        except Exception as e:
            logger.error(f"Error in price model: {e}")
        
        # Tester SimpleIndicatorModel
        try:
            logger.info("Testing SimpleIndicatorModel...")
            indicator_model = SimpleIndicatorModel()
            indicator_model.train(data, "TEST")
            logger.info("Indicator model training successful")
        except Exception as e:
            logger.error(f"Error in indicator model: {e}")
        
        logger.info("All simplified tests completed")
        
    except Exception as e:
        logger.error(f"General error: {e}")

if __name__ == "__main__":
    test_models_directly() 