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

# Créer un dataset de test simple
def create_test_data():
    # Créer un dataset simple pour le test
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    # Créer des données de prix simulées
    close = 100 + np.cumsum(np.random.normal(0, 1, 500))
    high = close + np.random.uniform(0.5, 3, 500)
    low = close - np.random.uniform(0.5, 3, 500)
    open_price = close + np.random.normal(0, 1, 500)
    volume = np.random.randint(1000, 10000, 500)
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    return df

def test_models_directly():
    try:
        # Importer les classes de modèle directement
        from app.models.price_prediction import PricePredictionModel
        from app.models.risk_management import RiskManagementModel
        from app.models.indicator_management import IndicatorManagementModel
        from app.models.tp_sl_management import TpSlManagementModel
        
        logger.info("Classes importées avec succès")
        
        # Créer des données de test
        data = create_test_data()
        logger.info(f"Données de test créées: {len(data)} points")
        
        # Tester PricePredictionModel
        try:
            logger.info("Test de PricePredictionModel...")
            price_model = PricePredictionModel()
            price_model.train(data, "TEST")
            logger.info("PricePredictionModel training successful")
        except Exception as e:
            logger.error(f"Erreur dans PricePredictionModel: {e}")
        
        # Tester IndicatorManagementModel
        try:
            logger.info("Test de IndicatorManagementModel...")
            indicator_model = IndicatorManagementModel()
            indicator_model.train(data, "TEST")
            logger.info("IndicatorManagementModel training successful")
        except Exception as e:
            logger.error(f"Erreur dans IndicatorManagementModel: {e}")
        
        # Tester TpSlManagementModel
        try:
            logger.info("Test de TpSlManagementModel...")
            tpsl_model = TpSlManagementModel()
            tpsl_model.train(data, "TEST")
            logger.info("TpSlManagementModel training successful")
        except Exception as e:
            logger.error(f"Erreur dans TpSlManagementModel: {e}")
            
        # Tester RiskManagementModel
        try:
            logger.info("Test de RiskManagementModel...")
            risk_model = RiskManagementModel()
            risk_model.train(data, "TEST")
            logger.info("RiskManagementModel training successful")
        except Exception as e:
            logger.error(f"Erreur dans RiskManagementModel: {e}")
            
        logger.info("Tous les tests directs terminés")
        
    except Exception as e:
        logger.error(f"Erreur générale: {e}")

if __name__ == "__main__":
    test_models_directly() 