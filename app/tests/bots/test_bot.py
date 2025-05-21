import os
import logging
import sys
from dotenv import load_dotenv

# Ajouter le répertoire parent au chemin Python pour permettre les importations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.daily_analysis_bot import DailyAnalysisBot

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_bot_initialization():
    # Charger les variables d'environnement
    load_dotenv()
    
    # Créer un bot avec un seul symbole pour simplifier le test
    symbols = ['AAPL']
    
    try:
        # Initialiser le bot
        logger.info("Initializing bot...")
        bot = DailyAnalysisBot(symbols)
        logger.info("Bot initialized successfully")
        
        # Récupérer des données pour le test
        logger.info("Fetching market data...")
        market_data = bot.fetch_market_data('AAPL', period="1y", interval="1d")
        logger.info(f"Fetched {len(market_data)} data points")
        
        # Tester l'entraînement d'un modèle spécifique
        logger.info("Testing price model training...")
        bot._train_price_model('AAPL', market_data)
        logger.info("Price model training completed successfully")
        
        # Succès!
        logger.info("All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_bot_initialization() 