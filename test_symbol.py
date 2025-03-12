import os
import sys
import logging
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ajouter le répertoire courant au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Testez simplement PricePredictionModel
from app.models.price_prediction import PricePredictionModel

def test_symbol(symbol):
    """
    Test the validity of a symbol by trying to download data for it
    """
    try:
        print(f"Testing symbol: {symbol}")
        # Use auto_adjust=False to maintain backward compatibility
        data = yf.download(symbol, period="1y", interval="1d", auto_adjust=False)
        
        if data is None or data.empty:
            print(f"Error: No data available for {symbol}")
            return False
        
        print(f"Successfully downloaded {len(data)} data points for {symbol}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        return True
    
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return False

def main():
    # Charger les variables d'environnement
    load_dotenv()
    
    symbol = "AAPL"
    logger.info(f"Test direct pour le symbole {symbol}")
    
    # Récupérer les données
    data = yf.download(symbol, period="1y", interval="1d")
    logger.info(f"Données téléchargées: {len(data)} points")
    
    # Initialiser le modèle
    model = PricePredictionModel()
    logger.info("Modèle initialisé")
    
    # Entraîner le modèle avec les données et le symbole
    try:
        logger.info("Démarrage de l'entraînement...")
        model.train(data, symbol)
        logger.info("Entraînement réussi!")
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)

if __name__ == "__main__":
    # Get symbol from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_symbol.py SYMBOL")
        sys.exit(1)
    
    symbol = sys.argv[1]
    test_symbol(symbol) 