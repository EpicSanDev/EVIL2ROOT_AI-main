
import os
import sys
import logging

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ajouter le répertoire courant au path Python
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)
    logger.info(f"Added {current_dir} to sys.path")

def minimal_test():
    try:
        # Importer PricePredictionModel
        logger.info("Importing PricePredictionModel...")
        
        # Option 1: via package app
        try:
            from app.models.price_prediction import PricePredictionModel as PricePredictionModel1
            logger.info("✅ Import via package app successful")
        except ImportError as e:
            logger.error(f"❌ Import via package app failed: {e}")
        
        # Option 2: direct import
        try:
            sys.path.append(os.path.join(current_dir, 'app', 'models'))
            from price_prediction import PricePredictionModel as PricePredictionModel2
            logger.info("✅ Direct import successful")
        except ImportError as e:
            logger.error(f"❌ Direct import failed: {e}")
            
        # Option 3: import specific file
        try:
            import importlib.util
            file_path = os.path.join(current_dir, 'app', 'models', 'price_prediction.py')
            spec = importlib.util.spec_from_file_location("price_prediction", file_path)
            price_prediction = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(price_prediction)
            PricePredictionModel3 = price_prediction.PricePredictionModel
            logger.info("✅ File-specific import successful")
            
            # Test if train method exists and has symbol parameter
            train_params = inspect.signature(PricePredictionModel3.train).parameters
            if 'symbol' in train_params:
                logger.info("✅ train() method has symbol parameter")
            else:
                logger.error("❌ train() method does NOT have symbol parameter")
                
        except Exception as e:
            logger.error(f"❌ File-specific import failed: {e}")
            
    except Exception as e:
        logger.error(f"Error in minimal test: {e}")

if __name__ == "__main__":
    minimal_test()
