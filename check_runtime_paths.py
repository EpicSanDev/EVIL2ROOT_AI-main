"""
Script qui analyse les chemins d'importation Python et le code exécuté
"""
import os
import sys
import inspect
import importlib
import logging
import traceback

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('runtime_check')

def log_system_info():
    """Affiche des informations sur l'environnement d'exécution"""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"sys.path: {sys.path}")
    logger.info(f"Current working directory: {os.getcwd()}")

def analyze_modules(module_name):
    """Analyse un module et ses dépendances"""
    try:
        # Tenter d'importer le module directement
        module = importlib.import_module(module_name)
        logger.info(f"✅ Module '{module_name}' importé avec succès")
        logger.info(f"Module path: {inspect.getfile(module)}")
        
        return module
    except ImportError as e:
        logger.error(f"❌ Impossible d'importer le module '{module_name}': {e}")
        
        # Essayer d'importer des modules parents
        parts = module_name.split('.')
        for i in range(1, len(parts)):
            try:
                parent_module = '.'.join(parts[:i])
                importlib.import_module(parent_module)
                logger.info(f"✅ Module parent '{parent_module}' importé avec succès")
            except ImportError as e:
                logger.error(f"❌ Impossible d'importer le module parent '{parent_module}': {e}")
        
        return None
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'importation du module '{module_name}': {e}")
        return None

def simulate_bot_startup():
    """Simule le démarrage du bot d'analyse quotidienne"""
    try:
        # Essayer d'importer le module daily_analysis_bot
        # On ajoute le répertoire courant au path pour faciliter l'importation
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        try:
            # Débit via le fichier __init__.py
            logger.info("Tentative d'importation via __init__.py...")
            import app
            logger.info(f"✅ Package 'app' importé avec succès")
            logger.info(f"Package path: {inspect.getfile(app)}")
        except ImportError as e:
            logger.error(f"❌ Impossible d'importer le package 'app': {e}")
            
            # Essayer une importation directe du fichier
            logger.info("Tentative d'importation directe...")
            sys.path.append(os.path.join(current_dir, 'app'))
            try:
                import daily_analysis_bot
                logger.info(f"✅ Module 'daily_analysis_bot' importé avec succès")
                logger.info(f"Module path: {inspect.getfile(daily_analysis_bot)}")
            except ImportError as e:
                logger.error(f"❌ Impossible d'importer 'daily_analysis_bot': {e}")
                logger.error(f"Trace complète: {traceback.format_exc()}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la simulation du démarrage du bot: {e}")
        logger.error(f"Trace complète: {traceback.format_exc()}")

def create_minimal_test():
    """Crée un script de test minimal pour identifier le problème"""
    test_script = """
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
"""
    
    # Écrire le script dans un fichier
    with open('minimal_test.py', 'w') as f:
        f.write(test_script)
    
    logger.info("Script de test minimal créé: minimal_test.py")

def main():
    """Fonction principale"""
    logger.info("=== ANALYSE DE L'ENVIRONNEMENT D'EXÉCUTION ===")
    
    # Afficher des informations sur l'environnement
    logger.info("\n--- Informations système ---")
    log_system_info()
    
    # Analyser les modules
    logger.info("\n--- Analyse des modules ---")
    modules_to_analyze = [
        'app',
        'app.models',
        'app.models.price_prediction',
        'app.model_trainer',
        'app.daily_analysis_bot'
    ]
    
    for module_name in modules_to_analyze:
        analyze_modules(module_name)
    
    # Simuler le démarrage du bot
    logger.info("\n--- Simulation du démarrage du bot ---")
    simulate_bot_startup()
    
    # Créer un script de test minimal
    logger.info("\n--- Création d'un script de test minimal ---")
    create_minimal_test()
    
    logger.info("=== ANALYSE TERMINÉE ===")

if __name__ == "__main__":
    main() 