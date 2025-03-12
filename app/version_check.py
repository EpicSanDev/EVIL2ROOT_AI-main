"""
Script qui vérifie les versions de code en cours d'exécution
"""
import os
import sys
import inspect
import logging
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('version_check')

def check_model_trainer():
    """Vérifie la version du code model_trainer.py"""
    try:
        # Import dynamique du module
        from app.model_trainer import ModelTrainer
        
        # Récupérer le chemin du fichier
        file_path = inspect.getfile(ModelTrainer)
        logger.info(f"ModelTrainer loaded from: {file_path}")
        
        # Analyse du fichier pour voir le code réel
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Rechercher les méthodes _optimize_tpsl_model et _optimize_indicator_model
            if '_optimize_tpsl_model' in content:
                # Extraire la méthode _optimize_tpsl_model
                start = content.find('async def _optimize_tpsl_model')
                end = content.find('async def', start + 1)
                if end == -1:
                    end = content.find('def', start + 1)
                method_code = content[start:end].strip()
                
                # Vérifier si la méthode contient train(train_data, symbol)
                if 'train(train_data, symbol)' in method_code:
                    logger.info("✅ _optimize_tpsl_model contient correctement 'train(train_data, symbol)'")
                else:
                    logger.error("❌ _optimize_tpsl_model ne contient PAS 'train(train_data, symbol)'")
                    logger.error(f"Contenu de la méthode: {method_code}")
            else:
                logger.error("❌ Méthode _optimize_tpsl_model non trouvée!")
            
            # Faire de même pour _optimize_indicator_model
            if '_optimize_indicator_model' in content:
                # Extraire la méthode _optimize_indicator_model
                start = content.find('async def _optimize_indicator_model')
                end = content.find('async def', start + 1)
                if end == -1:
                    end = content.find('def', start + 1)
                method_code = content[start:end].strip()
                
                # Vérifier si la méthode contient train(train_data, symbol)
                if 'train(train_data, symbol)' in method_code:
                    logger.info("✅ _optimize_indicator_model contient correctement 'train(train_data, symbol)'")
                else:
                    logger.error("❌ _optimize_indicator_model ne contient PAS 'train(train_data, symbol)'")
                    logger.error(f"Contenu de la méthode: {method_code}")
            else:
                logger.error("❌ Méthode _optimize_indicator_model non trouvée!")
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de model_trainer.py: {e}", exc_info=True)

def check_class_versions():
    """Vérifie les versions des classes de modèles"""
    try:
        # Import des classes
        from app.models.price_prediction import PricePredictionModel
        from app.models.indicator_management import IndicatorManagementModel
        from app.models.tp_sl_management import TpSlManagementModel
        from app.models.risk_management import RiskManagementModel
        
        # Vérifier les signatures des méthodes train
        for cls_name, cls in [
            ('PricePredictionModel', PricePredictionModel),
            ('IndicatorManagementModel', IndicatorManagementModel),
            ('TpSlManagementModel', TpSlManagementModel),
            ('RiskManagementModel', RiskManagementModel)
        ]:
            try:
                # Récupérer la signature de la méthode train
                train_method = cls.train
                sig = inspect.signature(train_method)
                params = list(sig.parameters.keys())
                
                logger.info(f"{cls_name}.train() parameters: {params}")
                
                # Vérifier si symbol est un paramètre requis
                if 'symbol' in params:
                    # Vérifier si c'est un paramètre obligatoire (pas de valeur par défaut)
                    param = sig.parameters['symbol']
                    if param.default == inspect.Parameter.empty:
                        logger.info(f"✅ {cls_name}.train() exige le paramètre 'symbol'")
                    else:
                        logger.warning(f"⚠️ {cls_name}.train() a 'symbol' avec une valeur par défaut: {param.default}")
                else:
                    logger.error(f"❌ {cls_name}.train() n'a PAS de paramètre 'symbol'!")
                
                # Afficher le chemin du fichier
                file_path = inspect.getfile(cls)
                logger.info(f"{cls_name} loaded from: {file_path}")
                
            except Exception as e:
                logger.error(f"Erreur lors de la vérification de {cls_name}: {e}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification des classes: {e}", exc_info=True)

def check_modified_files():
    """Vérifie si les fichiers ont été modifiés récemment"""
    try:
        files_to_check = [
            "app/model_trainer.py",
            "app/models/price_prediction.py",
            "app/models/indicator_management.py",
            "app/models/tp_sl_management.py",
            "app/models/risk_management.py"
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                mod_time = os.path.getmtime(file_path)
                from datetime import datetime
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"Fichier {file_path} dernière modification: {mod_time_str}")
            else:
                logger.error(f"Fichier {file_path} introuvable!")
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification des dates de modification: {e}")

def main():
    """Fonction principale"""
    logger.info("=== VÉRIFICATION DES VERSIONS DE CODE ===")
    
    # Vérifier model_trainer.py
    logger.info("\n--- Vérification de model_trainer.py ---")
    check_model_trainer()
    
    # Vérifier les classes de modèles
    logger.info("\n--- Vérification des classes de modèles ---")
    check_class_versions()
    
    # Vérifier les dates de modification
    logger.info("\n--- Vérification des dates de modification ---")
    check_modified_files()
    
    logger.info("=== VÉRIFICATION TERMINÉE ===")

if __name__ == "__main__":
    main() 