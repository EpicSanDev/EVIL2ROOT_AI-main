"""
Script simplifié qui vérifie directement les fichiers sans importation
"""
import os
import re
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('check_code')

def check_file_content(file_path, search_text):
    """Vérifie si un fichier contient un texte spécifique"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Vérifier la présence du texte
                if search_text in content:
                    logger.info(f"✅ Le fichier {file_path} contient bien '{search_text}'")
                    return True
                else:
                    # Essayer de trouver des variantes
                    if search_text.replace(' ', '') in content.replace(' ', ''):
                        logger.warning(f"⚠️ Le fichier {file_path} contient une variante de '{search_text}' (espaces différents)")
                        return True
                    
                    # Réessayer avec une expression régulière plus flexible
                    pattern = search_text.replace('(', r'\(').replace(')', r'\)').replace(',', r',\s*')
                    if re.search(pattern, content):
                        logger.warning(f"⚠️ Le fichier {file_path} contient une variante de '{search_text}' (expression régulière)")
                        return True
                    
                    logger.error(f"❌ Le fichier {file_path} ne contient PAS '{search_text}'")
                    
                    # Chercher des variantes proches pour aider au diagnostic
                    if 'train(' in content:
                        # Trouver tous les appels à train()
                        train_calls = re.findall(r'\.train\([^)]*\)', content)
                        logger.error(f"Appels à train() trouvés: {train_calls}")
                    
                    return False
        else:
            logger.error(f"❌ Le fichier {file_path} n'existe pas")
            return False
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du fichier {file_path}: {e}")
        return False

def check_file_modified_time(file_path):
    """Vérifie quand un fichier a été modifié pour la dernière fois"""
    try:
        if os.path.exists(file_path):
            mod_time = os.path.getmtime(file_path)
            mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Fichier {file_path} dernière modification: {mod_time_str}")
        else:
            logger.error(f"Fichier {file_path} introuvable!")
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de la date de modification: {e}")

def main():
    """Fonction principale"""
    logger.info("=== VÉRIFICATION DES FICHIERS DE CODE ===")
    
    # Vérifier le contenu des fichiers critiques
    logger.info("\n--- Vérification du contenu des fichiers ---")
    
    # 1. Vérifier model_trainer.py
    check_file_content(
        "app/model_trainer.py", 
        "model.train(train_data, symbol)"
    )
    
    # 2. Vérifier les méthodes train dans les modèles
    check_file_content(
        "app/models/price_prediction.py", 
        "def train(self, data, symbol,"
    )
    
    check_file_content(
        "app/models/indicator_management.py", 
        "def train(self, data, symbol"
    )
    
    check_file_content(
        "app/models/tp_sl_management.py", 
        "def train(self, data: pd.DataFrame, symbol: str)"
    )
    
    # 3. Vérifier daily_analysis_bot.py
    check_file_content(
        "app/daily_analysis_bot.py", 
        "self.price_prediction_models[symbol].train(market_data, symbol)"
    )
    
    # Vérifier les dates de modification
    logger.info("\n--- Vérification des dates de modification ---")
    files_to_check = [
        "app/model_trainer.py",
        "app/models/price_prediction.py",
        "app/models/indicator_management.py",
        "app/models/tp_sl_management.py",
        "app/models/risk_management.py",
        "app/daily_analysis_bot.py"
    ]
    
    for file_path in files_to_check:
        check_file_modified_time(file_path)
    
    logger.info("=== VÉRIFICATION TERMINÉE ===")

if __name__ == "__main__":
    main() 