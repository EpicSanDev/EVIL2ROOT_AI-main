"""
Script autonome qui vérifie les signatures des méthodes train sans importer les modules de l'application
"""
import os
import sys
import inspect
import logging
import importlib.util
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('check_signatures')

def load_module_from_file(file_path, module_name):
    """Charge un module Python à partir d'un fichier sans l'importer dans l'espace de noms global"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            logger.error(f"Impossible de charger le module depuis {file_path}")
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Erreur lors du chargement du module {module_name} depuis {file_path}: {e}")
        return None

def extract_class_from_file(file_path, class_name):
    """Extrait une classe d'un fichier Python sans importer le module"""
    try:
        # Lire le contenu du fichier
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Trouver la définition de la classe
        class_start = content.find(f"class {class_name}")
        if class_start == -1:
            logger.error(f"Classe {class_name} non trouvée dans {file_path}")
            return None
        
        # Trouver la fin de la classe (indentation)
        lines = content[class_start:].split('\n')
        class_def = lines[0]
        indent_level = len(class_def) - len(class_def.lstrip())
        
        class_code = [class_def]
        for line in lines[1:]:
            if line.strip() and not line.startswith(' ' * (indent_level + 4)) and not line.startswith('\t'):
                break
            class_code.append(line)
        
        class_content = '\n'.join(class_code)
        
        # Vérifier si la méthode train existe
        if "def train" in class_content:
            # Trouver la définition de la méthode train
            train_start = class_content.find("def train")
            train_lines = class_content[train_start:].split('\n')
            train_def = train_lines[0]
            
            # Extraire les paramètres
            params_start = train_def.find('(')
            params_end = train_def.find(')')
            if params_start != -1 and params_end != -1:
                params_str = train_def[params_start+1:params_end].strip()
                params = [p.strip() for p in params_str.split(',')]
                
                # Vérifier si symbol est un paramètre
                has_symbol = False
                symbol_required = False
                
                for param in params:
                    if param.startswith('symbol'):
                        has_symbol = True
                        if '=' not in param:
                            symbol_required = True
                        break
                
                if has_symbol:
                    if symbol_required:
                        logger.info(f"✅ {class_name}.train() exige le paramètre 'symbol'")
                    else:
                        logger.warning(f"⚠️ {class_name}.train() a 'symbol' avec une valeur par défaut")
                else:
                    logger.error(f"❌ {class_name}.train() n'a PAS de paramètre 'symbol'!")
                
                return {
                    'has_symbol': has_symbol,
                    'symbol_required': symbol_required,
                    'params': params
                }
            else:
                logger.error(f"Impossible d'extraire les paramètres de train() dans {class_name}")
        else:
            logger.error(f"Méthode train() non trouvée dans {class_name}")
        
        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction de la classe {class_name} depuis {file_path}: {e}")
        return None

def check_model_trainer_file():
    """Vérifie le fichier model_trainer.py"""
    file_path = os.path.join('app', 'model_trainer.py')
    
    try:
        # Lire le contenu du fichier
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
        logger.error(f"Erreur lors de la vérification de model_trainer.py: {e}")

def check_class_signatures():
    """Vérifie les signatures des méthodes train dans les classes de modèles"""
    models_dir = os.path.join('app', 'models')
    
    # Liste des classes à vérifier
    classes_to_check = [
        ('price_prediction.py', 'PricePredictionModel'),
        ('indicator_management.py', 'IndicatorManagementModel'),
        ('tp_sl_management.py', 'TpSlManagementModel'),
        ('risk_management.py', 'RiskManagementModel')
    ]
    
    for file_name, class_name in classes_to_check:
        file_path = os.path.join(models_dir, file_name)
        logger.info(f"Vérification de {class_name} dans {file_path}")
        
        if os.path.exists(file_path):
            result = extract_class_from_file(file_path, class_name)
            if result:
                logger.info(f"Paramètres de {class_name}.train(): {result['params']}")
        else:
            logger.error(f"Fichier {file_path} non trouvé")

def main():
    """Fonction principale"""
    logger.info("=== VÉRIFICATION DES SIGNATURES DES MÉTHODES TRAIN ===")
    
    # Vérifier le fichier model_trainer.py
    logger.info("\n--- Vérification de model_trainer.py ---")
    check_model_trainer_file()
    
    # Vérifier les signatures des classes
    logger.info("\n--- Vérification des signatures des classes ---")
    check_class_signatures()
    
    logger.info("=== VÉRIFICATION TERMINÉE ===")

if __name__ == "__main__":
    main() 