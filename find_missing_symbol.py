import os
import re
import sys
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_train_method_signatures():
    """Trouve toutes les signatures de méthode train() dans le projet"""
    train_methods = []
    
    # Parcourir les fichiers Python
    for root, dirs, files in os.walk("app"):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                        # Rechercher les définitions de méthode train
                        matches = re.finditer(r"def\s+train\s*\(([^)]*)\)", content)
                        for match in matches:
                            signature = match.group(1)
                            
                            # Vérifier si la signature contient un paramètre symbol
                            requires_symbol = 'symbol' in signature
                            train_methods.append({
                                'file': file_path,
                                'signature': signature.strip(),
                                'requires_symbol': requires_symbol
                            })
                except Exception as e:
                    logger.error(f"Erreur lors de la lecture de {file_path}: {e}")
    
    return train_methods

def find_train_calls():
    """Trouve tous les appels à train() dans le projet"""
    train_calls = []
    
    # Parcourir les fichiers Python
    for root, dirs, files in os.walk("app"):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                line_number = 0
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            line_number += 1
                            
                            # Rechercher les appels à la méthode train
                            if re.search(r"\.train\s*\(", line):
                                train_calls.append({
                                    'file': file_path,
                                    'line_number': line_number,
                                    'line': line.strip(),
                                    'has_symbol': 'symbol' in line
                                })
                except Exception as e:
                    logger.error(f"Erreur lors de la lecture de {file_path}: {e}")
    
    return train_calls

def find_problematic_calls(train_methods, train_calls):
    """Identifie les appels à train() qui pourraient manquer le paramètre symbol"""
    requires_symbol_files = [m['file'] for m in train_methods if m['requires_symbol']]
    potentially_problematic = []
    
    # Vérifier chaque appel
    for call in train_calls:
        if not call['has_symbol']:
            # Vérifier si l'appel est potentiellement problématique
            is_problematic = False
            for file in requires_symbol_files:
                # Si la ligne contient le nom du fichier ou de la classe qui requiert symbol
                file_base = os.path.basename(file).replace('.py', '')
                if file_base.lower() in call['line'].lower():
                    is_problematic = True
                    break
            
            if is_problematic:
                potentially_problematic.append(call)
    
    return potentially_problematic

def main():
    """Fonction principale"""
    logger.info("Recherche des signatures de méthode train()...")
    train_methods = find_train_method_signatures()
    logger.info(f"Trouvé {len(train_methods)} signatures de méthode train()")
    
    # Afficher les méthodes qui requièrent symbol
    requires_symbol = [m for m in train_methods if m['requires_symbol']]
    logger.info(f"Méthodes qui requièrent le paramètre 'symbol': {len(requires_symbol)}")
    for method in requires_symbol:
        logger.info(f"  - {method['file']}: {method['signature']}")
    
    logger.info("\nRecherche des appels à train()...")
    train_calls = find_train_calls()
    logger.info(f"Trouvé {len(train_calls)} appels à train()")
    
    # Rechercher les appels problématiques
    problematic_calls = find_problematic_calls(train_methods, train_calls)
    logger.info(f"\nAppels potentiellement problématiques: {len(problematic_calls)}")
    for call in problematic_calls:
        logger.info(f"  - {call['file']}:{call['line_number']}: {call['line']}")

if __name__ == "__main__":
    main() 