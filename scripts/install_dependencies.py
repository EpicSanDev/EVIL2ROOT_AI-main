"""
Script qui installe les dépendances manquantes
"""
import os
import sys
import subprocess
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('install_dependencies')

def check_module_installed(module_name):
    """Vérifie si un module est installé"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def install_module(module_name, version=None):
    """Installe un module Python"""
    package = module_name if version is None else f"{module_name}=={version}"
    
    logger.info(f"Installation de {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"✅ {package} installé avec succès")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur lors de l'installation de {package}: {e}")
        return False

def install_required_dependencies():
    """Installe les dépendances requises"""
    # Liste des dépendances à vérifier et installer si nécessaire
    dependencies = [
        ("skopt", "0.9.0"),  # scikit-optimize
        ("tensorflow", None),
        ("pandas", None),
        ("numpy", None),
        ("matplotlib", None),
        ("yfinance", None),
        ("scikit-learn", None),
        ("optuna", None),
        ("hyperopt", None),
        ("joblib", None),
        ("python-dotenv", None)
    ]
    
    # Vérifier et installer les dépendances manquantes
    for module_name, version in dependencies:
        if not check_module_installed(module_name):
            logger.info(f"Module {module_name} non installé")
            install_module(module_name, version)
        else:
            logger.info(f"✅ Module {module_name} déjà installé")

def main():
    """Fonction principale"""
    logger.info("=== INSTALLATION DES DÉPENDANCES MANQUANTES ===")
    
    # Installer les dépendances requises
    install_required_dependencies()
    
    # Vérifier si skopt est maintenant installé
    if check_module_installed("skopt"):
        logger.info("✅ Le module skopt est maintenant installé")
    else:
        logger.error("❌ Le module skopt n'a pas pu être installé")
    
    logger.info("=== INSTALLATION TERMINÉE ===")

if __name__ == "__main__":
    main() 