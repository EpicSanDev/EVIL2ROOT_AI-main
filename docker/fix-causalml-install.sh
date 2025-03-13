#!/bin/bash
set -e

echo "Résolution des conflits de dépendances et installation de causalml..."

# Résoudre le conflit OpenAI vs langchain-openai
echo "Résolution du conflit OpenAI vs langchain-openai..."
pip uninstall -y openai langchain-openai
pip install --no-cache-dir openai==1.6.1  # Version compatible avec langchain-openai
pip install --no-cache-dir langchain-openai==0.0.2

# Installer les dépendances nécessaires pour causalml avec versions spécifiques
echo "Installation des prérequis pour causalml..."
pip install --no-cache-dir cython==0.29.33  # Version spécifique testée
pip install --no-cache-dir numpy==1.24.3 pandas==2.0.1 scipy==1.10.1 scikit-learn==1.3.0 joblib==1.2.0

# Installer explicitement les autres dépendances nécessaires
pip install --no-cache-dir statsmodels==0.14.0 lightgbm==3.3.5 xgboost==1.7.5

# Vérification de la version de scikit-learn
echo "Vérification de la version de scikit-learn..."
SKLEARN_VERSION=$(pip show scikit-learn 2>/dev/null | grep Version | cut -d ' ' -f 2)
echo "Version de scikit-learn installée : $SKLEARN_VERSION"

# Télécharger manuellement les sources de causalml pour les compiler localement
echo "Installation manuelle de causalml 0.12.0..."
mkdir -p /tmp/causalml_install
cd /tmp/causalml_install

# Cloner le repo causalml à la version 0.12.0
git clone https://github.com/uber/causalml.git .
git checkout v0.12.0

# Configuration Cython avec les flags de compilation corrects
export CFLAGS="-std=c++11"
export LDFLAGS="-std=c++11"

# Modification des fichiers Cython pour éviter l'erreur "Accessing Python attribute not allowed without gil"
sed -i 's/nogil/with gil/g' causalml/inference/tree/causaltree.pyx
sed -i 's/nogil/with gil/g' causalml/inference/tree/uplift.pyx

# Installer en mode développement pour éviter certains problèmes de compilation
pip install -e .

# Vérifier l'installation
if pip show causalml > /dev/null 2>&1; then
    echo "causalml installé avec succès"
    pip show causalml | grep Version
else
    echo "L'installation de causalml a échoué. Tentative de contournement..."
    
    # Stratégie de secours - installation directe d'une version antérieure
    cd /tmp
    pip install --no-cache-dir causalml==0.11.0
    
    if pip show causalml > /dev/null 2>&1; then
        echo "Installation de secours réussie!"
        pip show causalml | grep Version
    else
        echo "Toutes les tentatives d'installation ont échoué."
        echo "Continuez sans causalml - les fonctionnalités associées ne seront pas disponibles."
        touch /opt/venv/lib/python3.9/site-packages/causalml_not_available
    fi
fi

# Réinstaller econml qui peut avoir été affecté
pip install --no-cache-dir econml==0.14.1

echo "Processus de résolution des dépendances terminé." 