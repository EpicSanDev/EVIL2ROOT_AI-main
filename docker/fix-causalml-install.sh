#!/bin/bash
set -e

echo "Installation de causalml avec contournement des problèmes de compilation..."

# Installer les dépendances nécessaires
echo "Installation des prérequis pour causalml..."
pip install --no-cache-dir cython>=0.29.33,<3.0.0
pip install --no-cache-dir numpy>=1.21.0 pandas>=1.3.0 scipy>=1.7.0 scikit-learn>=1.0.0 joblib>=1.0.0

# Installer explicitement les autres dépendances nécessaires
pip install --no-cache-dir statsmodels>=0.13.0 lightgbm>=3.3.0 xgboost>=1.5.0

# Version de scikit-learn installée
echo "Vérification de la version de scikit-learn..."
if ! pip show scikit-learn > /dev/null 2>&1; then
    echo "scikit-learn n'est pas installé. Installation en cours..."
    pip install --no-cache-dir scikit-learn==1.3.0
fi

# Récupérer la version complète
SKLEARN_VERSION=$(pip show scikit-learn 2>/dev/null | grep Version | cut -d ' ' -f 2)
if [ -z "$SKLEARN_VERSION" ]; then
    echo "Impossible de détecter la version de scikit-learn. Installation par défaut de causalml 0.12.0..."
    SKLEARN_VERSION="1.3.0"  # Valeur par défaut
fi
echo "Version de scikit-learn installée : $SKLEARN_VERSION"

# Extraire les composants de version
MAJOR=$(echo $SKLEARN_VERSION | cut -d '.' -f 1)
MINOR=$(echo $SKLEARN_VERSION | cut -d '.' -f 2)

# Si la version de scikit-learn est >= 1.3.0, nous utilisons causalml 0.12.0
if [ "$MAJOR" -ge 1 ] && [ "$MINOR" -ge 3 ]; then
    echo "Installation de causalml 0.12.0 (compatible avec scikit-learn >= 1.3.0)..."
    # Installation en deux étapes pour éviter les problèmes
    pip install --no-cache-dir --no-deps causalml==0.12.0
    pip install --no-cache-dir --no-deps econml==0.14.1
    # Résoudre les dépendances restantes
    pip install --no-cache-dir causalml==0.12.0
else
    echo "Installation de causalml 0.13.0..."
    # Installation en deux étapes pour éviter les problèmes
    pip install --no-cache-dir --no-deps causalml==0.13.0
    pip install --no-cache-dir --no-deps econml==0.14.1
    # Résoudre les dépendances restantes
    pip install --no-cache-dir causalml==0.13.0
fi

# Vérifier l'installation
if pip show causalml > /dev/null 2>&1; then
    echo "causalml installé avec succès"
    pip show causalml | grep Version
else
    echo "L'installation de causalml a échoué. Tentative de contournement..."
    
    # Stratégie de secours - installation directe sans vérification de version
    pip install --no-cache-dir causalml==0.12.0 || pip install --no-cache-dir causalml==0.11.0
    
    if pip show causalml > /dev/null 2>&1; then
        echo "Installation de secours réussie!"
        pip show causalml | grep Version
    else
        echo "Toutes les tentatives d'installation ont échoué."
        exit 1
    fi
fi 