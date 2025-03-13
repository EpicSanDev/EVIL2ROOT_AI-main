#!/bin/bash
set -e

echo "Installation de causalml avec contournement des problèmes de compilation..."

# Installer les dépendances nécessaires
pip install cython>=0.29.33,<3.0.0

# Version de scikit-learn compatible avec causalml
SKLEARN_VERSION=$(pip show scikit-learn | grep Version | cut -d ' ' -f 2)
echo "Version de scikit-learn installée : $SKLEARN_VERSION"

# Si la version de scikit-learn est >= 1.3.0, nous utilisons causalml 0.12.0 au lieu de 0.13.0
if [[ "$(pip show scikit-learn | grep Version | cut -d ' ' -f 2 | cut -d '.' -f 1)" -ge 1 ]] && 
   [[ "$(pip show scikit-learn | grep Version | cut -d ' ' -f 2 | cut -d '.' -f 2)" -ge 3 ]]; then
    echo "Installation de causalml 0.12.0 (compatible avec scikit-learn >= 1.3.0)..."
    pip install causalml==0.12.0
else
    echo "Installation de causalml 0.13.0..."
    pip install causalml==0.13.0
fi

# Vérifier l'installation
if pip show causalml > /dev/null; then
    echo "causalml installé avec succès"
    pip show causalml | grep Version
else
    echo "L'installation de causalml a échoué"
    exit 1
fi 