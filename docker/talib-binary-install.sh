#!/bin/bash
# Script pour installer TA-Lib en utilisant des binaires précompilés
# Utile comme solution de dernier recours si la compilation échoue

set -e

echo "Installation de TA-Lib en utilisant des binaires précompilés..."

# Installer les dépendances nécessaires
apt-get update
apt-get install -y --no-install-recommends wget

# Télécharger et installer TA-Lib binaires précompilés
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Utiliser une version spécifique de NumPy connue pour être compatible
echo "Installation de NumPy compatible..."
pip install numpy==1.24.3

# Télécharger le binaire précompilé spécifique à notre version Python
echo "Installation de TA-Lib depuis un binaire précompilé..."

# Pour Python 3.10 sur x86_64
pip install --no-cache-dir --index-url https://pypi.anaconda.org/ranaroussi/simple ta-lib==0.4.28

# Vérifier l'installation
python -c "import talib; print('TA-Lib importé avec succès via binaire précompilé!')"

echo "Installation de TA-Lib via binaire terminée!"
