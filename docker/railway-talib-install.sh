#!/bin/bash
# Script pour installer TA-Lib sur Railway
# Ce script est destiné à être exécuté dans le cadre d'un build sur Railway

set -e

echo "Installation de TA-Lib pour Railway..."

# Installer les dépendances nécessaires
apt-get update
apt-get install -y --no-install-recommends wget build-essential gcc g++ make pkg-config

# Télécharger et installer TA-Lib
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Téléchargement de TA-Lib 0.4.0..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/

echo "Configuration et compilation de TA-Lib..."
./configure --prefix=/usr
make -j$(nproc)
make install

# Créer des liens symboliques et structure de répertoire
echo "Configuration des en-têtes et des bibliothèques..."
mkdir -p /usr/include/ta-lib
cp -r /usr/include/ta_*.h /usr/include/ta-lib/

# Installer une version spécifique de numpy compatible
echo "Installation d'une version de NumPy compatible avec TA-Lib..."
pip install numpy==1.24.3

# Installer directement depuis le repo GitHub
echo "Installation de TA-Lib Python wrapper depuis GitHub..."
pip install git+https://github.com/TA-Lib/ta-lib-python.git@0.4.28

# Tester l'installation
echo "Test de l'installation de TA-Lib..."
python -c "import talib; print('TA-Lib importé avec succès!')"

echo "Installation de TA-Lib terminée!"
