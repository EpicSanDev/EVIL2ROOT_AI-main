#!/bin/bash
# Script pour installer TA-Lib dans les conteneurs Kubernetes

# Afficher le début de l'installation
echo "Début de l'installation de TA-Lib..."

# Créer un répertoire temporaire de travail
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Installer les dépendances nécessaires
apt-get update
apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    gcc \
    g++ \
    make \
    pkg-config

# Télécharger TA-Lib
echo "Téléchargement de TA-Lib 0.4.0..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

# Extraire l'archive
tar -xzf ta-lib-0.4.0-src.tar.gz

# Compiler et installer
cd ta-lib/
echo "Configuration et compilation de TA-Lib..."
./configure --prefix=/usr --build=$(uname -m)-unknown-linux-gnu
make -j$(nproc)
make install
cd ../

# S'assurer que les liens symboliques sont correctement créés
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta-lib.so
ldconfig

# Vérifier que la bibliothèque est bien installée
ls -la /usr/lib/libta_lib*
ls -la /usr/lib/libta-lib*
ldconfig -p | grep ta_lib

# Installer TA-Lib pour Python
echo "Installation du package Python TA-Lib..."
pip install --upgrade numpy
export TA_LIBRARY_PATH=/usr/lib
export TA_INCLUDE_PATH=/usr/include
pip install --no-binary TA-Lib TA-Lib

# Nettoyer
cd ..
rm -rf "$TEMP_DIR"
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "Installation de TA-Lib terminée avec succès!" 