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
    make

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

# Installer TA-Lib pour Python
echo "Installation du package Python TA-Lib..."
pip install --upgrade numpy
pip install TA-Lib

# Nettoyer
cd ..
rm -rf "$TEMP_DIR"
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "Installation de TA-Lib terminée avec succès!" 