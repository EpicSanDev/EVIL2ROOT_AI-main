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

# Rechercher les bibliothèques TA-Lib installées
find /usr/lib -name "libta_*" -type f -o -type l
find /usr/lib -name "libta-lib*" -type f -o -type l

# Symlinks
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta-lib.so
echo "/usr/lib" > /etc/ld.so.conf.d/talib.conf
ldconfig

# Configurer les variables d'environnement pour pip install
export TA_INCLUDE_PATH=/usr/include
export TA_LIBRARY_PATH=/usr/lib

echo "Installation de TA-Lib Python wrapper avec les flags spécifiques..."
pip install --global-option=build_ext --global-option="-I/usr/include/" --global-option="-L/usr/lib/" TA-Lib==0.4.28

# Vérifier l'installation
python -c "import talib; print('TA-Lib importé avec succès!')"

echo "Installation de TA-Lib terminée!"
echo "Contenu du répertoire /usr/include/ta-lib:"
ls -la /usr/include/ta-lib
