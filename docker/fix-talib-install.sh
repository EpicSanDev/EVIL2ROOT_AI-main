#!/bin/bash
echo "Début de l'installation de la bibliothèque C TA-Lib..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

apt-get update
apt-get install -y --no-install-recommends wget build-essential gcc g++ make pkg-config

echo "Téléchargement de TA-Lib 0.4.0..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
echo "Configuration et compilation de TA-Lib..."
./configure --prefix=/usr --build=$(uname -m)-unknown-linux-gnu
make -j$(nproc)
make install
cd ../

ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta-lib.so
ldconfig

echo "Vérification de l'installation de la bibliothèque C TA-Lib:"
ls -la /usr/lib/libta_lib*
ldconfig -p | grep ta_lib

cd / # Retour au répertoire racine ou un répertoire sûr
rm -rf "$TEMP_DIR"
# Ne pas nettoyer apt ici, car d'autres installations pourraient suivre dans le Dockerfile.
# Le nettoyage d'apt sera fait à la fin des installations dans le Dockerfile.

echo "Installation de la bibliothèque C TA-Lib terminée."