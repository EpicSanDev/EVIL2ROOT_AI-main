#!/bin/bash
echo "Début de l'installation de la bibliothèque C TA-Lib..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

apt-get update
apt-get install -y --no-install-recommends wget build-essential gcc g++ make pkg-config

# Télécharger et installer la version 0.4.0 de TA-Lib (plus stable)
echo "Téléchargement de TA-Lib 0.4.0..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
echo "Configuration et compilation de TA-Lib..."
./configure --prefix=/usr
make -j$(nproc)
make install
cd ../

# Créer les liens symboliques et les répertoires d'en-tête
echo "Configuration des en-têtes et des bibliothèques..."
mkdir -p /usr/include/ta-lib
cp -r /usr/include/ta_*.h /usr/include/ta-lib/
cp -r /usr/include/ta_*.h /usr/include/

# Symlinks explicites pour les bibliothèques
find /usr/lib -name "libta_*" -type f -o -type l
find /usr/lib -name "libta-lib*" -type f -o -type l

# Création des symlinks nécessaires
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta-lib.so
echo "/usr/lib" > /etc/ld.so.conf.d/talib.conf
ldconfig

echo "Installation de TA-Lib terminée avec succès!"
echo "Contenu du répertoire /usr/include/ta-lib:"
ls -la /usr/include/ta-lib
echo "Contenu de /usr/include avec les en-têtes TA-Lib:"
ls -la /usr/include/ta_*.h

ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta-lib.so
echo "/usr/lib" > /etc/ld.so.conf.d/talib.conf
ldconfig
echo "Installation de TA-Lib terminée avec succès!"

echo "Vérification de l'installation de la bibliothèque C TA-Lib:"
ls -la /usr/lib/libta_lib*
ldconfig -p | grep ta_lib

cd / # Retour au répertoire racine ou un répertoire sûr
rm -rf "$TEMP_DIR"
# Ne pas nettoyer apt ici, car d'autres installations pourraient suivre dans le Dockerfile.
# Le nettoyage d'apt sera fait à la fin des installations dans le Dockerfile.

echo "Installation de la bibliothèque C TA-Lib terminée."