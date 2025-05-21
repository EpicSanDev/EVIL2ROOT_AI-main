#!/bin/bash
echo "Début de l'installation de la bibliothèque C TA-Lib..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

apt-get update
apt-get install -y --no-install-recommends wget build-essential gcc g++ make pkg-config

# Utiliser une version plus récente de TA-Lib pour être compatible avec le wrapper Python >= 0.4.28
echo "Téléchargement et installation de TA-Lib depuis le dépôt git..."
git clone --depth=1 https://github.com/TA-Lib/ta-lib.git
cd ta-lib/
./autogen.sh
echo "Configuration et compilation de TA-Lib..."
./configure --prefix=/usr
make -j$(nproc)
make install
cd ../

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