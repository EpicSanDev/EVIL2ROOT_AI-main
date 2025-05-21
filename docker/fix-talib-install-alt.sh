#!/bin/bash
echo "Début de l'installation alternative de la bibliothèque C TA-Lib..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

apt-get update
apt-get install -y --no-install-recommends wget build-essential gcc g++ make pkg-config

# Méthode directe en téléchargeant le paquet depuis PyPI
echo "Téléchargement du paquet source TA-Lib depuis PyPI..."
pip download --no-binary :all: TA-Lib==0.4.28
tar -xzf TA-Lib-0.4.28.tar.gz
cd TA-Lib-0.4.28

# Télécharger et installer la version compatible de la bibliothèque C
echo "Téléchargement de la bibliothèque C TA-Lib..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make -j$(nproc)
make install
cd ..

ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta-lib.so
echo "/usr/lib" > /etc/ld.so.conf.d/talib.conf
ldconfig

# Créer une structure de répertoire correcte pour les en-têtes
echo "Création des répertoires d'en-têtes..."
mkdir -p /usr/include/ta-lib

# Copier tous les en-têtes vers le répertoire ta-lib
cp -r /usr/include/ta_*.h /usr/include/ta-lib/

# Installation de TA-Lib via pip avec options de build spécifiques
echo "Installation du wrapper Python TA-Lib..."
pip install --global-option=build_ext --global-option="-I/usr/include/" --global-option="-L/usr/lib/" TA-Lib==0.4.28

echo "Installation alternative de TA-Lib terminée avec succès!"
echo "Contenu du répertoire /usr/include/ta-lib:"
ls -la /usr/include/ta-lib
echo "Contenu de /usr/include avec les en-têtes TA-Lib:"
ls -la /usr/include/ta_*.h
