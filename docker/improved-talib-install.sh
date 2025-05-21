#!/bin/bash
set -e

echo "=== Installation améliorée de TA-Lib ==="

# Installer les dépendances système nécessaires
apt-get update
apt-get install -y --no-install-recommends wget build-essential gcc g++ make pkg-config

# Créer un répertoire temporaire pour le téléchargement et la compilation
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Télécharger et installer la bibliothèque C TA-Lib
echo "Téléchargement de TA-Lib 0.4.0..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/

# Configurer et compiler la bibliothèque C
echo "Configuration et compilation de TA-Lib..."
./configure --prefix=/usr
make -j$(nproc)
make install

# S'assurer que les liens symboliques et les répertoires d'en-tête sont correctement configurés
echo "Configuration des en-têtes et des bibliothèques..."
mkdir -p /usr/include/ta-lib
cp -r /usr/include/ta_*.h /usr/include/ta-lib/

# Création des symlinks nécessaires
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta-lib.so
echo "/usr/lib" > /etc/ld.so.conf.d/talib.conf
ldconfig

# Vérifier l'installation de la bibliothèque C
echo "Vérification de l'installation de la bibliothèque C TA-Lib:"
ls -la /usr/lib/libta_lib*
ldconfig -p | grep ta_lib

# Installer des pré-requis pour le wrapper Python
echo "Installation des pré-requis pour le wrapper Python..."
pip install --no-cache-dir --upgrade pip wheel setuptools Cython numpy==1.24.3

# Première stratégie: installer via le canal Anaconda
echo "Tentative d'installation de TA-Lib depuis le canal Anaconda..."
if pip install --no-cache-dir --index-url https://pypi.anaconda.org/ranaroussi/simple ta-lib==0.4.28; then
    echo "✅ TA-Lib installé avec succès depuis le canal Anaconda"
    python -c "import talib; print('TA-Lib importé avec succès!')"
    exit 0
fi

# Deuxième stratégie: installer via GitHub
echo "Tentative d'installation de TA-Lib depuis GitHub..."
if pip install --no-cache-dir git+https://github.com/TA-Lib/ta-lib-python.git@0.4.28; then
    echo "✅ TA-Lib installé avec succès depuis GitHub"
    python -c "import talib; print('TA-Lib importé avec succès!')"
    exit 0
fi

# Troisième stratégie: télécharger, modifier et installer manuellement
echo "Tentative d'installation manuelle de TA-Lib..."
cd "$TEMP_DIR"
git clone https://github.com/TA-Lib/ta-lib-python.git
cd ta-lib-python
sed -i 's|include_dirs=\[\]|include_dirs=["/usr/include", "/usr/include/ta-lib", "/usr/local/include", "/opt/include"]|g' setup.py
sed -i 's|library_dirs=\[\]|library_dirs=["/usr/lib", "/usr/local/lib", "/opt/lib"]|g' setup.py

# Compiler et installer
pip install --no-cache-dir .

# Vérifier l'installation
if python -c "import talib; print('TA-Lib installé avec succès!')"; then
    echo "✅ TA-Lib installé avec succès via installation manuelle"
    exit 0
else
    echo "❌ Échec de l'installation de TA-Lib"
    exit 1
fi
