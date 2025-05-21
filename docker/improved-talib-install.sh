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

# Fonction pour nettoyer après une tentative d'installation échouée
cleanup_and_prepare() {
    echo "Nettoyage après échec..."
    make clean || true
    # Supprimer les fichiers potentiellement corrompus
    find . -name "*.Tpo" -delete
    find . -name "*.Po" -delete
}

# Configurer et compiler la bibliothèque C
echo "Configuration et compilation de TA-Lib..."
./configure --prefix=/usr
# Compilation avec un seul thread pour éviter les erreurs de dépendance
echo "Compilation de TA-Lib avec un seul thread pour éviter les erreurs..."
if ! make; then
    echo "Première tentative de compilation échouée, nouvelle tentative après nettoyage..."
    cleanup_and_prepare
    ./configure --prefix=/usr --disable-shared
    if ! make; then
        echo "Deuxième tentative échouée, essai avec des options minimalistes..."
        cleanup_and_prepare
        ./configure --prefix=/usr --disable-shared --enable-static --without-docs
        make
    fi
fi
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

# Ajout de plus de détails sur le système et les fichiers pour diagnostiquer les problèmes
echo "Information système:"
uname -a
echo "Vérification des fichiers d'en-tête et des bibliothèques:"
ls -la /usr/include/ta-lib/
ls -la /usr/lib/libta*

# Modifier setup.py pour utiliser les bons chemins
echo "Modification du fichier setup.py avec les chemins explicites..."
sed -i 's|include_dirs=\[\]|include_dirs=["/usr/include", "/usr/include/ta-lib", "/usr/local/include", "/opt/include"]|g' setup.py
sed -i 's|library_dirs=\[\]|library_dirs=["/usr/lib", "/usr/local/lib", "/opt/lib"]|g' setup.py

# Essayer d'identifier et de corriger d'autres problèmes dans le code source si nécessaire
# Par exemple, vérifier les incompatibilités avec Numpy 1.24+
echo "Recherche et correction de problèmes de compatibilité avec NumPy récent..."
if grep -q "PyArray_Descr" talib/_ta_lib.c 2>/dev/null; then
    echo "Correction de problèmes de compatibilité avec NumPy dans _ta_lib.c..."
    # Cette commande recherche et remplace le code problématique
    sed -i 's/if (d->subarray)/if (PyDataType_HASSUBARRAY(d))/' talib/_ta_lib.c || true
fi

# Compiler et installer avec verbose pour voir les erreurs
echo "Installation du wrapper Python avec mode verbose..."
CFLAGS="-I/usr/include -I/usr/include/ta-lib" LDFLAGS="-L/usr/lib" pip install --no-cache-dir --verbose .

# Vérifier l'installation
if python -c "import talib; print('TA-Lib installé avec succès!')"; then
    echo "✅ TA-Lib installé avec succès via installation manuelle"
    exit 0
else
    echo "❌ Échec de l'installation manuelle, tentative avec pip et des options explicites..."
    cd "$TEMP_DIR"
    # Dernier recours: utiliser pip avec toutes les options explicites
    if pip install --no-cache-dir --global-option=build_ext --global-option="-I/usr/include/" --global-option="-I/usr/include/ta-lib/" --global-option="-L/usr/lib/" TA-Lib==0.4.28; then
        echo "✅ TA-Lib installé avec succès via pip avec options explicites"
        python -c "import talib; print('TA-Lib importé avec succès!')"
        exit 0
    else
        echo "❌ Toutes les tentatives d'installation ont échoué"
        exit 1
    fi
fi
