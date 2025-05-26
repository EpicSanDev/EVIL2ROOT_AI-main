#!/bin/bash
set -e

echo "=== Installation améliorée de TA-Lib ==="

# Installer les dépendances système nécessaires
apt-get update
apt-get install -y --no-install-recommends wget build-essential gcc g++ make pkg-config curl ca-certificates autoconf

# Fonction pour afficher les bandeaux
print_banner() {
    echo
    echo "==============================================="
    echo "  $1"
    echo "==============================================="
    echo
}

# Créer un répertoire temporaire pour le téléchargement et la compilation
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Détection de l'architecture
ARCH=$(uname -m)
echo "Architecture détectée: $ARCH"
IS_ARM64=false
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    IS_ARM64=true
    echo "Configuration pour ARM64 activée"
fi

# Télécharger et installer la bibliothèque C TA-Lib
print_banner "Téléchargement de TA-Lib 0.4.0"
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/

# Vérifier et mettre à jour les scripts config.guess et config.sub pour ARM64
print_banner "Mise à jour des scripts de détection d'architecture"
echo "Téléchargement des scripts config actualisés depuis GNU..."
mkdir -p config
curl -s -o config/config.guess 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD'
curl -s -o config/config.sub 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD'
chmod +x config/config.guess config/config.sub

# Vérifier que les fichiers ont été correctement téléchargés
if [[ ! -s config/config.guess || ! -s config/config.sub ]]; then
    echo "Erreur: Impossible de télécharger les scripts config à jour. Tentative avec un miroir alternatif..."
    curl -s -o config/config.guess 'https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.guess'
    curl -s -o config/config.sub 'https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.sub'
    chmod +x config/config.guess config/config.sub
    
    if [[ ! -s config/config.guess || ! -s config/config.sub ]]; then
        echo "Erreur critique: Impossible de mettre à jour les scripts config. Création manuelle..."
        # Créer un script config.guess minimal qui reconnaît ARM64
        cat > config/config.guess << 'EOF'
#!/bin/sh
cpu=$(uname -m)
os=$(uname -s)
echo "$cpu-unknown-$os"
EOF
        chmod +x config/config.guess
    fi
fi

# Fonction pour nettoyer après une tentative d'installation échouée
cleanup_and_prepare() {
    echo "Nettoyage après échec..."
    make clean || true
    # Supprimer les fichiers potentiellement corrompus
    find . -name "*.Tpo" -delete
    find . -name "*.Po" -delete
}

# Configurer et compiler la bibliothèque C
print_banner "Configuration et compilation de TA-Lib"

# Options de configuration spécifiques à l'architecture
CONFIG_OPTS="--prefix=/usr"
if [ "$IS_ARM64" = "true" ]; then
    echo "Configuration spécifique pour ARM64..."
    CONFIG_OPTS="$CONFIG_OPTS --build=aarch64-unknown-linux-gnu"
    
    # Ajouter des optimisations pour ARM64
    export CFLAGS="-O3 -pipe -fomit-frame-pointer -march=armv8-a+crc -mcpu=generic"
    export CXXFLAGS="$CFLAGS"
    
    # Désactiver certaines fonctionnalités problématiques sur ARM64
    CONFIG_OPTS="$CONFIG_OPTS --disable-dependency-tracking"
else
    # Optimisations génériques pour x86_64
    export CFLAGS="-O3 -pipe -fomit-frame-pointer"
    export CXXFLAGS="$CFLAGS"
fi

# Stratégies de compilation progressives
# Stratégie 1: Configuration standard
print_banner "Compilation stratégie 1: Standard"
if [ "$IS_ARM64" = "true" ]; then
    echo "Utilisation de la configuration ARM64 spécifique"
    ./configure $CONFIG_OPTS --enable-static --disable-shared
else
    ./configure $CONFIG_OPTS
fi

echo "Compilation de TA-Lib (tentative 1)..."
if ! make -j$(nproc); then
    print_banner "Tentative 1 échouée - Stratégie 2: Mode minimaliste"
    cleanup_and_prepare
    
    # Stratégie 2: Configuration minimale
    CONFIG_OPTS="$CONFIG_OPTS --disable-shared --enable-static"
    ./configure $CONFIG_OPTS
    
    echo "Compilation de TA-Lib (tentative 2)..."
    if ! make -j$(nproc); then
        print_banner "Tentative 2 échouée - Stratégie 3: Mode ultra-minimaliste"
        cleanup_and_prepare
        
        # Stratégie 3: Configuration ultra-minimale
        CONFIG_OPTS="$CONFIG_OPTS --without-docs"
        export CFLAGS="$CFLAGS -DNDEBUG"
        ./configure $CONFIG_OPTS
        
        echo "Compilation de TA-Lib (tentative 3)..."
        if ! make -j1; then
            print_banner "Toutes les tentatives ont échoué"
            if [ "$IS_ARM64" = "true" ]; then
                echo "Basculement vers l'approche arm64 spécifique..."
                cd "$TEMP_DIR"
                # Tentative alternative ARM64-spécifique
                if [ -f /tmp/talib-arm64-mock.sh ]; then
                    echo "Utilisation du script ARM64 mock..."
                    /tmp/talib-arm64-mock.sh
                    exit $?
                else
                    echo "Script mock ARM64 non trouvé"
                    exit 1
                fi
            else
                exit 1
            fi
        fi
    fi
fi

print_banner "Installation de TA-Lib"
make install

# S'assurer que les liens symboliques et les répertoires d'en-tête sont correctement configurés
echo "Configuration des en-têtes et des bibliothèques..."
mkdir -p /usr/include/ta-lib
cp -r src/ta_common/*.h /usr/include/ta-lib/ 2>/dev/null || true
cp -r include/*.h /usr/include/ta-lib/ 2>/dev/null || true
cp -r /usr/include/ta_*.h /usr/include/ta-lib/ 2>/dev/null || true

# Création des symlinks nécessaires
if [ -f /usr/lib/libta_lib.so.0 ]; then
    ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so
    ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta-lib.so
fi

if [ -f /usr/lib/libta_lib.a ]; then
    # Créer le symlink pour la version statique aussi
    ln -sf /usr/lib/libta_lib.a /usr/lib/libta-lib.a
fi

echo "/usr/lib" > /etc/ld.so.conf.d/talib.conf
ldconfig

# Vérifier l'installation de la bibliothèque C
print_banner "Vérification de l'installation de la bibliothèque C TA-Lib"
ls -la /usr/lib/libta_lib* 2>/dev/null || echo "Aucune bibliothèque trouvée"
ldconfig -p | grep ta_lib || echo "Aucune bibliothèque trouvée dans le cache ldconfig"

# Installer des pré-requis pour le wrapper Python
print_banner "Installation des pré-requis pour le wrapper Python"
pip install --no-cache-dir --upgrade pip wheel setuptools Cython numpy==1.24.3

# Fonction pour vérifier l'installation
verify_talib_install() {
    if python -c "import talib; print('TA-Lib importé avec succès: ' + talib.__version__)"; then
        return 0
    else
        return 1
    fi
}

# Stratégies d'installation du wrapper Python
# Stratégie 1: utiliser un canal pré-compilé (plus rapide)
print_banner "Stratégie 1: Installation depuis Anaconda (pré-compilé)"
if [ "$IS_ARM64" = "true" ]; then
    echo "Recherche de binaires ARM64 pré-compilés..."
    # Pour ARM64, chercher des versions spécifiques à l'architecture
    pip install --no-cache-dir --extra-index-url https://pypi.anaconda.org/scipy-wheels-nightly/simple --extra-index-url https://pypi.anaconda.org/ranaroussi/simple ta-lib==0.4.28
else
    # Canal standard pour x86_64
    pip install --no-cache-dir --index-url https://pypi.anaconda.org/ranaroussi/simple ta-lib==0.4.28
fi

if verify_talib_install; then
    print_banner "✅ TA-Lib installé avec succès depuis le canal pré-compilé"
    exit 0
fi

# Stratégie 2: installer depuis GitHub
print_banner "Stratégie 2: Installation depuis GitHub"
if pip install --no-cache-dir git+https://github.com/TA-Lib/ta-lib-python.git@0.4.28; then
    if verify_talib_install; then
        print_banner "✅ TA-Lib installé avec succès depuis GitHub"
        exit 0
    fi
fi

# Stratégie 3: Compilation manuelle optimisée pour l'architecture
print_banner "Stratégie 3: Installation manuelle optimisée"
cd "$TEMP_DIR"
git clone https://github.com/TA-Lib/ta-lib-python.git
cd ta-lib-python

# Informations de débogage
echo "Information système:"
uname -a
echo "Vérification des fichiers d'en-tête et des bibliothèques:"
ls -la /usr/include/ta-lib/ 2>/dev/null || echo "Répertoire d'en-têtes introuvable"
ls -la /usr/lib/libta* 2>/dev/null || echo "Bibliothèques introuvables"

# Modifier setup.py pour utiliser les bons chemins et options
echo "Modification du fichier setup.py avec les chemins explicites..."
sed -i 's|include_dirs=\[\]|include_dirs=["/usr/include", "/usr/include/ta-lib", "/usr/local/include", "/opt/include"]|g' setup.py
sed -i 's|library_dirs=\[\]|library_dirs=["/usr/lib", "/usr/local/lib", "/opt/lib"]|g' setup.py

# Optimisations d'architectures
if [ "$IS_ARM64" = "true" ]; then
    echo "Ajout d'optimisations ARM64 dans setup.py..."
    sed -i 's|extra_compile_args=\[\]|extra_compile_args=["-O3", "-march=armv8-a+crc", "-mcpu=generic"]|g' setup.py
else
    echo "Ajout d'optimisations génériques dans setup.py..."
    sed -i 's|extra_compile_args=\[\]|extra_compile_args=["-O3"]|g' setup.py
fi

# Correction des problèmes de compatibilité NumPy
if grep -q "PyArray_Descr" talib/_ta_lib.c 2>/dev/null; then
    echo "Correction de problèmes de compatibilité avec NumPy dans _ta_lib.c..."
    sed -i 's/if (d->subarray)/if (PyDataType_HASSUBARRAY(d))/' talib/_ta_lib.c || true
fi

# Compilation et installation
print_banner "Compilation optimisée du wrapper Python"
if [ "$IS_ARM64" = "true" ]; then
    CFLAGS="-I/usr/include -I/usr/include/ta-lib -O3 -march=armv8-a+crc -mcpu=generic" \
    LDFLAGS="-L/usr/lib" \
    pip install --no-cache-dir --verbose .
else
    CFLAGS="-I/usr/include -I/usr/include/ta-lib -O3" \
    LDFLAGS="-L/usr/lib" \
    pip install --no-cache-dir --verbose .
fi

# Vérification finale
if verify_talib_install; then
    print_banner "✅ TA-Lib installé avec succès via installation manuelle optimisée"
    exit 0
else
    print_banner "Tentative d'installation avec options explicites de pip"
    cd "$TEMP_DIR"
    # Dernier recours: utiliser pip avec toutes les options explicites
    if pip install --no-cache-dir --global-option=build_ext --global-option="-I/usr/include/" --global-option="-I/usr/include/ta-lib/" --global-option="-L/usr/lib/" TA-Lib==0.4.28; then
        if verify_talib_install; then
            print_banner "✅ TA-Lib installé avec succès via pip avec options explicites"
            exit 0
        fi
    fi
    
    # Si tout échoue et nous sommes sur ARM64, utiliser la mock implementation
    if [ "$IS_ARM64" = "true" ] && [ -f /tmp/talib-arm64-mock.sh ]; then
        print_banner "❌ Installation native échouée sur ARM64, utilisation du mock"
        /tmp/talib-arm64-mock.sh
        exit $?
    else
        print_banner "❌ Toutes les tentatives d'installation ont échoué"
        exit 1
    fi
fi
