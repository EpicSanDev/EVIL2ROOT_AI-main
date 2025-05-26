#!/bin/bash
set -e

echo "=== Installation de secours pour TA-Lib ==="

# Fonction pour afficher les bandeaux
print_banner() {
    echo
    echo "==============================================="
    echo "  $1"
    echo "==============================================="
    echo
}

# Installer les dépendances système nécessaires
print_banner "Installation des dépendances"
apt-get update
apt-get install -y --no-install-recommends wget build-essential gcc g++ make pkg-config git curl ca-certificates autoconf libtool

# Créer un répertoire temporaire
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

# Télécharger une version spécifique de TA-Lib source
print_banner "Téléchargement de TA-Lib 0.4.0"
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O ta-lib.tar.gz || {
    echo "Échec du téléchargement depuis SourceForge, essai avec GitHub..."
    wget https://github.com/TA-Lib/ta-lib/archive/refs/tags/v0.4.0.tar.gz -O ta-lib.tar.gz || {
        echo "Échec du téléchargement. Tentative avec un miroir alternatif..."
        curl -L https://anaconda.org/conda-forge/ta-lib/0.4.0/download/linux-64/ta-lib-0.4.0-h516909a_0.tar.bz2 -o ta-lib.tar.bz2
        mkdir -p ta-lib-extract
        tar -xjf ta-lib.tar.bz2 -C ta-lib-extract
        cd ta-lib-extract
        # Installer depuis les binaires précompilés
        if [ -d "lib" ]; then
            cp -r lib/* /usr/lib/ || true
        fi
        if [ -d "include" ]; then
            cp -r include/* /usr/include/ || true
        fi
        
        # Vérifier si l'installation a réussi
        if [ -f "/usr/lib/libta_lib.so" ] || [ -f "/usr/lib/libta_lib.a" ]; then
            print_banner "Installation depuis les binaires précompilés réussie"
            ldconfig
            mkdir -p /usr/include/ta-lib
            cp -r /usr/include/ta_*.h /usr/include/ta-lib/ 2>/dev/null || true
            cd "$TEMP_DIR"
            # Passer à l'installation Python
            goto_python_install
        else
            echo "Échec de l'installation depuis les binaires précompilés."
            cd "$TEMP_DIR"
        fi
    }
}

# Extraire l'archive
if [ -f ta-lib.tar.gz ]; then
    tar -xzf ta-lib.tar.gz
    # Vérifier si le dossier est nommé ta-lib ou ta-lib-0.4.0
    if [ -d ta-lib ]; then
        cd ta-lib
    elif [ -d ta-lib-0.4.0 ]; then
        cd ta-lib-0.4.0
    else
        echo "Structure de dossier inattendue. Recherche du dossier ta-lib..."
        TA_LIB_DIR=$(find . -type d -name "ta-lib*" | head -1)
        if [ -n "$TA_LIB_DIR" ]; then
            cd "$TA_LIB_DIR"
        else
            echo "Impossible de trouver le dossier ta-lib."
            exit 1
        fi
    fi
else
    echo "L'archive TA-Lib n'a pas été téléchargée correctement."
    exit 1
fi

# Mettre à jour les scripts de détection d'architecture
print_banner "Mise à jour des scripts de détection d'architecture"
mkdir -p config
echo "Téléchargement des scripts config.guess et config.sub pour ARM64..."
curl -s -o config/config.guess 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD' || {
    echo "Échec du téléchargement depuis git.savannah. Essai avec GitHub..."
    curl -s -o config/config.guess 'https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.guess'
}

curl -s -o config/config.sub 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD' || {
    echo "Échec du téléchargement depuis git.savannah. Essai avec GitHub..."
    curl -s -o config/config.sub 'https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.sub'
}

chmod +x config/config.guess config/config.sub

# Vérifier que les fichiers sont valides
if [ ! -s config/config.guess ] || [ ! -s config/config.sub ]; then
    echo "Les fichiers config.guess ou config.sub n'ont pas été téléchargés correctement."
    echo "Création de versions minimales locales..."
    
    # Créer un script config.guess minimal
    cat > config/config.guess << 'EOF'
#!/bin/sh
cpu=$(uname -m)
os=$(uname -s | tr '[:upper:]' '[:lower:]')
echo "${cpu}-unknown-${os}"
EOF
    chmod +x config/config.guess
    
    # Créer un script config.sub minimal qui accepte ARM64
    cat > config/config.sub << 'EOF'
#!/bin/sh
echo $1 | sed 's/aarch64-unknown-linux-gnu/aarch64-unknown-linux-gnu/'
EOF
    chmod +x config/config.sub
fi

# Fonction pour nettoyer après une tentative échouée
cleanup_and_retry() {
    echo "Nettoyage pour nouvelle tentative..."
    make clean >/dev/null 2>&1 || true
    find . -name "*.o" -delete
    find . -name "*.lo" -delete
    find . -name "*.la" -delete
    find . -name "*.a" -delete
    find . -name "*.Tpo" -delete
    find . -name ".deps" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".libs" -type d -exec rm -rf {} + 2>/dev/null || true
}

# Compiler et installer TA-Lib avec une configuration adaptée à l'architecture
print_banner "Compilation de TA-Lib"

# Options de configuration spécifiques à l'architecture
CONFIG_OPTS="--prefix=/usr"

if [ "$IS_ARM64" = "true" ]; then
    print_banner "Configuration spécifique pour ARM64"
    CONFIG_OPTS="$CONFIG_OPTS --build=aarch64-unknown-linux-gnu"
    
    # Optimisations pour ARM64
    export CFLAGS="-O3 -pipe -fomit-frame-pointer -march=armv8-a+crc -mcpu=generic"
    export CXXFLAGS="$CFLAGS"
    
    # Ajouter des options pour éviter des problèmes spécifiques à ARM64
    CONFIG_OPTS="$CONFIG_OPTS --disable-dependency-tracking"
else
    # Optimisations génériques
    export CFLAGS="-O2 -pipe"
    export CXXFLAGS="$CFLAGS"
fi

# Stratégies de compilation progressives
COMPILE_SUCCESS=false

# Stratégie 1: Configuration standard avec bibliothèques partagées désactivées
print_banner "Tentative 1: Configuration avec bibliothèques statiques uniquement"
CONFIG_OPTS="$CONFIG_OPTS --disable-shared --enable-static"

echo "Configuration: ./configure $CONFIG_OPTS"
./configure $CONFIG_OPTS

echo "Compilation avec parallélisme..."
if make -j$(nproc); then
    COMPILE_SUCCESS=true
    echo "Compilation réussie avec la stratégie 1!"
else
    echo "Échec de la compilation avec la stratégie 1."
    
    # Stratégie 2: Configuration minimale avec un seul thread
    print_banner "Tentative 2: Configuration minimale"
    cleanup_and_retry
    
    CONFIG_OPTS="$CONFIG_OPTS --without-docs"
    echo "Configuration: ./configure $CONFIG_OPTS"
    ./configure $CONFIG_OPTS
    
    echo "Compilation avec un seul thread..."
    if make -j1; then
        COMPILE_SUCCESS=true
        echo "Compilation réussie avec la stratégie 2!"
    else
        echo "Échec de la compilation avec la stratégie 2."
        
        # Stratégie 3: Configuration ultra-minimale avec patches
        print_banner "Tentative 3: Configuration ultra-minimale avec patches"
        cleanup_and_retry
        
        # Appliquer des patches pour corriger des problèmes potentiels
        echo "Application de patches..."
        
        # Patch pour src/ta_common/ta_global.c (problème connu sur ARM64)
        if [ -f src/ta_common/ta_global.c ]; then
            echo "Patching ta_global.c..."
            sed -i 's/TA_FIT_FUNCTION( ta_gDataTable/TA_INTERNAL_FUNCTION(void) TA_FIT_FUNCTION( TA_GlobalTable/' src/ta_common/ta_global.c || true
        fi
        
        # Patch pour src/ta_common/ta_defs.c
        if [ -f src/ta_common/ta_defs.c ]; then
            echo "Patching ta_defs.c..."
            sed -i 's/int64_t/int64_t;/g' src/ta_common/ta_defs.c || true
        fi
        
        # Régénérer les fichiers autoconf si nécessaire
        if [ -f autogen.sh ]; then
            echo "Régénération des fichiers autoconf..."
            ./autogen.sh || true
        fi
        
        # Utiliser une configuration encore plus minimale
        CONFIG_OPTS="--prefix=/usr --disable-shared --enable-static --without-docs --disable-dependency-tracking"
        if [ "$IS_ARM64" = "true" ]; then
            CONFIG_OPTS="$CONFIG_OPTS --build=aarch64-unknown-linux-gnu"
        fi
        
        echo "Configuration: ./configure $CONFIG_OPTS"
        ./configure $CONFIG_OPTS
        
        # Compilation avec options extrêmes
        export CFLAGS="$CFLAGS -DNDEBUG -DTA_DISABLE_THREADS"
        export MAKEFLAGS="-j1"
        
        echo "Compilation avec options extrêmes..."
        if make; then
            COMPILE_SUCCESS=true
            echo "Compilation réussie avec la stratégie 3!"
        else
            echo "Échec de toutes les stratégies de compilation."
            
            if [ "$IS_ARM64" = "true" ] && [ -f /tmp/talib-arm64-mock.sh ]; then
                print_banner "Utilisation du fallback Mock pour ARM64"
                /tmp/talib-arm64-mock.sh
                exit $?
            else
                echo "Impossible de compiler TA-Lib."
                exit 1
            fi
        fi
    fi
fi

# Installer la bibliothèque
if [ "$COMPILE_SUCCESS" = "true" ]; then
    print_banner "Installation de TA-Lib"
    make install
    
    # Configurer correctement les bibliothèques
    echo "Configuration des liens symboliques..."
    mkdir -p /usr/include/ta-lib
    
    # Copier les en-têtes vers le dossier standard
    cp -r src/ta_common/*.h /usr/include/ta-lib/ 2>/dev/null || true
    cp -r include/*.h /usr/include/ta-lib/ 2>/dev/null || true
    cp -r /usr/include/ta_*.h /usr/include/ta-lib/ 2>/dev/null || true
    
    # Créer les liens symboliques si nécessaire
    if [ -f /usr/lib/libta_lib.so.0 ]; then
        ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so
        ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta-lib.so
    fi
    
    # Créer aussi des liens pour les bibliothèques statiques
    if [ -f /usr/lib/libta_lib.a ]; then
        ln -sf /usr/lib/libta_lib.a /usr/lib/libta-lib.a
    fi
    
    # Mettre à jour le cache des bibliothèques
    echo "/usr/lib" > /etc/ld.so.conf.d/talib.conf
    ldconfig
    
    # Vérification
    print_banner "Vérification de l'installation C"
    ls -la /usr/lib/libta* || echo "Avertissement: Aucune bibliothèque trouvée!"
    ls -la /usr/include/ta-lib/ || echo "Avertissement: Aucun en-tête trouvé!"
fi

# Fonction pour passer à l'installation du wrapper Python
goto_python_install() {
    print_banner "Installation du wrapper Python TA-Lib"
    
    # Installer les pré-requis
    pip install --no-cache-dir --upgrade pip wheel setuptools Cython numpy==1.24.3
    
    # Différentes stratégies d'installation
    
    # Stratégie 1: Installation via Anaconda (pré-compilé)
    echo "Tentative d'installation depuis le canal Anaconda..."
    if [ "$IS_ARM64" = "true" ]; then
        # Canaux spécifiques pour ARM64
        pip install --no-cache-dir --extra-index-url https://pypi.anaconda.org/scipy-wheels-nightly/simple \
                                --extra-index-url https://pypi.anaconda.org/ranaroussi/simple ta-lib==0.4.28
    else
        pip install --no-cache-dir --index-url https://pypi.anaconda.org/ranaroussi/simple ta-lib==0.4.28
    fi
    
    # Vérifier si l'installation a réussi
    if python -c "import talib; print('TA-Lib importé avec succès!')" 2>/dev/null; then
        print_banner "✅ TA-Lib installé avec succès depuis Anaconda"
        return 0
    fi
    
    # Stratégie 2: Installation depuis GitHub
    echo "Tentative d'installation depuis GitHub..."
    pip install --no-cache-dir git+https://github.com/TA-Lib/ta-lib-python.git@0.4.28
    
    # Vérifier si l'installation a réussi
    if python -c "import talib; print('TA-Lib importé avec succès!')" 2>/dev/null; then
        print_banner "✅ TA-Lib installé avec succès depuis GitHub"
        return 0
    fi
    
    # Stratégie 3: Installation manuelle optimisée
    echo "Tentative d'installation manuelle optimisée..."
    cd "$TEMP_DIR"
    git clone https://github.com/TA-Lib/ta-lib-python.git
    cd ta-lib-python
    
    # Modifier setup.py pour utiliser les bons chemins
    sed -i 's|include_dirs=\[\]|include_dirs=["/usr/include", "/usr/include/ta-lib", "/usr/local/include"]|g' setup.py
    sed -i 's|library_dirs=\[\]|library_dirs=["/usr/lib", "/usr/local/lib"]|g' setup.py
    
    # Optimisations d'architecture
    if [ "$IS_ARM64" = "true" ]; then
        echo "Ajout d'optimisations ARM64 dans setup.py..."
        sed -i 's|extra_compile_args=\[\]|extra_compile_args=["-O3", "-march=armv8-a+crc", "-mcpu=generic"]|g' setup.py
    else
        sed -i 's|extra_compile_args=\[\]|extra_compile_args=["-O3"]|g' setup.py
    fi
    
    # Corriger les problèmes de compatibilité NumPy
    if grep -q "PyArray_Descr" talib/_ta_lib.c 2>/dev/null; then
        echo "Correction de problèmes de compatibilité avec NumPy..."
        sed -i 's/if (d->subarray)/if (PyDataType_HASSUBARRAY(d))/' talib/_ta_lib.c || true
    fi
    
    # Installation
    if [ "$IS_ARM64" = "true" ]; then
        CFLAGS="-I/usr/include -I/usr/include/ta-lib -O3 -march=armv8-a+crc" \
        LDFLAGS="-L/usr/lib" \
        pip install --no-cache-dir --verbose .
    else
        CFLAGS="-I/usr/include -I/usr/include/ta-lib -O3" \
        LDFLAGS="-L/usr/lib" \
        pip install --no-cache-dir --verbose .
    fi
    
    # Vérifier si l'installation a réussi
    if python -c "import talib; print('TA-Lib importé avec succès!')" 2>/dev/null; then
        print_banner "✅ TA-Lib installé avec succès via installation manuelle"
        return 0
    fi
    
    # Stratégie 4: Options explicites via pip
    echo "Tentative avec options explicites de pip..."
    cd "$TEMP_DIR"
    
    # Dernière tentative
    if [ "$IS_ARM64" = "true" ]; then
        pip install --no-cache-dir --global-option=build_ext \
                    --global-option="-I/usr/include/" \
                    --global-option="-I/usr/include/ta-lib/" \
                    --global-option="-L/usr/lib/" \
                    --global-option="-march=armv8-a+crc" \
                    TA-Lib==0.4.28
    else
        pip install --no-cache-dir --global-option=build_ext \
                    --global-option="-I/usr/include/" \
                    --global-option="-I/usr/include/ta-lib/" \
                    --global-option="-L/usr/lib/" \
                    TA-Lib==0.4.28
    fi
    
    # Vérification finale
    if python -c "import talib; print('TA-Lib importé avec succès!')" 2>/dev/null; then
        print_banner "✅ TA-Lib installé avec succès via pip avec options explicites"
        return 0
    fi
    
    # Si tout a échoué pour ARM64, utiliser le mock
    if [ "$IS_ARM64" = "true" ] && [ -f /tmp/talib-arm64-mock.sh ]; then
        print_banner "❌ Échec de toutes les méthodes d'installation, utilisation du mock ARM64"
        /tmp/talib-arm64-mock.sh
        return $?
    fi
    
    print_banner "❌ Toutes les tentatives d'installation ont échoué"
    return 1
}

# Installation du wrapper Python si la compilation a réussi
if [ "$COMPILE_SUCCESS" = "true" ]; then
    goto_python_install
fi
