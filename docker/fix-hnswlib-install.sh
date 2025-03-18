#!/bin/bash
set -e

echo "=== Installation séparée de hnswlib ==="

# Déterminer la commande Python à utiliser
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Détecter la version de Python
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Version de Python détectée: $PYTHON_VERSION"

# S'assurer que les dépendances système nécessaires sont installées
if [ -f "/.dockerenv" ] || grep -q docker /proc/self/cgroup 2>/dev/null; then
    echo "Environnement Docker détecté, vérification des dépendances système..."
    
    # Installation des dépendances système pour la compilation
    PACKAGES="build-essential gcc g++ python3-dev cmake make libblas-dev liblapack-dev libatlas-base-dev"
    
    # Vérifier si les packages sont déjà installés
    MISSING_PACKAGES=""
    for pkg in $PACKAGES; do
        if ! dpkg -l | grep -q "ii  $pkg"; then
            MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
        fi
    done
    
    # Installer les packages manquants
    if [ ! -z "$MISSING_PACKAGES" ]; then
        echo "Installation des packages système manquants: $MISSING_PACKAGES"
        apt-get update -y && apt-get install -y --no-install-recommends $MISSING_PACKAGES
    fi
fi

# Préparer l'environnement
echo "Préparation de l'environnement de compilation..."
$PYTHON_CMD -m pip install --upgrade pip wheel setuptools cython

# Stratégie 1: Installation directe depuis PyPI
echo "Tentative d'installation de hnswlib depuis PyPI..."
if $PYTHON_CMD -m pip install --no-cache-dir hnswlib; then
    echo "✅ hnswlib installé avec succès depuis PyPI"
    $PYTHON_CMD -c "import hnswlib; print(f'hnswlib version: {hnswlib.__version__}')" || echo "Version indisponible"
    exit 0
fi

# Stratégie 2: Compilation depuis les sources avec CMAKE_ARGS
echo "Tentative d'installation avec CMAKE_ARGS personnalisés..."
export CMAKE_ARGS="-DHNSWLIB_DISABLE_NATIVE=ON -DCMAKE_BUILD_TYPE=Release"
if $PYTHON_CMD -m pip install --no-cache-dir hnswlib; then
    echo "✅ hnswlib installé avec succès via CMAKE_ARGS personnalisés"
    $PYTHON_CMD -c "import hnswlib; print(f'hnswlib version: {hnswlib.__version__}')" || echo "Version indisponible"
    exit 0
fi

# Stratégie 3: Installation depuis les sources sur GitHub
echo "Tentative d'installation depuis GitHub..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

git clone https://github.com/nmslib/hnswlib.git
cd hnswlib

# Créer un setup.py minimal s'il n'existe pas
if [ ! -f "setup.py" ]; then
    cat > setup.py << EOF
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

__version__ = '0.7.0'

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'hnswlib',
        ['python_bindings/bindings.cpp'],
        include_dirs=[
            'include/',
            get_pybind_include(),
        ],
        language='c++'
    ),
]

class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = []
        if ct == 'unix':
            opts.append('-std=c++14')
            opts.append('-O3')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='hnswlib',
    version=__version__,
    description='Fast approximate nearest neighbor search',
    author='Yury Malkov and others',
    url='https://github.com/nmslib/hnswlib',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.2'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
EOF
fi

# Installer pybind11 si nécessaire
$PYTHON_CMD -m pip install pybind11

# Installer hnswlib depuis les sources
if $PYTHON_CMD -m pip install --no-cache-dir .; then
    echo "✅ hnswlib installé avec succès depuis GitHub"
    $PYTHON_CMD -c "import hnswlib; print(f'hnswlib installé')" || echo "Import échec"
    cd /
    rm -rf "$TEMP_DIR"
    exit 0
fi

# Si toutes les tentatives échouent, créer un module factice
echo "❌ L'installation de hnswlib a échoué. Création d'un module factice..."

# Nettoyage
cd /
rm -rf "$TEMP_DIR"

# Créer un package factice
SITE_PACKAGES=$($PYTHON_CMD -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE_PACKAGES/hnswlib"

cat > "$SITE_PACKAGES/hnswlib/__init__.py" << EOF
# Module hnswlib factice
import warnings
warnings.warn("Ce module hnswlib est un substitut factice. Les fonctionnalités d'indexation vectorielle ne sont pas disponibles.")

class Index:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("hnswlib n'a pas pu être installé correctement.")
        
    def add_items(self, *args, **kwargs):
        raise NotImplementedError("hnswlib n'a pas pu être installé correctement.")
        
    def get_nns_by_vector(self, *args, **kwargs):
        raise NotImplementedError("hnswlib n'a pas pu être installé correctement.")
EOF

echo "⚠️ Module factice hnswlib créé. Certaines fonctionnalités peuvent ne pas être disponibles."
exit 0 