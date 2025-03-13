#!/bin/bash
set -e

echo "=== Installation améliorée de causalml pour environnement cloud ==="

# Déterminer la commande Python à utiliser
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Détecter la version de Python pour aider à sélectionner les versions compatibles
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Version de Python détectée: $PYTHON_VERSION"

# Installer les dépendances systèmes nécessaires (si dans un environnement Docker)
if [ -f "/.dockerenv" ] || grep -q docker /proc/self/cgroup 2>/dev/null; then
    echo "Environnement conteneurisé détecté, installation des dépendances système..."
    apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc g++ python3-dev 2>/dev/null || true
fi

# Installer les dépendances Python requises en évitant la compilation
echo "Installation des prérequis via des wheels pré-compilés..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install --only-binary=:all: --no-cache-dir cython wheel setuptools

# Installer protobuf dans la version compatible avec TensorFlow
echo "Installation de protobuf dans une version compatible avec TensorFlow..."
$PYTHON_CMD -m pip install --no-cache-dir "protobuf>=3.20.3,<5.0.0dev"

# Installer numpy et d'autres dépendances avec des versions spécifiques pour éviter les incompatibilités
echo "Installation des dépendances scientifiques..."
$PYTHON_CMD -m pip install --only-binary=:all: --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    scikit-learn==1.3.0 \
    pandas==2.0.1 \
    matplotlib==3.7.1

echo "Vérification des dépendances installées..."
$PYTHON_CMD -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || echo "NumPy non disponible"
$PYTHON_CMD -c "import scipy; print(f'SciPy version: {scipy.__version__}')" || echo "SciPy non disponible"
$PYTHON_CMD -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')" || echo "Scikit-learn non disponible"
$PYTHON_CMD -c "import google.protobuf; print(f'Protobuf version: {google.protobuf.__version__}')" || echo "Protobuf non disponible"

# Éviter les problèmes de compilation en configurant l'environnement
export SKBUILD_CONFIGURE_OPTIONS="-DBUILD_TESTING=OFF"
export CFLAGS="-std=c99"
export PIP_NO_BUILD_ISOLATION=1
export SKLEARN_NO_OPENMP=1

# Stratégie 1: Utiliser des wheels pré-compilées via pypi avec désactivation du resolver pip
echo "Tentative d'installation à partir de wheels pré-compilées..."

# Pour Python 3.10+, utiliser une version récente
if [[ "${PYTHON_VERSION}" > "3.9" ]]; then
    echo "Python 3.10+ détecté, tentative avec causalml 0.10.0..."
    $PYTHON_CMD -m pip install --no-cache-dir --no-deps causalml==0.10.0
    
    # Installer les dépendances manquantes
    $PYTHON_CMD -m pip install --no-cache-dir statsmodels==0.14.0 graphviz==0.20.1
    
    if $PYTHON_CMD -m pip show causalml > /dev/null 2>&1; then
        echo "✅ causalml 0.10.0 installé avec succès"
        $PYTHON_CMD -m pip show causalml | grep Version
        exit 0
    fi
fi

# Tenter avec version 0.9.0 (plus largement compatible)
echo "Tentative avec causalml 0.9.0..."
$PYTHON_CMD -m pip install --no-cache-dir --no-deps causalml==0.9.0

# Installer les dépendances manquantes
$PYTHON_CMD -m pip install --no-cache-dir statsmodels==0.14.0 graphviz==0.20.1

if $PYTHON_CMD -m pip show causalml > /dev/null 2>&1; then
    echo "✅ causalml 0.9.0 installé avec succès"
    $PYTHON_CMD -m pip show causalml | grep Version
    exit 0
fi

# Stratégie 2: Installer sans la partie Cython (fonctionnalités réduites)
echo "Tentative d'installation sans modules Cython..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Téléchargement des sources causalml..."
git clone https://github.com/uber/causalml.git
cd causalml
git checkout v0.9.0

# Supprimer les extensions Cython du setup.py
echo "Modification du setup.py pour éviter la compilation Cython..."
sed -i 's/ext_modules=cythonize(extensions)/ext_modules=[]/g' setup.py
sed -i 's/extensions = \[.*\]/extensions = []/g' setup.py

# Installer avec pip en mode développement sans résolution de dépendances
echo "Installation sans extensions Cython..."
$PYTHON_CMD -m pip install --no-deps -e .

# Installer les dépendances manquantes manuellement
$PYTHON_CMD -m pip install --no-cache-dir statsmodels==0.14.0 graphviz==0.20.1

if $PYTHON_CMD -m pip show causalml > /dev/null 2>&1; then
    echo "✅ causalml installé avec succès (sans extensions Cython)"
    $PYTHON_CMD -m pip show causalml | grep Version
    cd /
    rm -rf "$TEMP_DIR"
    exit 0
fi

# Si toutes les tentatives ont échoué, créer un package factice
echo "❌ Toutes les tentatives d'installation ont échoué."
echo "Création d'un package factice pour éviter les erreurs d'importation..."

# Nettoyage
cd /
rm -rf "$TEMP_DIR"

# Créer un package factice plus complet
SITE_PACKAGES=$($PYTHON_CMD -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE_PACKAGES/causalml/inference/tree"
mkdir -p "$SITE_PACKAGES/causalml/inference/meta"
mkdir -p "$SITE_PACKAGES/causalml/metrics"
mkdir -p "$SITE_PACKAGES/causalml/dataset"
mkdir -p "$SITE_PACKAGES/causalml/propensity"

cat > "$SITE_PACKAGES/causalml/__init__.py" << EOF
# Package causalml factice
import warnings
warnings.warn("Ce package causalml est un substitut factice. Les fonctionnalités d'inférence causale ne sont pas disponibles.")

__version__ = "0.11.0-mock"
EOF

cat > "$SITE_PACKAGES/causalml/inference/tree/__init__.py" << EOF
class UpliftTreeClassifier:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("causalml n'a pas pu être installé correctement.")

class UpliftRandomForestClassifier:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("causalml n'a pas pu être installé correctement.")
EOF

cat > "$SITE_PACKAGES/causalml/inference/meta/__init__.py" << EOF
class BaseRLearner:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("causalml n'a pas pu être installé correctement.")

class BaseSLearner:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("causalml n'a pas pu être installé correctement.")

class BaseTLearner:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("causalml n'a pas pu être installé correctement.")

class BaseXLearner:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("causalml n'a pas pu être installé correctement.")
EOF

cat > "$SITE_PACKAGES/causalml/metrics/__init__.py" << EOF
def get_cumgain(*args, **kwargs):
    raise NotImplementedError("causalml n'a pas pu être installé correctement.")

def get_qini(*args, **kwargs):
    raise NotImplementedError("causalml n'a pas pu être installé correctement.")

def plot_gain(*args, **kwargs):
    raise NotImplementedError("causalml n'a pas pu être installé correctement.")

def plot_qini(*args, **kwargs):
    raise NotImplementedError("causalml n'a pas pu être installé correctement.")
EOF

cat > "$SITE_PACKAGES/causalml/dataset/__init__.py" << EOF
def synthetic_data(*args, **kwargs):
    raise NotImplementedError("causalml n'a pas pu être installé correctement.")
EOF

cat > "$SITE_PACKAGES/causalml/propensity/__init__.py" << EOF
def ElasticNetPropensityModel(*args, **kwargs):
    raise NotImplementedError("causalml n'a pas pu être installé correctement.")
EOF

echo "⚠️ Installation de causalml échouée - un package factice a été créé pour éviter les erreurs"
exit 1 