#!/bin/bash
set -e

echo "=== Installation améliorée de causalml ==="

# Installer les dépendances requises
echo "Installation des prérequis..."
pip install --no-cache-dir cython==0.29.36 numpy==1.24.3 scipy==1.10.1 scikit-learn==1.3.0 pandas==2.0.1 matplotlib==3.7.1

# Créer un répertoire temporaire pour les sources
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Téléchargement des sources causalml 0.11.0..."
git clone https://github.com/uber/causalml.git
cd causalml
git checkout v0.11.0

echo "Application des correctifs pour les fichiers Cython..."

# Ajouter les imports et la directive de language level
for file in causalml/inference/tree/*.pyx; do
    echo "Correction de $file"
    sed -i '1s/^/#cython: language_level=3\nimport numpy as np\ncimport numpy as np\nfrom numpy import ndarray\n\n/' "$file"
done

# Corriger les types dans causaltree.pyx
echo "Correction des erreurs de type dans causaltree.pyx..."
sed -i 's/cdef SIZE_t\* samples = self\.samples/cdef np.ndarray[np.npy_intp, ndim=1] samples = np.asarray(self.samples, dtype=np.intp)/g' causalml/inference/tree/causaltree.pyx
sed -i 's/cdef DOUBLE_t\* sample_weight = self\.sample_weight/cdef np.ndarray[np.float64_t, ndim=1] sample_weight = np.asarray(self.sample_weight, dtype=np.float64)/g' causalml/inference/tree/causaltree.pyx

# Remplacer 'nogil' par 'with gil' pour éviter les problèmes GIL
sed -i 's/nogil/with gil/g' causalml/inference/tree/*.pyx

# Corriger l'indexation des tableaux dans causaltree.pyx
echo "Correction des erreurs d'accès aux tableaux..."
sed -i 's/samples\[i\]/samples.item(i)/g' causalml/inference/tree/causaltree.pyx
sed -i 's/sample_weight\[i\]/sample_weight.item(i)/g' causalml/inference/tree/causaltree.pyx

echo "Installation de causalml depuis les sources corrigées..."
pip install -e .

# Vérifier l'installation
if pip show causalml > /dev/null 2>&1; then
    echo "✅ causalml installé avec succès"
    pip show causalml | grep Version
    # Nettoyage
    cd /
    rm -rf "$TEMP_DIR"
    exit 0
else
    echo "❌ L'installation a échoué. Tentative alternative avec pip..."
    
    # Tentative avec une version spécifique qui pourrait fonctionner
    cd /
    rm -rf "$TEMP_DIR"
    pip install --no-cache-dir causalml==0.9.0
    
    if pip show causalml > /dev/null 2>&1; then
        echo "✅ causalml 0.9.0 installé avec succès"
        pip show causalml | grep Version
        exit 0
    else
        echo "❌ Toutes les tentatives ont échoué."
        echo "Création d'un package factice pour éviter les erreurs d'importation..."
        
        # Créer un package factice
        SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
        mkdir -p "$SITE_PACKAGES/causalml"
        cat > "$SITE_PACKAGES/causalml/__init__.py" << EOF
# Package causalml factice
import warnings
warnings.warn("Ce package causalml est un substitut factice. Les fonctionnalités d'inférence causale ne sont pas disponibles.")

# Classes factices pour éviter les erreurs d'importation
class UpliftTreeClassifier:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("causalml n'a pas pu être installé correctement.")

class UpliftRandomForestClassifier:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("causalml n'a pas pu être installé correctement.")
EOF
        
        echo "📝 Package factice créé avec avertissement pour éviter les erreurs silencieuses."
        exit 1
    fi
fi 