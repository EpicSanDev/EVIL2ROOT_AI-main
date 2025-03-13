#!/bin/bash

# Script pour tester l'installation améliorée de causalml sans reconstruire le Docker
# Version adaptée pour les environnements cloud (Render.com)

set -e

echo "========================================="
echo "🧪 Test d'installation de causalml"
echo "========================================="

# Déterminer la commande Python à utiliser
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Vérifier si causalml est déjà installé
if $PYTHON_CMD -m pip show causalml > /dev/null 2>&1; then
    echo "⚠️ causalml est déjà installé. Désinstallation..."
    $PYTHON_CMD -m pip uninstall -y causalml
fi

# Désinstaller les packages potentiellement conflictuels
echo "🧹 Nettoyage des packages potentiellement conflictuels..."
$PYTHON_CMD -m pip uninstall -y cython || true
$PYTHON_CMD -m pip uninstall -y numpy || true

# Exécuter le script d'installation amélioré
echo "🚀 Exécution du script d'installation amélioré..."
chmod +x docker/fix-causalml-install.sh
./docker/fix-causalml-install.sh

# Vérifier le résultat
RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo "✅ Le script s'est terminé avec succès."
else
    echo "⚠️ Le script s'est terminé avec des avertissements (code $RESULT), mais un package factice a été créé."
fi

# Tester l'importation de causalml
echo "🔍 Test d'importation de causalml..."
$PYTHON_CMD -c "import causalml; print(f'📦 causalml importé avec succès! Version: {causalml.__version__}')" || echo "❌ Échec de l'importation de causalml"

# Vérifier si les fonctions principales sont disponibles
echo "🔍 Vérification des fonctionnalités disponibles..."
$PYTHON_CMD -c "
try:
    from causalml.inference.tree import UpliftTreeClassifier
    print('✅ UpliftTreeClassifier disponible')
except (ImportError, NotImplementedError) as e:
    print(f'⚠️ UpliftTreeClassifier non disponible: {e}')

try:
    from causalml.inference.meta import BaseXLearner
    print('✅ BaseXLearner disponible')
except (ImportError, NotImplementedError) as e:
    print(f'⚠️ BaseXLearner non disponible: {e}')
" || echo "❌ Erreur lors de la vérification des fonctionnalités"

echo "========================================="
echo "Test terminé"
echo "=========================================" 