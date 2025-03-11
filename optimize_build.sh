#!/bin/bash

# Script d'optimisation pour le build Docker d'EVIL2ROOT Trading Bot
# Ce script optimise l'installation des dépendances, notamment Plotly

set -e

echo "🚀 Démarrage de l'optimisation du build..."

# 1. Vérifier si des caches Docker existent déjà et les nettoyer si nécessaire
echo "🧹 Nettoyage des caches Docker inutilisés..."
docker system prune -f --filter "until=24h"

# 2. Créer un dossier pour le cache pip si nécessaire
mkdir -p .docker-cache/pip

# 3. Pré-télécharger plotly et dash sans leurs dépendances
echo "📦 Pré-téléchargement des packages Plotly et Dash..."
pip download --no-deps -d .docker-cache/pip plotly==5.14.1 dash==2.10.0

# 4. Optimiser le fichier Dockerfile temporairement pour le build
echo "⚙️ Configuration du build pour utiliser le cache local..."
if [ -f Dockerfile ]; then
    # Sauvegarde du Dockerfile original
    cp Dockerfile Dockerfile.bak
    
    # Modification pour utiliser le cache local
    sed -i.tmp 's|pip install --no-cache-dir plotly==5.14.1 --no-deps|pip install --find-links=./.docker-cache/pip plotly==5.14.1 --no-deps|g' Dockerfile
    sed -i.tmp 's|pip install --no-cache-dir dash==2.10.0 --no-deps|pip install --find-links=./.docker-cache/pip dash==2.10.0 --no-deps|g' Dockerfile
    rm -f Dockerfile.tmp
fi

# 5. Créer un .dockerignore temporaire optimisé
echo "📝 Configuration de .dockerignore pour exclure les fichiers inutiles..."
cat << EOF > .dockerignore.build
__pycache__
*.py[cod]
*$py.class
*.so
.git
.pytest_cache
.coverage
htmlcov
.env
.venv
env/
venv/
ENV/
data/
logs/
*.log
node_modules/
tests/
docs/
*.md
*.txt.bak
*.ipynb
.DS_Store
EOF

# Copier le fichier .dockerignore.build vers .dockerignore s'il existe
if [ -f .dockerignore ]; then
    cp .dockerignore .dockerignore.bak
    cp .dockerignore.build .dockerignore
fi

# 6. Lancer le build avec les optimisations
echo "🏗️ Lancement du build avec optimisations..."
DOCKER_BUILDKIT=1 docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from evil2root/trading-bot:latest \
  -t evil2root/trading-bot:latest \
  --progress=plain \
  .

# 7. Restaurer les fichiers originaux
echo "🔄 Restauration des fichiers originaux..."
if [ -f Dockerfile.bak ]; then
    mv Dockerfile.bak Dockerfile
fi

if [ -f .dockerignore.bak ]; then
    mv .dockerignore.bak .dockerignore
    rm -f .dockerignore.build
fi

echo "✅ Build optimisé terminé !"
echo "⏱️ Le temps de build devrait être considérablement réduit, en particulier pour l'étape 'web (4/6)' liée à Plotly."

exit 0 