#!/bin/bash
# Script pour préparer l'environnement Docker en copiant les scripts nécessaires
# vers le répertoire racine de l'application

set -e

echo "Préparation de l'environnement pour Docker..."

# Détecter le répertoire racine du projet
if [ -d "/app" ]; then
    # En contexte Docker
    ROOT_DIR="/app"
else
    # En contexte local
    ROOT_DIR="$(pwd)/.."
fi

echo "Utilisation du répertoire racine: $ROOT_DIR"

# Créer les répertoires nécessaires s'ils n'existent pas
mkdir -p "$ROOT_DIR/data" "$ROOT_DIR/logs" "$ROOT_DIR/saved_models" || echo "Impossible de créer les répertoires. Cela est normal si exécuté pendant le build."

# Détecter si nous sommes dans le répertoire docker ou dans le répertoire racine
if [ -d "./scripts" ]; then
    # Nous sommes dans le répertoire racine
    SCRIPTS_DIR="./scripts/shell"
    TARGET_DIR="./"
elif [ -d "../scripts" ]; then
    # Nous sommes dans le répertoire docker
    SCRIPTS_DIR="../scripts/shell"
    TARGET_DIR="../"
else
    echo "ERREUR: Structure de répertoire inattendue"
    exit 1
fi

echo "Répertoire des scripts: $SCRIPTS_DIR"
echo "Répertoire cible: $TARGET_DIR"

# Copier les scripts depuis leur emplacement d'origine vers la racine
cp -v "$SCRIPTS_DIR/docker-entrypoint.sh" "$TARGET_DIR" || echo "Échec de copie: docker-entrypoint.sh"
cp -v "$SCRIPTS_DIR/start_market_scheduler.sh" "$TARGET_DIR" || echo "Échec de copie: start_market_scheduler.sh"
cp -v "$SCRIPTS_DIR/stop_market_scheduler.sh" "$TARGET_DIR" || echo "Échec de copie: stop_market_scheduler.sh"
cp -v "$SCRIPTS_DIR/analysis-bot-entrypoint.sh" "$TARGET_DIR" || echo "Échec de copie: analysis-bot-entrypoint.sh"

# Chercher le script start_daily_analysis.py dans tout le projet
SCRIPT_PATH=$(find "$TARGET_DIR" -name "start_daily_analysis.py" -type f | head -n 1)

if [ -n "$SCRIPT_PATH" ]; then
    echo "Script start_daily_analysis.py trouvé à l'emplacement: $SCRIPT_PATH"
    cp -v "$SCRIPT_PATH" "$TARGET_DIR" || echo "Échec de copie: start_daily_analysis.py"
else
    echo "ATTENTION: Le script start_daily_analysis.py n'a pas été trouvé!"
    echo "Création d'un script temporaire..."
    
    # Créer un script temporaire qui affiche un message d'erreur
    cat > "$TARGET_DIR/start_daily_analysis.py" << 'EOF'
#!/usr/bin/env python3
import sys
import os

print("ERREUR: Script start_daily_analysis.py non trouvé!")
print("Ce script doit être ajouté au projet.")
print("Arguments passés:", sys.argv[1:])
sys.exit(1)
EOF
fi

# Rendre tous les scripts exécutables
chmod +x "$TARGET_DIR/docker-entrypoint.sh" \
    "$TARGET_DIR/start_market_scheduler.sh" \
    "$TARGET_DIR/stop_market_scheduler.sh" \
    "$TARGET_DIR/analysis-bot-entrypoint.sh" \
    "$TARGET_DIR/start_daily_analysis.py" || echo "Échec d'attribution des permissions. Cela peut être normal lors du build."

echo "Environnement préparé avec succès!" 