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

# Afficher la structure actuelle des répertoires pour debug
echo "Structure des répertoires:"
ls -la "$ROOT_DIR"
if [ -d "$ROOT_DIR/scripts" ]; then
    echo "Contenu du répertoire scripts:"
    ls -la "$ROOT_DIR/scripts"
    
    if [ -d "$ROOT_DIR/scripts/shell" ]; then
        echo "Contenu du répertoire scripts/shell:"
        ls -la "$ROOT_DIR/scripts/shell"
    else
        echo "ATTENTION: Le répertoire scripts/shell n'existe pas!"
    fi
else
    echo "ATTENTION: Le répertoire scripts n'existe pas!"
fi

# Définir les emplacements source et cible des scripts
SCRIPTS_SHELL_DIR="$ROOT_DIR/scripts/shell"
TARGET_DIR="$ROOT_DIR"

echo "Répertoire des scripts shell: $SCRIPTS_SHELL_DIR"
echo "Répertoire cible: $TARGET_DIR"

# Vérifier l'existence du répertoire des scripts
if [ ! -d "$SCRIPTS_SHELL_DIR" ]; then
    echo "ERREUR: Le répertoire $SCRIPTS_SHELL_DIR n'existe pas!"
    echo "Vérification d'autres emplacements possibles..."
    
    # Vérifier si les scripts sont peut-être déjà à la racine
    if [ -f "$TARGET_DIR/docker-entrypoint.sh" ] && [ -f "$TARGET_DIR/start_market_scheduler.sh" ]; then
        echo "Les scripts semblent déjà être présents à la racine. Aucune copie nécessaire."
        # Rendre les scripts exécutables
        chmod +x "$TARGET_DIR/docker-entrypoint.sh" \
            "$TARGET_DIR/start_market_scheduler.sh" \
            "$TARGET_DIR/stop_market_scheduler.sh" \
            "$TARGET_DIR/analysis-bot-entrypoint.sh" \
            "$TARGET_DIR/start_daily_analysis.py" 2>/dev/null || echo "Impossible de rendre certains scripts exécutables"
        exit 0
    else
        echo "ERREUR: Impossible de trouver les scripts nécessaires!"
        exit 1
    fi
fi

# Copier les scripts depuis leur emplacement d'origine vers la racine
for script in "docker-entrypoint.sh" "start_market_scheduler.sh" "stop_market_scheduler.sh" "analysis-bot-entrypoint.sh"; do
    if [ -f "$SCRIPTS_SHELL_DIR/$script" ]; then
        echo "Copie de $script..."
        cp -v "$SCRIPTS_SHELL_DIR/$script" "$TARGET_DIR/" || echo "Échec de copie: $script"
    else
        echo "ATTENTION: Le script $script n'existe pas dans $SCRIPTS_SHELL_DIR"
    fi
done

# Chercher le script start_daily_analysis.py dans tout le projet
SCRIPT_PATH=$(find "$ROOT_DIR" -name "start_daily_analysis.py" -type f | head -n 1)

if [ -n "$SCRIPT_PATH" ]; then
    echo "Script start_daily_analysis.py trouvé à l'emplacement: $SCRIPT_PATH"
    # Vérifier si le chemin trouvé est différent du chemin cible avant de copier
    TARGET_PATH="$TARGET_DIR/start_daily_analysis.py"
    if [ "$SCRIPT_PATH" != "$TARGET_PATH" ] && [ "$(realpath $SCRIPT_PATH 2>/dev/null || echo $SCRIPT_PATH)" != "$(realpath $TARGET_PATH 2>/dev/null || echo $TARGET_PATH)" ]; then
        cp -v "$SCRIPT_PATH" "$TARGET_DIR" || echo "Échec de copie: start_daily_analysis.py"
    else
        echo "Pas besoin de copier start_daily_analysis.py car il est déjà à l'emplacement cible"
    fi
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
for script in "docker-entrypoint.sh" "start_market_scheduler.sh" "stop_market_scheduler.sh" "analysis-bot-entrypoint.sh" "start_daily_analysis.py"; do
    if [ -f "$TARGET_DIR/$script" ]; then
        echo "Attribution des permissions d'exécution à $script..."
        chmod +x "$TARGET_DIR/$script" || echo "Échec d'attribution des permissions pour $script"
    else
        echo "ATTENTION: Impossible de trouver $script pour lui attribuer des permissions d'exécution"
    fi
done

echo "Environnement préparé avec succès!" 