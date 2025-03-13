#!/bin/bash
# Script pour préparer l'environnement Docker en copiant les scripts nécessaires
# vers le répertoire racine de l'application

set -e

echo "Préparation de l'environnement pour Docker..."

# Créer les répertoires nécessaires s'ils n'existent pas
mkdir -p /app/data /app/logs /app/saved_models

# Copier les scripts depuis leur emplacement d'origine vers la racine
cp -v scripts/shell/docker-entrypoint.sh ./
cp -v scripts/shell/start_market_scheduler.sh ./
cp -v scripts/shell/stop_market_scheduler.sh ./
cp -v scripts/shell/analysis-bot-entrypoint.sh ./

# Chercher le script start_daily_analysis.py dans tout le projet
SCRIPT_PATH=$(find . -name "start_daily_analysis.py" -type f | head -n 1)

if [ -n "$SCRIPT_PATH" ]; then
    echo "Script start_daily_analysis.py trouvé à l'emplacement: $SCRIPT_PATH"
    cp -v "$SCRIPT_PATH" ./
else
    echo "ATTENTION: Le script start_daily_analysis.py n'a pas été trouvé!"
    echo "Création d'un script temporaire..."
    
    # Créer un script temporaire qui affiche un message d'erreur
    cat > ./start_daily_analysis.py << 'EOF'
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
chmod +x docker-entrypoint.sh \
    start_market_scheduler.sh \
    stop_market_scheduler.sh \
    analysis-bot-entrypoint.sh \
    start_daily_analysis.py

echo "Environnement préparé avec succès!" 