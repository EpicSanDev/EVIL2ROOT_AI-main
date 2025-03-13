#!/bin/bash
set -e

echo "=== Evil2Root Trading Bot - Lancement du bot d'analyse ==="
echo "Date de démarrage: $(date)"

# Création des répertoires nécessaires
mkdir -p /app/data /app/logs /app/saved_models

# Exécution du script d'analyse
exec python /app/start_daily_analysis.py "$@" 