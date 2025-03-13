#!/bin/bash
set -e

# Création des répertoires nécessaires
mkdir -p data logs saved_models

# Affichage des informations
echo "=== Evil2Root Trading Bot - Démarrage Docker ==="
echo "Date de démarrage: $(date)"
echo "Environnement: $FLASK_ENV"

# Fonction pour démarrer le scheduler d'analyse en arrière-plan
start_market_scheduler() {
    echo "Démarrage du scheduler d'analyse de marché..."
    python -c "from app.market_analysis_scheduler import run_market_analysis_scheduler; run_market_analysis_scheduler()" &
    SCHEDULER_PID=$!
    echo "Scheduler démarré avec PID: $SCHEDULER_PID"
    echo $SCHEDULER_PID > /app/.scheduler_pid
}

# Détection du mode de fonctionnement
if [ "$1" = "scheduler" ]; then
    echo "Mode scheduler activé"
    exec python -c "from app.market_analysis_scheduler import run_market_analysis_scheduler; run_market_analysis_scheduler()"
elif [ "$1" = "web-with-scheduler" ]; then
    echo "Mode web avec scheduler activé"
    # Démarrage du scheduler en arrière-plan
    start_market_scheduler
    # Démarrage de l'application web
    exec gunicorn run:app --bind=0.0.0.0:5000
elif [ "$1" = "analysis-bot" ]; then
    echo "Mode analysis-bot activé"
    exec python start_daily_analysis.py
elif [ "$1" = "train-and-analyze" ]; then
    echo "Mode entraînement forcé et analyse activé"
    # Lancer le script avec l'option --force-train
    exec python start_daily_analysis.py --force-train
elif [ "$1" = "scheduler-force-train" ]; then
    echo "Mode scheduler avec entraînement forcé activé"
    # Définir la variable d'environnement pour forcer l'entraînement
    export FORCE_MODEL_TRAINING=true
    exec python -c "from app.market_analysis_scheduler import run_market_analysis_scheduler; run_market_analysis_scheduler()"
else
    # Mode par défaut: application web uniquement
    echo "Mode par défaut (web) activé"
    exec "$@"
fi 