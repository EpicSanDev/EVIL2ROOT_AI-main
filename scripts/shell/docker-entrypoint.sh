#!/bin/bash
set -e

# Création des répertoires nécessaires
mkdir -p data logs saved_models

# Affichage des informations
echo "=== Evil2Root Trading Bot - Démarrage Docker ==="
echo "Date de démarrage: $(date)"
echo "Environnement: $FLASK_ENV"

# Fonction pour charger les secrets depuis les fichiers Docker Secrets
load_secrets() {
    echo "Chargement des secrets Docker..."
    
    # Fonction pour charger un secret
    load_secret() {
        local file_var="$1_FILE"
        local secret_file="${!file_var}"
        
        if [ -n "$secret_file" ] && [ -f "$secret_file" ]; then
            local secret_value=$(cat "$secret_file")
            export "$1"="$secret_value"
            echo "Secret $1 chargé à partir de $secret_file"
        fi
    }
    
    # Chargement des secrets spécifiques
    load_secret "DB_USER"
    load_secret "DB_PASSWORD"
    load_secret "SECRET_KEY"
    load_secret "ADMIN_PASSWORD"
    load_secret "TELEGRAM_TOKEN"
    load_secret "FINNHUB_API_KEY"
    load_secret "OPENROUTER_API_KEY"
    load_secret "COINBASE_API_KEY"
    load_secret "COINBASE_WEBHOOK_SECRET"
    
    # Construction dynamique de DATABASE_URI avec les identifiants chargés des secrets
    if [ -n "$DB_USER" ] && [ -n "$DB_PASSWORD" ]; then
        export DATABASE_URI="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST:-db}:${DB_PORT:-5432}/${DB_NAME:-trading_db}"
        echo "DATABASE_URI mise à jour avec les secrets"
    fi
}

# Charger les secrets au démarrage
load_secrets

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